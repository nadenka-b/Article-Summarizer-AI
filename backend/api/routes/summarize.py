# backend/api/routes/summarize.py
"""
Эндпоинты суммаризации (этап 9).

- POST /summarize/text  — резюме сырого текста;
- POST /summarize/pdf   — загрузка PDF-файла;
- POST /summarize/arxiv — статья с arXiv по идентификатору;
- GET  /summaries/{id}  — резюме из БД (владелец или админ).

Вся логика — в backend/services/pipeline.py, роуты только принимают
запрос и отдают результат.
"""

import logging
from dataclasses import asdict
from typing import Optional

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    status,
    UploadFile,
)
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.core.models import (
    SummarizeArxivRequest,
    SummarizeTextRequest,
    SummaryOut,
    SummaryResponse,
)
from backend.db.models import Article, Summary, User
from backend.services.pipeline import safe_perform_summarization

logger = logging.getLogger(__name__)

router = APIRouter(tags=["summarize"])
summaries_router = APIRouter(prefix="/summaries", tags=["summarize"])

MAX_PDF_SIZE = 20 * 1024 * 1024  # 20 МБ


def _to_response(result) -> SummaryResponse:
    return SummaryResponse(**asdict(result))


def _raise_on_error(result) -> None:
    if result.error:
        code = (
            status.HTTP_502_BAD_GATEWAY
            if "LLM" in result.error or "недоступна" in result.error
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=code, detail=result.error)


# ============================================================================
# POST /summarize/text
# ============================================================================
@router.post(
    "/summarize/text",
    response_model=SummaryResponse,
    summary="Суммаризация текста статьи",
)
async def summarize_text(
    payload: SummarizeTextRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SummaryResponse:
    result = await safe_perform_summarization(
        user=user,
        db=db,
        mode=payload.mode,
        summary_mode=payload.summary_mode,
        article_text=payload.text,
        title=payload.title,
        field=payload.field,
        use_cache=payload.use_cache,
    )
    _raise_on_error(result)
    return _to_response(result)


# ============================================================================
# POST /summarize/pdf
# ============================================================================
@router.post(
    "/summarize/pdf",
    response_model=SummaryResponse,
    summary="Суммаризация PDF-файла",
)
async def summarize_pdf(
    file: UploadFile = File(..., description="PDF-файл статьи"),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    mode: Optional[str] = None,
    summary_mode: Optional[str] = None,
    title: Optional[str] = None,
    field: Optional[str] = None,
    use_cache: bool = True,
) -> SummaryResponse:
    pdf_bytes = await file.read()

    if len(pdf_bytes) > MAX_PDF_SIZE:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Файл больше {MAX_PDF_SIZE // (1024 * 1024)} МБ",
        )

    if not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Это не PDF-файл",
        )

    result = await safe_perform_summarization(
        user=user,
        db=db,
        mode=mode,
        summary_mode=summary_mode,
        pdf_bytes=pdf_bytes,
        title=title or (file.filename or None),
        field=field,
        use_cache=use_cache,
    )
    _raise_on_error(result)
    return _to_response(result)


# ============================================================================
# POST /summarize/arxiv
# ============================================================================
@router.post(
    "/summarize/arxiv",
    response_model=SummaryResponse,
    summary="Суммаризация статьи с arXiv по ID",
)
async def summarize_arxiv(
    payload: SummarizeArxivRequest,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SummaryResponse:
    result = await safe_perform_summarization(
        user=user,
        db=db,
        mode=payload.mode,
        summary_mode=payload.summary_mode,
        arxiv_id=payload.arxiv_id,
        title=payload.title or f"arXiv:{payload.arxiv_id}",
        field=payload.field,
        use_cache=payload.use_cache,
    )
    _raise_on_error(result)
    return _to_response(result)


# ============================================================================
# GET /summaries/{id}
# ============================================================================
@summaries_router.get(
    "/{summary_id}",
    response_model=SummaryOut,
    summary="Резюме по id (владелец или админ)",
)
async def get_summary(
    summary_id: str,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> SummaryOut:
    result = await db.execute(
        select(Summary)
        .join(Article, Summary.article_id == Article.id)
        .where(Summary.id == summary_id)
    )
    summary = result.scalars().first()

    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Резюме не найдено",
        )

    article = await db.get(Article, summary.article_id)

    if article.owner_id != user.id and user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Доступ к чужому резюме запрещён",
        )

    return SummaryOut.model_validate(summary)
