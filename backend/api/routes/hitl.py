# backend/api/routes/hitl.py
"""
Интерфейс верификации HITL (этап 10).

Очередь резюме с низким качеством для эксперта-редактора:
- GET  /hitl/queue            — pending-записи (пагинация);
- GET  /hitl/queue/{id}       — детали с текстом статьи;
- POST /hitl/queue/{id}/confirm — правка становится финальным резюме
                                 И новой эталонной парой базы знаний;
- POST /hitl/queue/{id}/reject  — отклонение (в БЗ не попадает);
- GET  /hitl/stats            — статистика очереди.

Все эндпоинты требуют роль editor или admin.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db, require_editor
from backend.core.models import (
    HITLConfirmRequest,
    HITLDetail,
    HITLQueueItem,
    HITLRejectRequest,
    HITLReviewResponse,
    HITLStats,
)
from backend.db.models import Article, Summary, User
from backend.hitl.active_learning import ActiveLearningManager

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/hitl", tags=["hitl"])


def _example_or_404(example):
    if example is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Запись очереди не найдена",
        )
    return example


def _conflict_if_processed(example):
    if example.status != "pending":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Запись уже обработана (status={example.status})",
        )


# ============================================================================
# GET /hitl/queue
# ============================================================================
@router.get(
    "/queue",
    response_model=list[HITLQueueItem],
    summary="Очередь модерации (pending)",
)
async def get_queue(
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(require_editor),
    db: AsyncSession = Depends(get_db),
) -> list[HITLQueueItem]:
    examples = await ActiveLearningManager.get_pending_examples(
        db, limit=limit, offset=offset
    )
    return [HITLQueueItem.model_validate(e) for e in examples]


# ============================================================================
# GET /hitl/queue/{example_id}
# ============================================================================
@router.get(
    "/queue/{example_id}",
    response_model=HITLDetail,
    summary="Детали записи с текстом статьи",
)
async def get_example_detail(
    example_id: str,
    user: User = Depends(require_editor),
    db: AsyncSession = Depends(get_db),
) -> HITLDetail:
    example = _example_or_404(
        await ActiveLearningManager.get_example(db, example_id)
    )

    article_title = None
    article_text = None
    summary = await db.get(Summary, example.summary_id)
    if summary is not None:
        article = await db.get(Article, summary.article_id)
        if article is not None:
            article_title = article.title
            article_text = article.full_text

    detail = HITLDetail.model_validate(example)
    detail.article_title = article_title
    detail.article_text = article_text
    return detail


# ============================================================================
# POST /hitl/queue/{example_id}/confirm
# ============================================================================
@router.post(
    "/queue/{example_id}/confirm",
    response_model=HITLReviewResponse,
    summary="Подтвердить правку (резюме + новая пара БЗ)",
)
async def confirm_example(
    example_id: str,
    payload: HITLConfirmRequest,
    user: User = Depends(require_editor),
    db: AsyncSession = Depends(get_db),
) -> HITLReviewResponse:
    example = _example_or_404(
        await ActiveLearningManager.get_example(db, example_id)
    )
    _conflict_if_processed(example)

    try:
        outcome = await ActiveLearningManager.confirm_correction(
            db,
            example_id=example_id,
            corrected_summary=payload.corrected_summary,
            reviewer_id=user.id,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        )

    kb_inserted = outcome["kb_inserted"]

    # После подтверждения — фоновая переиндексация RAG (этап 11).
    # Redis недоступен → warning, подтверждение это не ломает.
    if kb_inserted:
        try:
            from backend.worker.tasks import reindex_knowledge_base

            reindex_knowledge_base.delay()
        except Exception as e:
            logger.warning(
                "Не удалось поставить задачу переиндексации "
                "(Redis недоступен?): %s", e,
            )

    return HITLReviewResponse(
        id=example_id,
        status="confirmed",
        kb_inserted=kb_inserted,
        message=(
            "Резюме подтверждено и добавлено в базу знаний как эталон"
            if kb_inserted
            else "Резюме подтверждено (пара уже была в базе знаний)"
        ),
    )


# ============================================================================
# POST /hitl/queue/{example_id}/reject
# ============================================================================
@router.post(
    "/queue/{example_id}/reject",
    response_model=HITLReviewResponse,
    summary="Отклонить (в базу знаний не попадает)",
)
async def reject_example(
    example_id: str,
    payload: HITLRejectRequest | None = None,
    user: User = Depends(require_editor),
    db: AsyncSession = Depends(get_db),
) -> HITLReviewResponse:
    example = _example_or_404(
        await ActiveLearningManager.get_example(db, example_id)
    )
    _conflict_if_processed(example)

    try:
        await ActiveLearningManager.reject_example(
            db,
            example_id=example_id,
            reviewer_id=user.id,
            reason=payload.reason if payload else None,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(e)
        )

    return HITLReviewResponse(
        id=example_id,
        status="rejected",
        message="Резюме отклонено, в базу знаний не попало",
    )


# ============================================================================
# GET /hitl/stats
# ============================================================================
@router.get(
    "/stats",
    response_model=HITLStats,
    summary="Статистика очереди",
)
async def get_stats(
    user: User = Depends(require_editor),
    db: AsyncSession = Depends(get_db),
) -> HITLStats:
    counts = await ActiveLearningManager.get_stats(db)
    return HITLStats(**counts)
