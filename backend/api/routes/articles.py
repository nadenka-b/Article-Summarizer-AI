# backend/api/routes/articles.py
"""
Роуты статей пользователя и поиска по базе знаний (этап 12, для UI).

- GET /articles/my      — статьи текущего пользователя с резюме;
- GET /kb/search        — гибридный поиск по базе знаний (демо RAG).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.api.deps import get_current_user, get_db
from backend.data.knowledge_base import KBManager
from backend.db.models import Article, User
from backend.retrieval.hybrid_retriever import hybrid_search
from backend.retrieval.vector_index import get_vector_index

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/articles", tags=["articles"])
kb_router = APIRouter(prefix="/kb", tags=["knowledge-base"])


# ============================================================================
# GET /articles/my
# ============================================================================
@router.get(
    "/my",
    summary="Статьи текущего пользователя с резюме (пагинация)",
)
async def my_articles(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Страница статей пользователя + общее число (total) по пользователю.

    total — отдельный COUNT-запрос: len(articles) отражает только размер
    текущей страницы, и на объёме больше limit счётчик в UI «замирал».
    """
    total = await db.scalar(
        select(func.count())
        .select_from(Article)
        .where(Article.owner_id == user.id)
    )

    # selectinload: резюме всех статей страницы одним запросом (было N+1 —
    # по запросу на каждую статью).
    result = await db.execute(
        select(Article)
        .options(selectinload(Article.summaries))
        .where(Article.owner_id == user.id)
        .order_by(Article.created_at.desc())
        .limit(limit)
        .offset(offset)
    )
    articles = result.scalars().all()

    response = []
    for article in articles:
        # Свежие резюме сверху — как раньше, но сортировка здесь,
        # а не в SQL (коллекцию грузит selectinload).
        summaries = sorted(
            article.summaries, key=lambda s: s.created_at, reverse=True
        )

        response.append({
            "article_id": article.id,
            "title": article.title,
            "source": article.source,
            "status": article.status,
            "char_count": article.char_count,
            "created_at": article.created_at.isoformat(),
            "summaries": [
                {
                    "summary_id": s.id,
                    "summary_mode": s.summary_mode,
                    "critic_score": s.critic_score,
                    "validation_passed": s.validation_passed,
                    "supervisor_iterations": s.supervisor_iterations,
                    "summary_text": s.summary_text,
                    "created_at": s.created_at.isoformat(),
                }
            for s in summaries
        ],
    })

    return {"total": total, "limit": limit, "offset": offset, "articles": response}


# ============================================================================
# GET /kb/search
# ============================================================================
@kb_router.get(
    "/search",
    summary="Гибридный поиск по базе знаний (BM25 + Qdrant + RRF)",
)
async def kb_search(
    q: str = Query(..., min_length=3, max_length=1000),
    top_k: int = Query(default=5, ge=1, le=20),
    field: Optional[str] = None,
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> dict:
    """Демо-эндпоинт RAG: находит похожие пары «текст → аннотация»."""
    try:
        kb_manager = KBManager()
        await kb_manager.initialize(db)
        results = hybrid_search(
            kb_manager, get_vector_index(), q, field=field, top_k=top_k
        )
    except Exception as e:
        logger.exception("kb_search failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Поиск по базе знаний недоступен: {e}",
        )

    return {
        "query": q,
        "results": [
            {
                "score": round(score, 4),
                "field": entry.field,
                "language": entry.language,
                "source": entry.source,
                # Для UI показываем аннотацию и начало исходного текста:
                "target_summary": entry.target_summary,
                "source_excerpt": entry.search_text[:500],
            }
            for entry, score in results
        ],
    }
