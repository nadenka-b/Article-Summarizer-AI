# backend/api/routes/articles.py
"""
Роуты статей пользователя и поиска по базе знаний (этап 12, для UI).

- GET /articles/my      — статьи текущего пользователя с резюме;
- GET /kb/search        — гибридный поиск по базе знаний (демо RAG).
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.data.knowledge_base import KBManager
from backend.db.models import Article, Summary, User
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
    summary="Статьи текущего пользователя с резюме",
)
async def my_articles(
    limit: int = Query(default=50, ge=1, le=200),
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> list[dict]:
    result = await db.execute(
        select(Article)
        .where(Article.owner_id == user.id)
        .order_by(Article.created_at.desc())
        .limit(limit)
    )
    articles = result.scalars().all()

    response = []
    for article in articles:
        summaries_result = await db.execute(
            select(Summary)
            .where(Summary.article_id == article.id)
            .order_by(Summary.created_at.desc())
        )
        summaries = summaries_result.scalars().all()

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

    return response


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
