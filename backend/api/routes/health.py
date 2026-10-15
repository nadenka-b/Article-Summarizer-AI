# backend/api/routes/health.py
"""
Healthcheck endpoints для Article-Summarizer-AI.

/healthz  — простая liveness-проверка, приложение запустилось.
/fields   — список научных областей и секций статьи.
"""

import logging
from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.constants import (
    RESEARCH_FIELDS,
    ARTICLE_SECTIONS,
    RELATED_FIELDS,
    EVAL_METRICS,
)
from backend.db.session import get_async_session

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """Liveness-проверка. Возвращает 200, если приложение живо."""
    return {"status": "ok"}


@router.get("/health")
async def health(db: AsyncSession = Depends(get_async_session)):
    """Readiness-проверка. Проверяет БД через SELECT 1."""
    try:
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        db_status = "unhealthy"

    return {
        "status": "ok",
        "database": db_status,
    }


@router.get("/fields")
async def fields():
    """Список научных областей, секций статей и метрик."""
    return {
        "fields": RESEARCH_FIELDS,
        "sections": ARTICLE_SECTIONS,
        "related_fields": RELATED_FIELDS,
        "eval_metrics": EVAL_METRICS,
    }