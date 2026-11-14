# backend/api/routes/health.py
"""
Healthcheck endpoints для Article-Summarizer-AI.

/healthz  — простая liveness-проверка, приложение запустилось.
/health   — readiness-проверка: БД + LLM-движок (LM Studio);
            503, если хотя бы один сервис нездоров.
/fields   — список научных областей и секций статьи.
"""

import logging
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.core.constants import (
    RESEARCH_FIELDS,
    ARTICLE_SECTIONS,
    RELATED_FIELDS,
    EVAL_METRICS,
)
from backend.db.session import get_async_session
from backend.utils.llm_client import get_llm_client

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """Liveness-проверка. Возвращает 200, если приложение живо."""
    return {"status": "ok"}


@router.get("/health")
async def health(db: AsyncSession = Depends(get_async_session)):
    """Readiness-проверка: БД (SELECT 1) + LM Studio (GET /models)."""
    try:
        await db.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        logger.error("Database health check failed: %s", e)
        db_status = "unhealthy"

    try:
        lmstudio_ok = await get_llm_client().probe()
        llm_status = "healthy" if lmstudio_ok else "unreachable"
    except Exception as e:
        logger.error("LLM engine health check failed: %s", e)
        llm_status = "unreachable"

    healthy = db_status == "healthy" and llm_status == "healthy"

    body = {
        "status": "ok" if healthy else "degraded",
        "database": db_status,
        "llm_engine": llm_status,
        "llm_engine_name": get_settings().LLM_ENGINE,
    }
    if not healthy:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=body)
    return body


@router.get("/fields")
async def fields():
    """Список научных областей, секций статей и метрик."""
    return {
        "fields": RESEARCH_FIELDS,
        "sections": ARTICLE_SECTIONS,
        "related_fields": RELATED_FIELDS,
        "eval_metrics": EVAL_METRICS,
    }