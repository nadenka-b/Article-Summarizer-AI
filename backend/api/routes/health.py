# backend/api/routes/health.py
"""
Healthcheck endpoints для Article-Summarizer-AI.

/healthz  — простая liveness-проверка, приложение запустилось.
/fields   — список научных областей и секций статьи.
"""

import logging
from fastapi import APIRouter

from backend.core.constants import (
    RESEARCH_FIELDS,
    ARTICLE_SECTIONS,
    RELATED_FIELDS,
    EVAL_METRICS,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["health"])


@router.get("/healthz")
async def healthz():
    """Liveness-проверка. Возвращает 200, если приложение живо."""
    return {"status": "ok"}


@router.get("/fields")
async def fields():
    """Список научных областей, секций статей и метрик."""
    return {
        "fields": RESEARCH_FIELDS,
        "sections": ARTICLE_SECTIONS,
        "related_fields": RELATED_FIELDS,
        "eval_metrics": EVAL_METRICS,
    }