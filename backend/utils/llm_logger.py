# backend/utils/llm_logger.py
"""Логирование вызовов LLM в таблицу llm_calls."""

import hashlib
import logging
import time
from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import LLMCall

logger = logging.getLogger(__name__)


def _hash(text: str) -> str:
    """SHA256-хэш текста (для промптов/ответов — не храним сами тексты)."""
    if not text:
        return ""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


async def log_llm_call(
    db: AsyncSession,
    *,
    model_name: str,
    provider: str,
    agent_name: Optional[str] = None,
    request_id: Optional[str] = None,
    prompt: Optional[str] = None,
    response: Optional[str] = None,
    parameters: Optional[str] = None,
    tokens_used: Optional[int] = None,
    latency_ms: Optional[int] = None,
    status: str = "success",
    error_message: Optional[str] = None,
) -> None:
    """
    Записывает вызов LLM в БД.

    ВАЖНО: тексты промптов/ответов не сохраняются — только хэши.
    """
    call = LLMCall(
        model_name=model_name,
        provider=provider,
        agent_name=agent_name,
        request_id=request_id,
        prompt_hash=_hash(prompt) if prompt else None,
        response_hash=_hash(response) if response else None,
        parameters=parameters,
        tokens_used=tokens_used,
        latency_ms=latency_ms,
        status=status,
        error_message=error_message,
    )
    db.add(call)
    try:
        await db.flush()
    except Exception as e:
        logger.warning("Failed to log LLM call: %s", e)