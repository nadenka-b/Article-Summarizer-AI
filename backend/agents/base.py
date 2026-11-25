# backend/agents/base.py
"""Базовый класс для всех агентов."""

import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict

from backend.utils.async_utils import retry

logger = logging.getLogger(__name__)


class Agent(ABC):
    """Абстрактный агент. Наследники реализуют process(state)."""

    name: str = "agent"

    def __init__(self) -> None:
        self._log = logging.getLogger(f"agent.{self.name}")
        from backend.utils.llm_client import get_llm_client
        self._llm = get_llm_client()

    @property
    def _model_override(self):
        """
        Пер-агентное переопределение модели (этап 8, тюнинг движка).

        Имя агента маппится на настройку {NAME}_MODEL:
        summarizer → SUMMARIZER_MODEL, critic → CRITIC_MODEL и т.д.
        None → используется LMSTUDIO_MODEL.
        """
        from backend.core.config import get_settings

        return getattr(get_settings(), f"{self.name.upper()}_MODEL", None)

    @retry(max_attempts=3, delay=1.0)
    async def _call_llm(self, prompt, model=None, temperature=0.3, task="generate", db=None, request_id=None):
        from backend.utils.llm_logger import log_llm_call

        model = model or self._model_override
        start = time.perf_counter()
        status = "success"
        error_msg = None
        response_text = ""
        engine_used = "unknown"
        model_used = model or "unknown"

        try:
            response_text, engine_used, model_used = await self._llm.generate(
                prompt, model=model, temperature=temperature, task=task
            )
            return response_text
        except Exception as e:
            status = "error"
            error_msg = f"{type(e).__name__}: {e}"
            raise
        finally:
            latency_ms = int((time.perf_counter() - start) * 1000)
            # Агенты получают db/request_id из state пайплайна (self._db)
            db = db or getattr(self, "_db", None)
            request_id = request_id or getattr(self, "_request_id", None)
            if db is not None:
                await log_llm_call(
                    db,
                    model_name=model_used,
                    provider=engine_used,
                    agent_name=self.name,
                    request_id=request_id,
                    prompt=prompt,
                    response=response_text,
                    latency_ms=latency_ms,
                    status=status,
                    error_message=error_msg,
                )

    @abstractmethod
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает state и возвращает изменённый state."""
        ...

    async def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Обёртка: тайминги, ошибки, запись в reasoning_log."""
        # Трассировка llm_calls: сессия и request_id из state пайплайна
        self._db = state.get("_db")
        self._request_id = state.get("_request_id")

        start_perf = time.perf_counter()
        log_entry: Dict[str, Any] = {
            "agent": self.name,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "error": None,
        }

        try:
            self._log.info("Starting processing")
            state = await self.process(state)
            self._log.info("Finished in %.2fs", time.perf_counter() - start_perf)
        except Exception as e:
            log_entry["status"] = "error"
            log_entry["error"] = f"{type(e).__name__}: {e}"
            self._log.exception("Processing failed")
            state["last_error"] = log_entry["error"]

        log_entry["duration_sec"] = round(time.perf_counter() - start_perf, 4)
        log_entry["finished_at"] = datetime.now(timezone.utc).isoformat()

        state.setdefault("reasoning_log", []).append(log_entry)
        return state