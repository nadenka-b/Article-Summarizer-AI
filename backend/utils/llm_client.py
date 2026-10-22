# backend/utils/llm_client.py
"""
LLMClient: LM Studio (OpenAI-compatible)

Приоритет:
1. LM Studio (если настроен и доступен)

"""

import logging
from typing import Optional

import httpx

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class LLMClient:
    """Клиент для LM Studio ."""

    def __init__(self) -> None:
        self._lmstudio_available: Optional[bool] = None


    async def _check_lmstudio(self) -> bool:
        """Проверяет доступность LM Studio."""
        if not settings.LMSTUDIO_URL:
            self._lmstudio_available = False
            return False
        if self._lmstudio_available is not None:
            return self._lmstudio_available
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                r = await client.get(settings.lmstudio_models_url)
                self._lmstudio_available = r.status_code == 200
        except Exception:
            self._lmstudio_available = False
        return self._lmstudio_available

    async def call_lmstudio(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Вызов LM Studio через /chat/completions."""
        model = model or settings.LMSTUDIO_MODEL
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                settings.lmstudio_chat_url,
                headers={"Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": temperature,
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["message"]["content"]

    
    async def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
        task: str = "generate",  # "generate" | "summarize"
    ) -> tuple[str, str, str]:
        """
        Универсальный вызов.

        Возвращает: (текст, движок, модель).
        """
        # 1. LM Studio
        if await self._check_lmstudio():
            try:
                text = await self.call_lmstudio(prompt, model, temperature)
                return text, "lmstudio", model or settings.LMSTUDIO_MODEL
            except Exception as e:
                logger.warning("LM Studio call failed: %s", e)
                self._lmstudio_available = False


        raise RuntimeError("LM Studio недоступны")


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client