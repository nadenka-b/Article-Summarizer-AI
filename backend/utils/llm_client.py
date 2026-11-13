# backend/utils/llm_client.py
"""
LLMClient: LM Studio (OpenAI-совместимый API).

Приоритет:
1. LM Studio (если настроен и доступен).

httpx-клиенты кэшируются на event loop (клиент привязан к loops при
первом запросе), переиспользуются между вызовами и закрываются через
close_llm_clients() при остановке приложения.
"""

import asyncio
import logging
import re
import weakref
from typing import Optional

import httpx

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Пиннинг seed для воспроизводимости генерации (как в методичке этапа 5).
LLM_SEED = 42

# Один клиент на event loop: ключ — слабая ссылка на loop.
_clients: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient]" = (
    weakref.WeakKeyDictionary()
)


def _get_client() -> httpx.AsyncClient:
    """Возвращает httpx.AsyncClient для текущего event loop."""
    loop = asyncio.get_running_loop()
    client = _clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=300.0)
        _clients[loop] = client
    return client


async def close_llm_clients() -> None:
    """Закрывает все кэшированные httpx-клиенты (вызывать при остановке)."""
    for client in _clients.values():
        try:
            await client.aclose()
        except Exception as e:
            logger.warning("Failed to close httpx client: %s", e)
    _clients.clear()


class LLMClient:
    """Клиент для LM Studio."""

    def __init__(self) -> None:
        self._lmstudio_available: Optional[bool] = None

    async def _check_lmstudio(self, force: bool = False) -> bool:
        """Проверяет доступность LM Studio.

        force=True — свежая проверка мимо кэша (для /health).
        """
        if not settings.LMSTUDIO_URL:
            self._lmstudio_available = False
            return False
        if not force and self._lmstudio_available is not None:
            return self._lmstudio_available
        try:
            client = _get_client()
            r = await client.get(settings.lmstudio_models_url, timeout=3.0)
            self._lmstudio_available = r.status_code == 200
        except Exception:
            self._lmstudio_available = False
        return self._lmstudio_available

    async def probe(self) -> bool:
        """Свежая проверка доступности движка (используется /health)."""
        return await self._check_lmstudio(force=True)

    async def call_lmstudio(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Вызов LM Studio через /chat/completions."""
        model = model or settings.LMSTUDIO_MODEL
        client = _get_client()
        response = await client.post(
            settings.lmstudio_chat_url,
            headers={"Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": temperature,
                "seed": LLM_SEED,
                "stream": False,
                # Отключаем «размышления» thinking-моделей (Qwen3+):
                # без этого ответ может целиком уйти в reasoning_content,
                # а content останется пустым. Модели без thinking игнорируют.
                "chat_template_kwargs": {"enable_thinking": False},
                # Страховка от бесконечной генерации:
                "max_tokens": settings.LLM_MAX_TOKENS,
            },
            timeout=settings.LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        content = self._extract_content(message)

        if not content or not str(content).strip():
            # У thinking-моделей (Qwen3.5+) ответ может уходить в
            # reasoning_content, а content остаётся пустым.
            raise RuntimeError(
                "Модель вернула пустой ответ (content пуст; ответ ушёл в "
                "reasoning_content — используйте не-thinking модель "
                "(напр. Qwen2.5-7B-Instruct) или отключите reasoning "
                "в настройках LM Studio)"
            )
        return str(content)

    @staticmethod
    def _extract_content(message: dict) -> Optional[str]:
        """
        Достаёт ответ из сообщения, отбрасывая размышления.

        Некоторые серверы возвращают блок <think>...</think> прямо в
        content — вырезаем его. reasoning_content игнорируем:
        устойчивого способа достать финальный текст из цепочки
        рассуждений нет.
        """
        content = message.get("content")
        if not content:
            return content
        text = str(content)
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
        # Незакрытый блок (<think> без </think>) — берём хвост после него.
        if "<think>" in text and "</think>" not in text:
            text = text.split("<think>", 1)[1]
        return text.strip()

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
                # Ошибка вызова ≠ сервер недоступен: кэш доступности
                # не отравляем, чтобы retry мог попробовать снова.
                logger.warning("LM Studio call failed: %s", e)
                raise

        raise RuntimeError("LM Studio недоступен")


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
