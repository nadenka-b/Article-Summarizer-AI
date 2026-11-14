# backend/utils/llm_client.py
"""
LLMClient: OpenAI-совместимый клиент с выбором движка (этап 8).

Движок задаётся LLM_ENGINE в .env:
- lmstudio  — локальный LM Studio на хосте (LMSTUDIO_URL, без ключа);
- openrouter — облачный агрегатор OpenRouter (OPENROUTER_BASE_URL,
              OPENROUTER_API_KEY; платный, модель OPENROUTER_MODEL).

Оба говорят по одному протоколу (/models, /chat/completions), отличается
только базовый URL, модель по умолчанию и заголовок авторизации.

httpx-клиенты кэшируются на event loop (клиент привязан к loops при
первом запросе), переиспользуются между вызовами и закрываются через
close_llm_clients() при остановке приложения.
"""

import asyncio
import logging
import re
import time
import weakref
from typing import Optional

import httpx

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Пиннинг seed для воспроизводимости генерации (как в методичке этапа 5).
LLM_SEED = 42

# Probe доступности (паттерн этапа 8): лёгкий GET /models, результат
# кэшируется на 30 с — 100+ запросов без повторного probe; при обрыве
# связи «фантомная доступность» живёт не дольше TTL.
_PROBE_TIMEOUT = 3.0
_PROBE_TTL = 30.0

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
    """Клиент активного LLM-движка (LM Studio или OpenRouter)."""

    def __init__(self) -> None:
        self._available: Optional[bool] = None
        self._probe_time = 0.0

    # ------------------------------------------------------------------
    # Доступность (probe GET /models с TTL-кэшем)
    # ------------------------------------------------------------------
    async def _check_engine(self, force: bool = False) -> bool:
        """Проверяет доступность движка; force — свежая проверка мимо кэша."""
        now = time.monotonic()
        if (
            not force
            and self._available is not None
            and now - self._probe_time < _PROBE_TTL
        ):
            return self._available

        try:
            client = _get_client()
            r = await client.get(
                f"{settings.llm_base_url}/models",
                timeout=_PROBE_TIMEOUT,
                headers=self._headers(),
            )
            self._available = r.status_code == 200
        except Exception:
            self._available = False

        self._probe_time = now
        return self._available

    async def probe(self) -> bool:
        """Свежая проверка доступности движка (используется /health)."""
        return await self._check_engine(force=True)

    # ------------------------------------------------------------------
    # Заголовки и тело запроса
    # ------------------------------------------------------------------
    @staticmethod
    def _headers() -> dict:
        headers = {"Content-Type": "application/json"}
        if settings.LLM_ENGINE == "openrouter":
            if not settings.OPENROUTER_API_KEY:
                logger.warning(
                    "LLM_ENGINE=openrouter, но OPENROUTER_API_KEY не задан"
                )
            else:
                headers["Authorization"] = (
                    f"Bearer {settings.OPENROUTER_API_KEY}"
                )
            # Рекомендация OpenRouter: идентификация приложения для дашборда.
            headers["X-Title"] = "Article-Summarizer-AI"
        return headers

    @staticmethod
    def _request_body(
        model: str, prompt: str, temperature: float
    ) -> dict:
        body: dict = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": LLM_SEED,
            "stream": False,
            # Страховка от бесконечной генерации:
            "max_tokens": settings.LLM_MAX_TOKENS,
        }
        if settings.LLM_ENGINE == "lmstudio":
            # Отключаем «размышления» thinking-моделей (Qwen3+) — это
            # параметр локального vLLM/LM Studio-рантайма. Облаку его не
            # шлём: провайдеры OpenRouter могут ответить 400 на незнакомое
            # поле.
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body

    # ------------------------------------------------------------------
    # Вызов
    # ------------------------------------------------------------------
    async def call_chat(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.3,
    ) -> str:
        """Вызов /chat/completions активного движка."""
        model = model or settings.default_model
        client = _get_client()
        response = await client.post(
            f"{settings.llm_base_url}/chat/completions",
            headers=self._headers(),
            json=self._request_body(model, prompt, temperature),
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
                "в настройках движка)"
            )
        return str(content)

    # Историческое имя (этапы 5-9 вызывали именно его)
    call_lmstudio = call_chat

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
        if await self._check_engine():
            try:
                text = await self.call_chat(prompt, model, temperature)
                return text, settings.LLM_ENGINE, model or settings.default_model
            except Exception as e:
                # Ошибка вызова ≠ сервер недоступен: кэш доступности
                # не отравляем, чтобы retry мог попробовать снова.
                logger.warning("LLM call failed (%s): %s", settings.LLM_ENGINE, e)
                raise

        raise RuntimeError(f"LLM-движок {settings.LLM_ENGINE!r} недоступен")


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
