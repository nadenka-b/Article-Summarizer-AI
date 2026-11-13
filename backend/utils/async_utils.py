# backend/utils/async_utils.py
"""Утилиты для асинхронного кода: повтор вызовов при сбоях."""

import asyncio
import functools
import logging
from typing import Any, Callable, Tuple, Type

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[BaseException], ...] = (Exception,),
) -> Callable:
    """
    Декоратор для async-функций: повторяет вызов при ошибке.

    Между попытками — пауза `delay` секунд, растущая в `backoff` раз
    (1с → 2с → 4с при backoff=2).

    Использование:
        @retry(max_attempts=3, delay=1.0)
        async def call_api(): ...
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 1
            current_delay = delay
            while True:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    if attempt >= max_attempts:
                        logger.error(
                            "%s failed after %d attempts: %s",
                            func.__name__, attempt, e,
                        )
                        raise
                    logger.warning(
                        "%s failed (attempt %d/%d): %s — retry in %.1fs",
                        func.__name__, attempt, max_attempts, e, current_delay,
                    )
                    await asyncio.sleep(current_delay)
                    attempt += 1
                    current_delay *= backoff

        return wrapper

    return decorator
