# backend/worker/tasks.py
"""
Фоновые задачи Celery (этап 11).

- summarize — фоновая суммаризация через perform_summarization;
- reindex_knowledge_base — перестройка BM25 + Qdrant после пополнения БЗ
  (например, после подтверждения правки в HITL);
- clear_old_cache — очистка устаревшего кэша результатов.

Асинхронный код выполняется на персистентном event loop процесса
воркера: SQLAlchemy async-движки и httpx-клиенты привязаны к loop,
поэтому loop переиспользуется между задачами и закрывается один раз
при остановке воркера.
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from sqlalchemy import delete, select

from backend.worker.celery_app import celery_app

logger = logging.getLogger(__name__)

# Внутри воркера — не больше 4 параллельных суммаризаций на процесс
# (дублирует семафор пайплайна; страховка при нескольких воркерах).
_MAX_CONCURRENT_SUMMARIZATIONS = 4
_summarize_semaphore: Optional[asyncio.Semaphore] = None

_worker_loop: Optional[asyncio.AbstractEventLoop] = None


# ============================================================================
# Персистентный event loop
# ============================================================================
def _get_worker_loop() -> asyncio.AbstractEventLoop:
    """Возвращает event loop процесса воркера, создавая при необходимости."""
    global _worker_loop
    if _worker_loop is None or _worker_loop.is_closed():
        _worker_loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_worker_loop)
    return _worker_loop


def _run_async(coro):
    """Выполняет корутину на персистентном loop воркера."""
    return _get_worker_loop().run_until_complete(coro)


def close_worker_loop() -> None:
    """Закрывает loop (вызывается сигналом worker_shutdown)."""
    global _worker_loop
    if _worker_loop is not None and not _worker_loop.is_closed():
        _worker_loop.run_until_complete(_close_clients())
        _worker_loop.close()
    _worker_loop = None


async def _close_clients() -> None:
    """Закрывает httpx-клиенты LLM перед закрытием loop."""
    try:
        from backend.utils.llm_client import close_llm_clients

        await close_llm_clients()
    except Exception as e:  # pragma: no cover - страховка
        logger.warning("close_llm_clients failed: %s", e)


def _get_semaphore() -> asyncio.Semaphore:
    global _summarize_semaphore
    if _summarize_semaphore is None:
        _summarize_semaphore = asyncio.Semaphore(
            _MAX_CONCURRENT_SUMMARIZATIONS
        )
    return _summarize_semaphore


# ============================================================================
# Задача: фоновая суммаризация
# ============================================================================
@celery_app.task(
    bind=True,
    name="tasks.summarize",
    autoretry_for=(ConnectionError,),
    retry_kwargs={"max_retries": 2, "countdown": 5},
)
def summarize(self, user_id: str, **kwargs) -> Dict[str, Any]:
    """
    Фоновая суммаризация. kwargs — параметры perform_summarization:
    mode, summary_mode, article_text, arxiv_id, title, field, use_cache.

    Возвращает asdict(SummarizationResult) — JSON-сериализуемо.
    """
    logger.info("Task summarize[%s] started for user %s", self.request.id, user_id)
    return _run_async(_summarize_async(user_id, kwargs))


async def _summarize_async(
    user_id: str, kwargs: Dict[str, Any]
) -> Dict[str, Any]:
    from dataclasses import asdict as _asdict

    from backend.db.models import User
    from backend.db.session import AsyncSessionLocal
    from backend.services.pipeline import safe_perform_summarization

    async with _get_semaphore():
        async with AsyncSessionLocal() as session:
            async with session.begin():
                result = await session.execute(
                    select(User).where(User.id == user_id)
                )
                user = result.scalars().first()

            if user is None:
                return {"error": f"Пользователь {user_id} не найден"}

            async with session.begin():
                outcome = await safe_perform_summarization(
                    user=user, db=session, **kwargs
                )
            return _asdict(outcome)


# ============================================================================
# Задача: переиндексация базы знаний
# ============================================================================
@celery_app.task(name="tasks.reindex_knowledge_base")
def reindex_knowledge_base() -> Dict[str, Any]:
    """
    Перестраивает индексы RAG: BM25 в памяти + векторный индекс Qdrant.

    Вызывается после подтверждения HITL-правки (новая пара в БЗ).
    Qdrant недоступен → warning, задача не падает.
    """
    return _run_async(_reindex_async())


async def _reindex_async() -> Dict[str, Any]:
    from backend.data.knowledge_base import KBManager
    from backend.db.session import AsyncSessionLocal, engine
    from backend.retrieval.vector_index import VectorIndex

    kb_manager = KBManager()
    vector_index = VectorIndex()

    async with AsyncSessionLocal() as session:
        total = await kb_manager.initialize(session)

    point_ids = vector_index.add_entries(kb_manager.entries)
    points_count = vector_index.count()

    try:
        await engine.dispose()
    except Exception:  # pragma: no cover
        pass

    result = {
        "kb_entries": total,
        "points_upserted": len(point_ids),
        "qdrant_available": points_count is not None,
    }
    if points_count is not None:
        result["points_total"] = points_count
    logger.info("reindex_knowledge_base: %s", result)
    return result


# ============================================================================
# Задача: очистка старого кэша
# ============================================================================
@celery_app.task(name="tasks.clear_old_cache")
def clear_old_cache(days: int = 30) -> Dict[str, Any]:
    """Удаляет записи кэша старше days дней. Возвращает число удалённых."""
    return _run_async(_clear_old_cache_async(days))


async def _clear_old_cache_async(days: int) -> Dict[str, Any]:
    from backend.db.models import SummaryCache
    from backend.db.session import AsyncSessionLocal, engine

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    async with AsyncSessionLocal() as session:
        async with session.begin():
            result = await session.execute(
                delete(SummaryCache).where(
                    SummaryCache.created_at < cutoff
                )
            )
            deleted = result.rowcount

    try:
        await engine.dispose()
    except Exception:  # pragma: no cover
        pass

    logger.info("clear_old_cache(days=%s): удалено %s", days, deleted)
    return {"deleted": deleted, "days": days}
