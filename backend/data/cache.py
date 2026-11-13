# backend/data/cache.py
"""
Кэш результатов суммаризации.

Ключ: (source_text_hash, summary_mode, engine_used, model_used).
None в engine/model означает «без фильтра» — кэш срабатывает
независимо от движка/модели.
"""

import logging
from typing import Optional

from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import SummaryCache

logger = logging.getLogger(__name__)


class SummaryCacheManager:
    """Чтение/запись кэша с счётчиком попаданий (hits)."""

    async def get_cached(
        self,
        session: AsyncSession,
        source_text_hash: str,
        summary_mode: str,
        engine_used: Optional[str] = None,
        model_used: Optional[str] = None,
    ) -> Optional[str]:
        """
        Возвращает кэшированное резюме или None.

        engine_used/model_used = None → фильтр по этим колонкам
        не применяется (берём первую подходящую запись).
        При попадании инкрементирует hits.
        """
        stmt = select(SummaryCache).where(
            SummaryCache.source_text_hash == source_text_hash,
            SummaryCache.summary_mode == summary_mode,
        )
        if engine_used is not None:
            stmt = stmt.where(SummaryCache.engine_used == engine_used)
        if model_used is not None:
            stmt = stmt.where(SummaryCache.model_used == model_used)

        result = await session.execute(stmt)
        entry = result.scalars().first()

        if entry is None:
            return None

        await session.execute(
            update(SummaryCache)
            .where(SummaryCache.id == entry.id)
            .values(hits=SummaryCache.hits + 1)
        )
        await session.flush()
        return entry.output

    async def put(
        self,
        session: AsyncSession,
        source_text_hash: str,
        summary_mode: str,
        output: str,
        engine_used: Optional[str] = None,
        model_used: Optional[str] = None,
    ) -> bool:
        """
        Сохраняет результат в кэш (идемпотентно: при конфликте ключа
        существующая запись остаётся). Возвращает True, если записали.
        """
        values = {
            "source_text_hash": source_text_hash,
            "summary_mode": summary_mode,
            "engine_used": engine_used,
            "model_used": model_used,
            "output": output,
        }

        dialect = session.bind.dialect.name if session.bind else "sqlite"
        if dialect == "postgresql":
            stmt = (
                pg_insert(SummaryCache)
                .values(**values)
                .on_conflict_do_nothing(constraint="uq_cache_key")
            )
        else:
            stmt = (
                sqlite_insert(SummaryCache)
                .values(**values)
                .on_conflict_do_nothing(
                    index_elements=[
                        "source_text_hash",
                        "summary_mode",
                        "engine_used",
                        "model_used",
                    ],
                )
            )

        result = await session.execute(stmt)
        await session.flush()
        return result.rowcount > 0
