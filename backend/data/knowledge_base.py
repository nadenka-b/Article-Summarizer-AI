# backend/data/knowledge_base.py
"""
Менеджер базы знаний: эталонные пары «текст статьи → аннотация».

- Пары хранятся в таблице knowledge_base (PostgreSQL/SQLite).
- Лексический поиск — in-memory BM25 по поисковому представлению пары.
- BM25 строится лениво и только для «грязных» областей (field):
  add_many помечает область грязной, search при необходимости перестраивает.
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from backend.core.config import get_settings
from backend.db.models import KnowledgeBase
from backend.retrieval.bm25_index import BM25Index

logger = logging.getLogger(__name__)
settings = get_settings()

ALL_FIELDS = "*"  # псевдо-область: BM25 сразу по всем парам


@dataclass
class KBEntry:
    """Пара БЗ в памяти (для BM25/векторов/few-shot промптов)."""

    id: str
    field: str
    language: str
    source_text: str        # полный текст пары (может быть большим)
    target_summary: str     # эталонная аннотация
    search_text: str        # урезанное представление для поиска (заголовок + начало)
    source: Optional[str] = None
    quality_score: Optional[float] = None

    def bm25_text(self) -> str:
        """Текст, по которому работает лексический поиск."""
        return self.search_text


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class KBManager:
    """
    Работа с парами БЗ: загрузка из БД, идемпотентная вставка, BM25-поиск.
    """

    def __init__(self) -> None:
        self._entries: List[KBEntry] = []
        self._by_id: Dict[str, KBEntry] = {}
        self._loaded: bool = False
        # BM25: одна «глобальная» очередь по всем парам + по областям
        self._bm25: Dict[str, BM25Index] = {}
        self._bm25_dirty: set = set()

    # ------------------------------------------------------------------------
    # Загрузка из БД
    # ------------------------------------------------------------------------
    async def initialize(self, session: AsyncSession) -> int:
        """Загружает все пары из БД в память. Возвращает количество."""
        result = await session.execute(select(KnowledgeBase))
        rows = result.scalars().all()

        self._entries = [
            KBEntry(
                id=row.id,
                field=row.field,
                language=row.language,
                source_text=row.source_text,
                target_summary=row.target_summary,
                search_text=self._make_search_text(row.source_text),
                source=row.source,
                quality_score=row.quality_score,
            )
            for row in rows
        ]
        self._by_id = {e.id: e for e in self._entries}
        self._loaded = True
        self._mark_all_dirty()
        return len(self._entries)

    @staticmethod
    def _make_search_text(source_text: str) -> str:
        """Урезанное представление пары для BM25/эмбеддингов."""
        limit = settings.KB_SEARCH_TEXT_CHARS
        return source_text[:limit]

    def _mark_all_dirty(self) -> None:
        self._bm25_dirty = {ALL_FIELDS} | {e.field for e in self._entries}

    # ------------------------------------------------------------------------
    # Вставка (идемпотентная)
    # ------------------------------------------------------------------------
    async def insert_in_session(
        self,
        session: AsyncSession,
        field: str,
        language: str,
        source_text: str,
        target_summary: str,
        source: Optional[str] = None,
        quality_score: Optional[float] = None,
    ) -> Optional[str]:
        """
        INSERT ... ON CONFLICT (field, source_text_hash) DO NOTHING RETURNING id.

        Возвращает id новой записи или None, если пара уже существует.
        """
        entry_values = {
            "field": field,
            "language": language,
            "source_text": source_text,
            "source_text_hash": sha256_text(source_text),
            "target_summary": target_summary,
            "source": source,
            "quality_score": quality_score,
        }

        dialect = session.bind.dialect.name if session.bind else "sqlite"
        if dialect == "postgresql":
            stmt = (
                pg_insert(KnowledgeBase)
                .values(**entry_values)
                .on_conflict_do_nothing(
                    constraint="uq_kb_field_hash",
                )
                .returning(KnowledgeBase.id)
            )
        else:
            stmt = (
                sqlite_insert(KnowledgeBase)
                .values(**entry_values)
                .on_conflict_do_nothing(
                    index_elements=["field", "source_text_hash"],
                )
                .returning(KnowledgeBase.id)
            )

        result = await session.execute(stmt)
        return result.scalar_one_or_none()

    async def add_many(
        self,
        session: AsyncSession,
        pairs: Sequence[Dict[str, Any]],
    ) -> Tuple[int, int]:
        """
        Добавляет пары (dict с ключами field, language, source_text,
        target_summary, source, quality_score).

        Возвращает (added, skipped). BM25 НЕ перестраивает — только помечает
        области грязными; фактическая перестройка произойдёт в search.
        Новые записи сразу попадают в память (без перечитывания БД).
        """
        added = 0
        skipped = 0
        for pair in pairs:
            new_id = await self.insert_in_session(session, **pair)
            if new_id is not None:
                added += 1
                entry = KBEntry(
                    id=new_id,
                    field=pair["field"],
                    language=pair["language"],
                    source_text=pair["source_text"],
                    target_summary=pair["target_summary"],
                    search_text=self._make_search_text(pair["source_text"]),
                    source=pair.get("source"),
                    quality_score=pair.get("quality_score"),
                )
                self._entries.append(entry)
                self._by_id[entry.id] = entry
                self._bm25_dirty.add(pair["field"])
                self._bm25_dirty.add(ALL_FIELDS)
            else:
                skipped += 1
        await session.flush()
        return added, skipped

    # ------------------------------------------------------------------------
    # Доступ к записям
    # ------------------------------------------------------------------------
    @property
    def entries(self) -> List[KBEntry]:
        return list(self._entries)

    def get_by_id(self, entry_id: str) -> Optional[KBEntry]:
        return self._by_id.get(entry_id)

    # ------------------------------------------------------------------------
    # Поиск (BM25, ленивая перестройка)
    # ------------------------------------------------------------------------
    def _ensure_bm25(self, field_key: str) -> BM25Index:
        """Возвращает индекс области, перестроив его при необходимости."""
        if field_key not in self._bm25_dirty and field_key in self._bm25:
            return self._bm25[field_key]

        if field_key == ALL_FIELDS:
            subset = self._entries
        else:
            subset = [e for e in self._entries if e.field == field_key]

        index = BM25Index(entry_getter=KBEntry.bm25_text)
        index.initialize(subset)
        self._bm25[field_key] = index
        self._bm25_dirty.discard(field_key)
        logger.debug(
            "BM25 rebuilt for %s: %d entries", field_key, len(index)
        )
        return index

    def search(
        self,
        query: str,
        field: Optional[str] = None,
        top_k: int = 5,
    ) -> List[Tuple[KBEntry, float]]:
        """Лексический поиск по парам. Только score > 0."""
        field_key = field if field else ALL_FIELDS
        index = self._ensure_bm25(field_key)
        return index.search(query, top_k=top_k)

    async def count(self, session: AsyncSession) -> int:
        result = await session.execute(select(func.count(KnowledgeBase.id)))
        return result.scalar_one()
