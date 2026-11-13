# tests/unit/test_hybrid_search.py
"""Юнит-тесты гибридного поиска (BM25 + вектора + RRF).

Реальная модель эмбеддингов и Qdrant не используются:
VectorIndex подменяется заглушкой, KBManager работает на in-memory SQLite.
"""

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.data.knowledge_base import KBEntry, KBManager
from backend.db.base import Base
from backend.db import models  # noqa: F401
from backend.retrieval.hybrid_retriever import (
    format_few_shot_examples,
    hybrid_search,
)


def _entry(entry_id: str, text: str, summary: str = "резюме") -> KBEntry:
    return KBEntry(
        id=entry_id,
        field="ai_ml",
        language="ru",
        source_text=text,
        target_summary=summary,
        search_text=text,
    )


class FakeVectorIndex:
    """Заглушка векторного индекса: возвращает заданные записи."""

    def __init__(self, results=None):
        self._results = results or []
        self.calls = 0

    def search(self, query, top_k=5, field=None):
        self.calls += 1
        return self._results[:top_k]


@pytest.fixture
async def kb_manager():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    manager = KBManager()
    async with maker() as session:
        async with session.begin():
            await manager.add_many(session, [
                {
                    "field": "ai_ml",
                    "language": "ru",
                    "source_text": "Нейронные сети для распознавания речи",
                    "target_summary": "Резюме про нейросети и речь",
                    "source": "corpus",
                    "quality_score": None,
                },
                {
                    "field": "ai_ml",
                    "language": "ru",
                    "source_text": "Экономические модели прогнозирования",
                    "target_summary": "Резюме про экономику",
                    "source": "corpus",
                    "quality_score": None,
                },
                {
                    "field": "ai_ml",
                    "language": "ru",
                    "source_text": "Квантовые алгоритмы факторизации",
                    "target_summary": "Резюме про квантовые вычисления",
                    "source": "corpus",
                    "quality_score": None,
                },
            ])
    await engine.dispose()
    return manager


def _find_entry(kb_manager: KBManager, summary_part: str) -> KBEntry:
    for entry in kb_manager.entries:
        if summary_part in entry.target_summary:
            return entry
    raise AssertionError(f"Запись с '{summary_part}' не найдена")


class TestHybridSearch:
    def test_merges_both_sources(self, kb_manager):
        # Векторный поиск возвращает реальную запись из БЗ
        nn_entry = _find_entry(kb_manager, "нейросети")
        vector = FakeVectorIndex(results=[(nn_entry, 0.92)])
        results = hybrid_search(
            kb_manager, vector, "нейронные сети распознавание речи", top_k=2
        )
        assert len(results) >= 1
        assert vector.calls == 1

    def test_agreement_ranks_first(self, kb_manager):
        """Запись, найденная ОБИМИ источниками, получает лучший RRF-балл."""
        nn_entry = _find_entry(kb_manager, "нейросети")
        econ_entry = _find_entry(kb_manager, "экономику")
        # Векторный поиск ставит экономику выше, но нейросети
        # подтверждены ещё и BM25 → RRF выведет их вперёд.
        vector = FakeVectorIndex(results=[(econ_entry, 0.9), (nn_entry, 0.8)])

        results = hybrid_search(
            kb_manager, vector, "нейронные сети распознавание речи", top_k=2
        )
        ids = [e.id for e, _ in results]
        assert ids[0] == nn_entry.id

    def test_degrades_to_bm25_when_vector_empty(self, kb_manager):
        vector = FakeVectorIndex(results=[])
        results = hybrid_search(
            kb_manager, vector, "нейронные сети распознавание речи", top_k=2
        )
        # Пустой векторный поиск не должен ломать гибридный
        assert len(results) >= 1
        assert results[0][0].target_summary == "Резюме про нейросети и речь"

    def test_no_results_at_all(self, kb_manager):
        vector = FakeVectorIndex(results=[])
        assert hybrid_search(kb_manager, vector, "zzz xxx", top_k=2) == []


class TestFormatFewShot:
    def test_formats_examples(self):
        entry = _entry("e1", "текст статьи про нейросети", "эталонная аннотация")
        text = format_few_shot_examples([(entry, 0.123)])
        assert "ПРИМЕР" in text
        assert "эталонная аннотация" in text
        assert "0.123" in text

    def test_respects_max_examples(self):
        entries = [
            (_entry(f"e{i}", f"текст {i}", f"резюме {i}"), 0.5) for i in range(5)
        ]
        text = format_few_shot_examples(entries, max_examples=2)
        assert text.count("ПРИМЕР") == 2

    def test_empty_results(self):
        assert format_few_shot_examples([]) == ""
