# tests/unit/test_kb_manager.py
"""Юнит-тесты KBManager: идемпотентная вставка, ленивый BM25, поиск."""

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.data.knowledge_base import KBManager, sha256_text
from backend.db.base import Base
from backend.db import models  # noqa: F401 — регистрация моделей


@pytest.fixture
async def session_maker():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )
    yield maker
    await engine.dispose()


def _pair(source_text: str, field: str = "ai_ml", summary: str = "аннотация"):
    return {
        "field": field,
        "language": "ru",
        "source_text": source_text,
        "target_summary": summary,
        "source": "corpus",
        "quality_score": None,
    }


class TestInsert:
    async def test_insert_returns_id(self, session_maker):
        manager = KBManager()
        async with session_maker() as session:
            async with session.begin():
                new_id = await manager.insert_in_session(
                    session, **_pair("текст статьи про нейросети")
                )
        assert new_id is not None

    async def test_duplicate_insert_returns_none(self, session_maker):
        manager = KBManager()
        pair = _pair("текст статьи про нейросети")

        async with session_maker() as session:
            async with session.begin():
                first = await manager.insert_in_session(session, **pair)

        async with session_maker() as session:
            async with session.begin():
                second = await manager.insert_in_session(session, **pair)

        assert first is not None
        assert second is None

    async def test_same_text_different_field_both_inserted(self, session_maker):
        """Уникальность — по паре (field, source_text_hash)."""
        manager = KBManager()
        text = "один и тот же текст"

        async with session_maker() as session:
            async with session.begin():
                a = await manager.insert_in_session(
                    session, **_pair(text, field="ai_ml")
                )
                b = await manager.insert_in_session(
                    session, **_pair(text, field="nlp")
                )

        assert a is not None
        assert b is not None
        assert a != b

    async def test_add_many_counts(self, session_maker):
        manager = KBManager()
        pairs = [
            _pair("первый текст"),
            _pair("второй текст"),
            _pair("первый текст"),  # дубль
        ]
        async with session_maker() as session:
            async with session.begin():
                added, skipped = await manager.add_many(session, pairs)

        assert (added, skipped) == (2, 1)

    async def test_repeated_add_many_adds_nothing(self, session_maker):
        manager = KBManager()
        pairs = [_pair("первый"), _pair("второй")]

        async with session_maker() as session:
            async with session.begin():
                first = await manager.add_many(session, pairs)

        async with session_maker() as session:
            async with session.begin():
                second = await manager.add_many(session, pairs)

        assert first == (2, 0)
        assert second == (0, 2)


class TestSearch:
    async def test_initialize_and_search(self, session_maker):
        manager = KBManager()
        async with session_maker() as session:
            async with session.begin():
                await manager.add_many(session, [
                    _pair(
                        "Нейронные сети применяются для распознавания речи",
                        summary="Про нейросети и речь",
                    ),
                    _pair(
                        "Экономические модели прогнозирования спроса",
                        summary="Про экономику",
                    ),
                    _pair(
                        "Квантовые алгоритмы факторизации чисел",
                        summary="Про квантовые вычисления",
                    ),
                ])

        async with session_maker() as session:
            count = await manager.initialize(session)

        assert count == 3

        results = manager.search("нейронные сети распознавание")
        assert len(results) == 1
        entry, score = results[0]
        assert score > 0
        assert entry.target_summary == "Про нейросети и речь"

    async def test_search_filters_by_field(self, session_maker):
        manager = KBManager()
        async with session_maker() as session:
            async with session.begin():
                await manager.add_many(session, [
                    _pair("нейросети для анализа тональности", field="nlp"),
                    _pair("нейросети для машинного перевода", field="nlp"),
                    _pair("стемминг русских текстов корпусов", field="nlp"),
                    _pair("нейросети для анализа тональности", field="ai_ml"),
                ])
        async with session_maker() as session:
            await manager.initialize(session)

        # BM25 матчит точные словоформы — используем форму из документа
        results = manager.search("тональности", field="nlp")
        assert len(results) == 1
        assert results[0][0].field == "nlp"
        assert "тональности" in results[0][0].source_text

    async def test_add_many_marks_dirty_not_rebuild(self, session_maker):
        """add_many не перестраивает BM25 сразу — только помечает."""
        manager = KBManager()
        async with session_maker() as session:
            async with session.begin():
                await manager.add_many(session, [_pair("нейросети текст")])

        # После add_many индекса ещё нет (ленивость)
        assert "*" in manager._bm25_dirty
        assert manager._bm25 == {}

    async def test_search_after_add_rebuilds_lazily(self, session_maker):
        """search перестраивает индекс по требованию."""
        manager = KBManager()
        async with session_maker() as session:
            async with session.begin():
                await manager.add_many(session, [
                    _pair("нейросети применяются в задачах классификации"),
                    _pair("экономические модели спроса и предложения"),
                    _pair("квантовые вычисления и запутанность"),
                ])

        results = manager.search("нейросети классификация")
        # индекс перестроен в момент поиска
        assert "*" in manager._bm25
        assert results


class TestSha256:
    def test_deterministic(self):
        assert sha256_text("abc") == sha256_text("abc")
        assert sha256_text("abc") != sha256_text("abd")

    def test_hex_length(self):
        assert len(sha256_text("текст")) == 64
