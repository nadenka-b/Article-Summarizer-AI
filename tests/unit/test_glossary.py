# tests/unit/test_glossary.py
"""Юнит-тесты встроенного глоссария и GlossaryManager."""

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.data.glossary import BUILTIN_GLOSSARY, GlossaryManager
from backend.db.base import Base
from backend.db import models  # noqa: F401


class TestBuiltinGlossary:
    def test_at_least_500_entries(self):
        # Цель этапа 6: не менее 500 терминов в глоссарии
        assert len(BUILTIN_GLOSSARY) >= 500

    def test_unique_field_term_pairs(self):
        pairs = [(e["field"], e["term"]) for e in BUILTIN_GLOSSARY]
        assert len(pairs) == len(set(pairs))

    def test_all_entries_have_required_keys(self):
        for entry in BUILTIN_GLOSSARY:
            assert entry["term"].strip()
            assert entry["definition"].strip()
            assert entry["category"] in {
                "concept", "abbreviation", "template"
            }
            assert entry["field"]
            assert entry["language"] == "ru"

    def test_categories_present(self):
        categories = {e["category"] for e in BUILTIN_GLOSSARY}
        assert {"concept", "abbreviation", "template"} <= categories

    def test_abbreviations_decoded(self):
        abbreviations = {
            e["term"]: e["definition"]
            for e in BUILTIN_GLOSSARY
            if e["category"] == "abbreviation"
        }
        assert "ИИ" in abbreviations
        assert "искусственный интеллект" in abbreviations["ИИ"]


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


class TestGlossaryManager:
    async def test_add_many_and_reimport_idempotent(self, session_maker):
        manager = GlossaryManager()
        entries = BUILTIN_GLOSSARY[:20]

        async with session_maker() as session:
            async with session.begin():
                first = await manager.add_many(session, entries)

        async with session_maker() as session:
            async with session.begin():
                second = await manager.add_many(session, entries)

        assert first == (20, 0)
        assert second == (0, 20)  # повторный запуск → added=0

    async def test_count(self, session_maker):
        manager = GlossaryManager()
        async with session_maker() as session:
            async with session.begin():
                await manager.add_many(session, BUILTIN_GLOSSARY[:5])
        async with session_maker() as session:
            assert await manager.count(session) == 5

    async def test_search_terms_finds_substring(self, session_maker):
        manager = GlossaryManager()
        async with session_maker() as session:
            async with session.begin():
                await manager.add_many(session, [
                    {
                        "term": "нейронная сеть",
                        "definition": "модель из связанных нейронов",
                        "category": "concept",
                        "field": "ai",
                        "language": "ru",
                    },
                    {
                        "term": "свёрточная нейронная сеть",
                        "definition": "нейросеть для изображений",
                        "category": "concept",
                        "field": "ai",
                        "language": "ru",
                    },
                ])

        async with session_maker() as session:
            found = await manager.search_terms(session, "нейронн")

        terms = [g.term for g in found]
        assert "нейронная сеть" in terms
        assert "свёрточная нейронная сеть" in terms
