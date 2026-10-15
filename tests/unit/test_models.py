# tests/unit/test_models.py
"""Юнит-тесты для SQLAlchemy-моделей."""

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from backend.db.base import Base
from backend.db import models  # noqa: F401
from backend.db.models import User, Article, KnowledgeBase, GlossaryDB


@pytest.fixture
async def db_session():
    """In-memory SQLite для тестов."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    async with session_maker() as session:
        yield session

    await engine.dispose()


class TestUser:
    async def test_create_user(self, db_session: AsyncSession):
        user = User(
            email="test@example.com",
            username="testuser",
            hashed_password="hashed",
            role="user",
        )
        db_session.add(user)
        await db_session.commit()

        result = await db_session.execute(select(User).where(User.email == "test@example.com"))
        fetched = result.scalar_one()
        assert fetched.username == "testuser"
        assert fetched.role == "user"
        assert fetched.is_active is True


class TestArticle:
    async def test_create_article(self, db_session: AsyncSession):
        user = User(
            email="owner@example.com",
            username="owner",
            hashed_password="hashed",
        )
        db_session.add(user)
        await db_session.commit()

        article = Article(
            owner_id=user.id,
            title="Attention Is All You Need",
            source="arxiv",
            source_id="1706.03762",
            field="cs",
            status="uploaded",
        )
        db_session.add(article)
        await db_session.commit()

        result = await db_session.execute(select(Article))
        fetched = result.scalar_one()
        assert fetched.title == "Attention Is All You Need"
        assert fetched.source == "arxiv"
        assert fetched.field == "cs"


class TestKnowledgeBase:
    async def test_create_kb_entry(self, db_session: AsyncSession):
        kb = KnowledgeBase(
            field="cs",
            language="en",
            source_text="The transformer architecture...",
            source_text_hash="abc123",
            target_summary="We propose a novel transformer...",
        )
        db_session.add(kb)
        await db_session.commit()

        result = await db_session.execute(select(KnowledgeBase))
        fetched = result.scalar_one()
        assert fetched.field == "cs"
        assert fetched.language == "en"


class TestGlossary:
    async def test_create_glossary_term(self, db_session: AsyncSession):
        term = GlossaryDB(
            field="cs",
            term="Transformer",
            definition="A neural network architecture based on self-attention.",
            category="concept",
        )
        db_session.add(term)
        await db_session.commit()

        result = await db_session.execute(select(GlossaryDB))
        fetched = result.scalar_one()
        assert fetched.term == "Transformer"
        assert fetched.category == "concept"