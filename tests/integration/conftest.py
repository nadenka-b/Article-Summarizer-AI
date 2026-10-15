# tests/integration/conftest.py
"""
Общие фикстуры для интеграционных тестов.

Заменяем БД на SQLite in-memory и переопределяем FastAPI-зависимость get_db.
"""

import pytest
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.api.deps import get_db
from backend.api.main import app
from backend.db.base import Base
from backend.db import models  # noqa: F401 — регистрация моделей в Base.metadata


@pytest.fixture
async def test_engine():
    """Создаёт in-memory SQLite и все таблицы."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield engine
    await engine.dispose()


@pytest.fixture
async def test_session_maker(test_engine):
    """Фабрика сессий, привязанная к in-memory SQLite."""
    return async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


@pytest.fixture
async def client(test_session_maker):
    """
    HTTP-клиент к FastAPI с подменённой зависимостью get_db.

    Все запросы роутов идут в in-memory SQLite, а не в реальную БД.
    """
    async def override_get_db():
        async with test_session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()