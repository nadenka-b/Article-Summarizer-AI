# backend/db/session.py
"""
Настройка подключения к базе данных.

- Создаёт async engine (пул соединений).
- Создаёт фабрику сессий (async_sessionmaker).
- Предоставляет зависимость get_async_session() для FastAPI.
"""

import logging
from typing import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.core.config import get_settings


logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Async engine
# ============================================================================
# Для SQLite: pool_size не работает (SQLite не поддерживает),
#            нужен StaticPool для тестов in-memory.
# Для PostgreSQL: используем обычный пул.
# ============================================================================
_connect_args = {}
_engine_kwargs = {
    "echo": settings.DEBUG,  # логировать SQL в DEBUG-режиме
    "future": True,
}

if settings.ASYNC_DATABASE_URL.startswith("sqlite"):
    # SQLite не поддерживает pool_size/max_overflow
    _connect_args = {"check_same_thread": False}
else:
    # PostgreSQL
    _engine_kwargs.update(
        {
            "pool_size": 10,
            "max_overflow": 20,
            "pool_pre_ping": True,  # проверять соединение перед использованием
            "pool_recycle": 3600,   # пересоздавать соединения раз в час
        }
    )


engine = create_async_engine(
    settings.ASYNC_DATABASE_URL,
    connect_args=_connect_args,
    **_engine_kwargs,
)


# ============================================================================
# Фабрика сессий
# ============================================================================
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,   # не истекать объектам после commit (важно для async)
    autoflush=False,
    autocommit=False,
)


# ============================================================================
# FastAPI dependency
# ============================================================================
async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Возвращает сессию БД для использования в роутах FastAPI.

    Использование:
        @router.get("/users")
        async def get_users(db: AsyncSession = Depends(get_async_session)):
            result = await db.execute(select(User))
            ...
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()