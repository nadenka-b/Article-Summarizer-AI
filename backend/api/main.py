# backend/api/main.py
"""
Точка входа FastAPI для Article-Summarizer-AI.

Здесь создаётся приложение, подключаются middleware и роутеры.
Uvicorn запускает объект app по пути backend.api.main:app.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.routes import auth, health
from backend.core.config import get_settings
from backend.utils.logging import setup_logging
from backend.db.base import Base
from backend.db.session import engine
# ВАЖНО: импортируем модели, чтобы SQLAlchemy знала о них
from backend.db import models  # noqa: F401

setup_logging()
logger = logging.getLogger(__name__)
settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Код, выполняющийся при старте и остановке приложения."""
    logger.info("=" * 60)
    logger.info("%s starting...", settings.PROJECT_NAME)
    logger.info("Debug mode: %s", settings.DEBUG)
    logger.info("Summarization engine: %s", settings.SUMMARIZATION_ENGINE)
    logger.info("Summary mode: %s", settings.SUMMARY_MODE)
    logger.info("=" * 60)

    # Здесь позже будет:
    # - создание таблиц БД (Base.metadata.create_all) — этап 3
    # - инициализация Qdrant-коллекции — этап 7
    # - прогрев эмбеддингов — этап 7

    # Создание таблиц
    logger.info("Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables ready")
    yield

    # Закрытие соединений
    logger.info("Closing database connections...")
    await engine.dispose()

    logger.info("=" * 60)
    logger.info("%s stopped", settings.PROJECT_NAME)
    logger.info("=" * 60)


app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Мультиагентная платформа суммаризации научных статей",
    version="0.1.0",
    debug=settings.DEBUG,
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.DEBUG else [],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(health.router)
app.include_router(auth.router)


@app.get("/", tags=["root"])
async def root():
    """Корневой эндпоинт."""
    return {
        "message": f"{settings.PROJECT_NAME} is running",
        "docs": "/docs",
        "healthz": "/healthz",
        "fields": "/fields",
    }