# tests/unit/test_tasks.py
"""Юнит-тесты Celery-задач (этап 11).

task_always_eager=True — задачи выполняются синхронно, Redis не нужен.
"""

import json

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

import backend.worker.tasks as tasks_module
from backend.db.base import Base
from backend.db import models  # noqa: F401
from backend.worker.celery_app import celery_app


class FakeLLM:
    def __init__(self):
        self.calls = 0

    async def generate(self, prompt, model=None, temperature=0.3,
                       task="generate"):
        self.calls += 1
        if "Оцени резюме" in prompt:
            return json.dumps({
                "completeness": 9.0, "accuracy": 9.0,
                "coherence": 9.0, "style": 9.0,
                "reasoning": "ok", "suggestions": [], "missing_terms": [],
            }), "mock", "mock-model"
        return (
            "This test summary describes a novel scientific method "
            "for article summarization and reports improved quality "
            "metrics. " * 4
        ), "mock", "mock-model"


@pytest.fixture
def eager_celery(monkeypatch):
    """Задачи выполняются синхронно, без брокера."""
    celery_app.conf.task_always_eager = True
    celery_app.conf.task_store_eager_result = True
    yield celery_app
    celery_app.conf.task_always_eager = False
    celery_app.conf.task_store_eager_result = False


@pytest.fixture
async def worker_db(monkeypatch, tmp_path):
    """
    Подменяет БД воркера на файловый SQLite во временной папке.

    Файловая БД (а не :memory:) обязательна: тест выполняется в двух
    event loop (pytest и персистентный loop воркера), а in-memory БД
    живёт только внутри одного соединения одного loop.
    """
    db_path = tmp_path / "worker_test.db"
    url = f"sqlite+aiosqlite:///{db_path}"
    engine = create_async_engine(url, echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    maker = async_sessionmaker(
        bind=engine, class_=AsyncSession, expire_on_commit=False
    )

    # Задачи импортируют AsyncSessionLocal внутри функций —
    # подменяем в модуле db.session, откуда они его берут.
    import backend.db.session as db_session

    monkeypatch.setattr(db_session, "AsyncSessionLocal", maker)
    monkeypatch.setattr(db_session, "engine", engine)

    # Пользователь, от имени которого работает задача.
    from backend.db.models import User

    async with maker() as session:
        async with session.begin():
            user = User(
                email="worker@example.com",
                username="worker",
                hashed_password="x",
                role="user",
            )
            session.add(user)
            await session.flush()
            user_id = user.id

    yield {"maker": maker, "user_id": user_id}

    await engine.dispose()


class TestSummarizeTask:
    def test_summarize_creates_summary(
        self, eager_celery, worker_db, monkeypatch
    ):
        fake = FakeLLM()
        monkeypatch.setattr(
            "backend.utils.llm_client.get_llm_client", lambda: fake
        )

        result = tasks_module.summarize(
            user_id=worker_db["user_id"],
            article_text=(
                "Статья про нейронные сети и обработку текста. "
                "Предложен новый метод. " * 40
            ),
            mode="baseline",
            summary_mode="abstract",
            use_cache=False,
        )

        assert result["error"] is None
        assert result["summary"]
        assert result["mode"] == "baseline"
        assert fake.calls == 1  # baseline: один вызов LLM

    def test_unknown_user_returns_error(self, eager_celery, worker_db):
        result = tasks_module.summarize(
            user_id="00000000-0000-0000-0000-000000000000",
            article_text="любой текст достаточной длины для теста",
            mode="baseline",
            use_cache=False,
        )
        assert "не найден" in result["error"]


class TestClearOldCache:
    def test_clear_nothing_on_fresh_db(self, eager_celery, worker_db):
        result = tasks_module.clear_old_cache(days=30)
        assert result == {"deleted": 0, "days": 30}

    @pytest.fixture
    async def db_with_old_cache(self, worker_db):
        """Сеет старую и свежую записи кэша (async), тест — sync."""
        from datetime import datetime, timedelta, timezone

        from backend.db.models import SummaryCache

        maker = worker_db["maker"]
        async with maker() as session:
            async with session.begin():
                session.add(SummaryCache(
                    source_text_hash="old" * 16,
                    summary_mode="abstract",
                    output="старый кэш",
                    created_at=datetime.now(timezone.utc) - timedelta(days=60),
                ))
                session.add(SummaryCache(
                    source_text_hash="new" * 16,
                    summary_mode="abstract",
                    output="свежий кэш",
                ))
        return maker

    def test_clear_deletes_old_entries(
        self, eager_celery, db_with_old_cache
    ):
        # Задача синхронная: сеединг сделан async-фикстурой выше.
        result = tasks_module.clear_old_cache(days=30)
        assert result["deleted"] == 1

        async def _check():
            from sqlalchemy import select

            from backend.db.models import SummaryCache

            async with db_with_old_cache() as session:
                remaining = (
                    await session.execute(select(SummaryCache))
                ).scalars().all()
                return remaining

        remaining = tasks_module._run_async(_check())
        assert len(remaining) == 1
        assert remaining[0].output == "свежий кэш"


class TestReindexTask:
    def test_reindex_empty_kb_graceful(self, eager_celery, worker_db):
        """Пустая БЗ и недоступный Qdrant — задача не падает.

        qdrant_available зависит от окружения (локальный режим может
        быть включён в .env) — проверяем только форму результата.
        """
        result = tasks_module.reindex_knowledge_base()

        assert result["kb_entries"] == 0
        assert "qdrant_available" in result
        assert "points_upserted" in result


class TestWorkerLoop:
    def test_persistent_loop_reused(self):
        loop1 = tasks_module._get_worker_loop()
        loop2 = tasks_module._get_worker_loop()
        assert loop1 is loop2

    def test_close_worker_loop(self):
        loop = tasks_module._get_worker_loop()
        tasks_module.close_worker_loop()
        assert loop.is_closed()
        # Повторное использование создаёт новый loop
        new_loop = tasks_module._get_worker_loop()
        assert new_loop is not loop
        assert not new_loop.is_closed()
        tasks_module.close_worker_loop()
