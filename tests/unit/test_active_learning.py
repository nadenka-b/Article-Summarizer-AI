# tests/unit/test_active_learning.py
"""Юнит-тесты ActiveLearningManager (этап 10)."""

import pytest
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.data.knowledge_base import KBManager
from backend.db.base import Base
from backend.db import models  # noqa: F401
from backend.db.models import Article, KnowledgeBase, Summary, User
from backend.hitl.active_learning import ActiveLearningManager


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


async def _seed(session) -> tuple[str, str]:
    """Статья + резюме; возвращает (article_id, summary_id)."""
    user = User(
        email="t@example.com", username="t", hashed_password="x", role="user"
    )
    session.add(user)
    await session.flush()

    article = Article(
        owner_id=user.id,
        title="Тестовая статья",
        source="upload",
        field="ai_ml",
        language="ru",
        full_text="Текст статьи про нейронные сети и их применение.",
        status="summarized",
    )
    session.add(article)
    await session.flush()

    summary = Summary(
        article_id=article.id,
        summary_text="Плохое резюме",
        summary_mode="abstract",
        critic_score=5.0,
        validation_passed=False,
    )
    session.add(summary)
    await session.flush()

    return article.id, summary.id


class TestAddUncertainExample:
    async def test_add_returns_id(self, session_maker):
        async with session_maker() as session:
            async with session.begin():
                _, summary_id = await _seed(session)
                example_id = await ActiveLearningManager.add_uncertain_example(
                    session,
                    summary_id=summary_id,
                    model_output="Плохое резюме",
                    critic_score=5.0,
                    validation_passed=False,
                )
        assert example_id

    async def test_no_duplicate_for_pending(self, session_maker):
        """Повторное добавление того же резюме не создаёт дубль."""
        async with session_maker() as session:
            async with session.begin():
                _, summary_id = await _seed(session)
                first = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary_id, model_output="резюме",
                )

        async with session_maker() as session:
            async with session.begin():
                second = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary_id, model_output="резюме",
                )

        assert first == second


class TestConfirmCorrection:
    async def test_confirm_inserts_kb_pair_and_updates_summary(
        self, session_maker
    ):
        async with session_maker() as session:
            async with session.begin():
                _, summary_id = await _seed(session)
                example_id = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary_id, model_output="Плохое",
                    critic_score=5.0,
                )
                outcome = await ActiveLearningManager.confirm_correction(
                    session,
                    example_id=example_id,
                    corrected_summary="Хорошее исправленное резюме редактора "
                                      "с методами и результатами исследования.",
                )

        assert outcome["kb_inserted"] is True

        async with session_maker() as session:
            example = await ActiveLearningManager.get_example(
                session, example_id
            )
            assert example.status == "confirmed"
            assert example.corrected_summary.startswith("Хорошее")
            assert example.reviewed_at is not None

            summary = await session.get(Summary, summary_id)
            assert summary.summary_text.startswith("Хорошее")

            kb_count = (
                await session.execute(
                    __import__("sqlalchemy").select(KnowledgeBase)
                )
            ).scalars().all()
            assert len(kb_count) == 1
            assert kb_count[0].source == "hitl"
            assert kb_count[0].target_summary.startswith("Хорошее")

    async def test_confirm_twice_conflicts(self, session_maker):
        async with session_maker() as session:
            async with session.begin():
                _, summary_id = await _seed(session)
                example_id = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary_id, model_output="Плохое",
                )
                await ActiveLearningManager.confirm_correction(
                    session, example_id=example_id,
                    corrected_summary="Первое исправление резюме достаточной "
                                      "длины для сохранения в базе знаний.",
                )

        async with session_maker() as session:
            async with session.begin():
                with pytest.raises(ValueError, match="уже обработан"):
                    await ActiveLearningManager.confirm_correction(
                        session, example_id=example_id,
                        corrected_summary="Второе исправление резюме "
                                          "достаточной длины.",
                    )

    async def test_confirm_existing_kb_pair_still_confirms(
        self, session_maker
    ):
        """Пара уже в БЗ → kb_inserted=False, но пример подтверждён."""
        async with session_maker() as session:
            async with session.begin():
                article_id, summary_id = await _seed(session)
                article = await session.get(Article, article_id)

                kb = KBManager()
                await kb.insert_in_session(
                    session, field="ai_ml", language="ru",
                    source_text=article.full_text,
                    target_summary="Уже существующая пара базы знаний "
                                   "с исправленным текстом резюме.",
                )

                example_id = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary_id, model_output="Плохое",
                )
                outcome = await ActiveLearningManager.confirm_correction(
                    session, example_id=example_id,
                    corrected_summary="Уже существующая пара базы знаний "
                                      "с исправленным текстом резюме.",
                )

        assert outcome["kb_inserted"] is False
        async with session_maker() as session:
            example = await ActiveLearningManager.get_example(
                session, example_id
            )
            assert example.status == "confirmed"

    async def test_failure_mid_transaction_creates_nothing(
        self, session_maker, monkeypatch
    ):
        """
        Атомарность: при падении вставки пары БЗ статус остаётся pending,
        пара в БЗ не создаётся.
        """
        async with session_maker() as session:
            async with session.begin():
                _, summary_id = await _seed(session)
                example_id = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary_id, model_output="Плохое",
                )

        async def broken_insert(*args, **kwargs):
            raise RuntimeError("KB down")

        monkeypatch.setattr(
            KBManager, "insert_in_session", broken_insert
        )

        async with session_maker() as session:
            with pytest.raises(RuntimeError, match="KB down"):
                async with session.begin():
                    await ActiveLearningManager.confirm_correction(
                        session, example_id=example_id,
                        corrected_summary="Резюме, которое не должно "
                                          "сохраниться из-за сбоя БЗ.",
                    )

        async with session_maker() as session:
            example = await ActiveLearningManager.get_example(
                session, example_id
            )
            assert example.status == "pending"
            from sqlalchemy import select as sa_select

            kb_rows = (
                await session.execute(sa_select(KnowledgeBase))
            ).scalars().all()
            assert kb_rows == []


class TestReject:
    async def test_reject_no_kb_write(self, session_maker):
        from sqlalchemy import select as sa_select

        async with session_maker() as session:
            async with session.begin():
                _, summary_id = await _seed(session)
                example_id = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary_id, model_output="Плохое",
                )
                await ActiveLearningManager.reject_example(
                    session, example_id=example_id, reason="мусор",
                )

        async with session_maker() as session:
            example = await ActiveLearningManager.get_example(
                session, example_id
            )
            assert example.status == "rejected"
            assert "[REJECT REASON] мусор" in example.model_output

            kb_rows = (
                await session.execute(sa_select(KnowledgeBase))
            ).scalars().all()
            assert kb_rows == []


class TestStats:
    async def test_stats_counts_by_status(self, session_maker):
        async with session_maker() as session:
            async with session.begin():
                _, s1 = await _seed(session)
                e1 = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=s1, model_output="a",
                )
                # Вторая статья → второй пример
                user = User(email="t2@example.com", username="t2",
                            hashed_password="x", role="user")
                session.add(user)
                await session.flush()
                article = Article(
                    owner_id=user.id, title="2", source="upload",
                    full_text="второй текст", status="summarized",
                )
                session.add(article)
                await session.flush()
                summary2 = Summary(article_id=article.id,
                                   summary_text="b", summary_mode="abstract")
                session.add(summary2)
                await session.flush()
                e2 = await ActiveLearningManager.add_uncertain_example(
                    session, summary_id=summary2.id, model_output="b",
                )

                await ActiveLearningManager.confirm_correction(
                    session, example_id=e1,
                    corrected_summary="Исправленное резюме номер один "
                                      "достаточной длины.",
                )
                await ActiveLearningManager.reject_example(
                    session, example_id=e2,
                )

        async with session_maker() as session:
            stats = await ActiveLearningManager.get_stats(session)

        assert stats == {"pending": 0, "confirmed": 1, "rejected": 1}
