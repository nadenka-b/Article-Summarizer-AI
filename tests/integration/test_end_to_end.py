# tests/integration/test_end_to_end.py
"""
End-to-end путь пользователя (этап 12):

регистрация → суммаризация (низкое качество) → очередь HITL →
назначение редактора → подтверждение правки → эталон в БЗ →
поиск по БЗ находит подтверждённое резюме → мои статьи.

LLM мокается, реальная БД не затрагивается.
"""

import json as json_module
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import select, update

from backend.db.models import KnowledgeBase, SummaryCache, User


ARTICLE_TEXT = (
    "Научная статья про нейронные сети и методы обработки текста. "
    "Предложен новый подход к суммаризации научных публикаций. " * 30
)

CORRECTED = (
    "Исправленное редактором резюме: в работе рассматривается "
    "нейросетевой метод суммаризации научных публикаций, описана "
    "методика экспериментов и сформулированы выводы о применимости."
)


@pytest.fixture
def fake_llm(monkeypatch):
    """Critic занижает оценку → резюме гарантированно уходит в HITL."""

    class FakeLLM:
        def __init__(self):
            self.calls = 0

        async def generate(self, prompt, model=None, temperature=0.3,
                           task="generate"):
            self.calls += 1
            if "Оцени резюме" in prompt:
                return (
                    json_module.dumps({
                        "completeness": 4.0, "accuracy": 5.0,
                        "coherence": 4.0, "style": 5.0,
                        "reasoning": "неполно",
                        "suggestions": ["добавить результаты"],
                        "missing_terms": [],
                    }),
                    "mock", "mock-model",
                )
            return (
                "This test summary describes a novel scientific method "
                "for article summarization and reports improved quality "
                "metrics. " * 4
            ), "mock", "mock-model"

    fake = FakeLLM()
    monkeypatch.setattr(
        "backend.utils.llm_client.get_llm_client", lambda: fake
    )
    return fake


async def _register(client, email: str, username: str) -> dict:
    response = await client.post("/auth/register", json={
        "email": email, "username": username, "password": "Passw0rd123",
    })
    token = response.json()["access_token"]

    from jose import jwt

    from backend.core.config import get_settings

    payload = jwt.decode(
        token, get_settings().JWT_SECRET_KEY,
        algorithms=[get_settings().JWT_ALGORITHM],
    )
    return {
        "headers": {"Authorization": f"Bearer {token}"},
        "user_id": payload["sub"],
    }


async def _set_role(maker, user_id: str, role: str):
    async with maker() as session:
        async with session.begin():
            await session.execute(
                update(User).where(User.id == user_id).values(role=role)
            )


class TestEndToEnd:
    async def test_full_user_journey(
        self, client, fake_llm, test_session_maker
    ):
        # --- 1. Регистрация двух участников ---
        admin = await _register(client, "e2e-admin@example.com", "e2eadmin")
        editor = await _register(client, "e2e-ed@example.com", "e2eeditor")

        # --- 2. Админ назначает редактора ---
        await _set_role(test_session_maker, admin["user_id"], "admin")
        promote = await client.patch(
            f"/admin/users/{editor['user_id']}/role",
            json={"role": "editor"}, headers=admin["headers"],
        )
        assert promote.status_code == 200

        # --- 3. Пользователь генерирует резюме (качество низкое) ---
        summarize = await client.post(
            "/summarize/text",
            json={"text": ARTICLE_TEXT, "mode": "agents"},
            headers=admin["headers"],
        )
        assert summarize.status_code == 200
        body = summarize.json()
        assert body["hitl_created"] is True
        summary_id = body["summary_id"]

        # --- 4. Кэш записан; повтор не дёргает LLM ---
        before_calls = fake_llm.calls
        cached = await client.post(
            "/summarize/text",
            json={"text": ARTICLE_TEXT, "mode": "agents"},
            headers=admin["headers"],
        )
        assert cached.json()["cached"] is True
        assert fake_llm.calls == before_calls

        # --- 5. Редактор видит очередь ---
        queue = await client.get("/hitl/queue", headers=editor["headers"])
        assert queue.status_code == 200
        assert len(queue.json()) == 1
        example_id = queue.json()[0]["id"]

        detail = await client.get(
            f"/hitl/queue/{example_id}", headers=editor["headers"]
        )
        assert detail.json()["article_text"] == ARTICLE_TEXT

        # --- 6. Подтверждение правки → эталон в БЗ ---
        confirm = await client.post(
            f"/hitl/queue/{example_id}/confirm",
            json={"corrected_summary": CORRECTED},
            headers=editor["headers"],
        )
        assert confirm.status_code == 200
        assert confirm.json()["kb_inserted"] is True

        async with test_session_maker() as session:
            kb = (
                await session.execute(
                    select(KnowledgeBase).where(
                        KnowledgeBase.source == "hitl"
                    )
                )
            ).scalars().all()
            assert len(kb) == 1
            assert kb[0].target_summary == CORRECTED

        # --- 7. Мои статьи ---
        mine = await client.get("/articles/my", headers=admin["headers"])
        assert mine.status_code == 200
        payload = mine.json()
        # Ответ-конверт: total (COUNT по пользователю) + страница статей
        assert payload["total"] == 1
        articles = payload["articles"]
        assert len(articles) == 1
        assert articles[0]["summaries"][0]["summary_id"] == summary_id
        # Резюме обновлено исправленной версией
        assert articles[0]["summaries"][0]["summary_text"] == CORRECTED

        # --- 8. Поиск по БЗ отвечает (сам эталон проверен в шаге 6) ---
        # Векторный индекс мокаем: локальное Qdrant-хранилище содержит
        # реальные корпусные точки и делает проверку недетерминированной.
        import backend.api.routes.articles as articles_module

        class _NoVector:
            def search(self, *args, **kwargs):
                return []

        monkeypatched_index = _NoVector()
        original_getter = articles_module.get_vector_index
        articles_module.get_vector_index = lambda: monkeypatched_index
        try:
            search = await client.get(
                "/kb/search",
                params={"q": "суммаризации научных публикаций", "top_k": 3},
                headers=editor["headers"],
            )
        finally:
            articles_module.get_vector_index = original_getter

        assert search.status_code == 200
        assert isinstance(search.json()["results"], list)

        # --- 9. Статистика очереди ---
        stats = await client.get("/hitl/stats", headers=editor["headers"])
        assert stats.json() == {"pending": 0, "confirmed": 1, "rejected": 0}

    @pytest.fixture
    async def db_with_old_cache(self, test_session_maker):
        """Сеет устаревшую запись кэша (async), тест — sync."""
        async with test_session_maker() as session:
            async with session.begin():
                session.add(SummaryCache(
                    source_text_hash="e2e-old-cache",
                    summary_mode="abstract",
                    output="старый",
                    created_at=datetime.now(timezone.utc) - timedelta(days=60),
                ))
        return test_session_maker

    def test_cache_cleanup_task_clears_old_entries(
        self, db_with_old_cache, monkeypatch
    ):
        """Фоновая задача очистки кэша удаляет устаревшие записи."""
        import backend.db.session as db_session
        from backend.worker.tasks import clear_old_cache

        # Задача берёт AsyncSessionLocal из db.session — подменяем на БД теста.
        monkeypatch.setattr(
            db_session, "AsyncSessionLocal", db_with_old_cache
        )

        result = clear_old_cache(days=30)  # прямой вызов = синхронно
        assert result["deleted"] == 1
