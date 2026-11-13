# tests/integration/test_hitl_api.py
"""Интеграционные тесты HITL-интерфейса и админ-роутов (этап 10).

Повышение ролей делается через test_session_maker (тестовая БД),
реальная БД не затрагивается.
"""

import json as json_module

import pytest
from sqlalchemy import select, update

from backend.db.models import KnowledgeBase, User
from backend.db.session import AsyncSessionLocal  # noqa: F401 — не использовать!


CORRECTED = (
    "Исправленное редактором резюме: рассматривается метод нейросетевой "
    "суммаризации, приведены результаты экспериментов и сделаны выводы "
    "о применимости подхода в промышленных системах."
)


async def _register(client, email: str, username: str) -> dict:
    response = await client.post("/auth/register", json={
        "email": email,
        "username": username,
        "password": "Passw0rd123",
    })
    token = response.json()["access_token"]
    return {
        "headers": {"Authorization": f"Bearer {token}"},
        "user_id": _user_id_from_token(token),
    }


def _user_id_from_token(token: str) -> str:
    from jose import jwt

    from backend.core.config import get_settings

    payload = jwt.decode(
        token, get_settings().JWT_SECRET_KEY,
        algorithms=[get_settings().JWT_ALGORITHM],
    )
    return payload["sub"]


async def _set_role(test_session_maker, user_id: str, role: str) -> None:
    async with test_session_maker() as session:
        async with session.begin():
            await session.execute(
                update(User).where(User.id == user_id).values(role=role)
            )


async def _make_hitl_entry(client, monkeypatch) -> dict:
    """Создаёт pending-запись очереди через пайплайн (с моком LLM)."""

    class FakeLLM:
        async def generate(self, prompt, model=None, temperature=0.3,
                           task="generate"):
            if "Оцени резюме" in prompt:
                return (
                    json_module.dumps({
                        "completeness": 4.0, "accuracy": 4.0,
                        "coherence": 4.0, "style": 4.0,
                        "reasoning": "слабо", "suggestions": ["добавь метод"],
                        "missing_terms": [],
                    }),
                    "mock", "mock-model",
                )
            return (
                "This test summary describes a novel scientific method "
                "for article summarization and reports improved quality "
                "metrics. " * 4
            ), "mock", "mock-model"

    monkeypatch.setattr(
        "backend.utils.llm_client.get_llm_client", lambda: FakeLLM()
    )

    owner = await _register(client, "hitl-owner@example.com", "hitlowner")
    response = await client.post(
        "/summarize/text",
        json={
            "text": (
                "Научная статья про нейронные сети и методы обработки "
                "текста. Предложен новый подход к суммаризации. " * 30
            ),
            "mode": "agents",
            "use_cache": False,
        },
        headers=owner["headers"],
    )
    assert response.status_code == 200
    assert response.json()["hitl_created"] is True
    return owner


class TestAccessControl:
    async def test_queue_requires_auth(self, client):
        response = await client.get("/hitl/queue")
        assert response.status_code == 401

    async def test_queue_forbidden_for_plain_user(self, client):
        user = await _register(client, "plain@example.com", "plain")
        response = await client.get("/hitl/queue", headers=user["headers"])
        assert response.status_code == 403

    async def test_admin_routes_forbidden_for_user(self, client):
        user = await _register(client, "plain2@example.com", "plain2")
        response = await client.get("/admin/users", headers=user["headers"])
        assert response.status_code == 403


class TestAdminRoles:
    async def test_change_role_to_editor(
        self, client, test_session_maker
    ):
        admin = await _register(client, "boss@example.com", "boss")
        await _set_role(test_session_maker, admin["user_id"], "admin")

        # Админ не может снять роль admin с себя
        self_demote = await client.patch(
            f"/admin/users/{admin['user_id']}/role",
            json={"role": "editor"},
            headers=admin["headers"],
        )
        assert self_demote.status_code == 400

        # Зато может повысить другого
        worker = await _register(client, "worker@example.com", "worker")
        promoted = await client.patch(
            f"/admin/users/{worker['user_id']}/role",
            json={"role": "editor"},
            headers=admin["headers"],
        )
        assert promoted.status_code == 200
        assert promoted.json()["role"] == "editor"

        # Список пользователей доступен админу
        users = await client.get("/admin/users", headers=admin["headers"])
        assert users.status_code == 200
        assert any(u["role"] == "editor" for u in users.json())

    async def test_invalid_role_rejected(
        self, client, test_session_maker
    ):
        admin = await _register(client, "boss2@example.com", "boss2")
        await _set_role(test_session_maker, admin["user_id"], "admin")

        response = await client.patch(
            f"/admin/users/{admin['user_id']}/role",
            json={"role": "superuser"},
            headers=admin["headers"],
        )
        assert response.status_code == 422

    async def test_missing_user_404(self, client, test_session_maker):
        admin = await _register(client, "boss3@example.com", "boss3")
        await _set_role(test_session_maker, admin["user_id"], "admin")

        response = await client.patch(
            "/admin/users/00000000-0000-0000-0000-000000000000/role",
            json={"role": "editor"},
            headers=admin["headers"],
        )
        assert response.status_code == 404


class TestHitlFlow:
    async def test_full_confirm_flow(
        self, client, monkeypatch, test_session_maker
    ):
        owner = await _make_hitl_entry(client, monkeypatch)
        await _set_role(test_session_maker, owner["user_id"], "editor")

        queue = await client.get("/hitl/queue", headers=owner["headers"])
        assert queue.status_code == 200
        items = queue.json()
        assert len(items) == 1
        example_id = items[0]["id"]
        assert items[0]["status"] == "pending"

        # Детали с текстом статьи
        detail = await client.get(
            f"/hitl/queue/{example_id}", headers=owner["headers"]
        )
        assert detail.status_code == 200
        assert detail.json()["article_text"]

        # Подтверждение правки
        confirm = await client.post(
            f"/hitl/queue/{example_id}/confirm",
            json={"corrected_summary": CORRECTED},
            headers=owner["headers"],
        )
        assert confirm.status_code == 200
        body = confirm.json()
        assert body["status"] == "confirmed"
        assert body["kb_inserted"] is True

        # Повторное подтверждение → 409
        again = await client.post(
            f"/hitl/queue/{example_id}/confirm",
            json={"corrected_summary": CORRECTED},
            headers=owner["headers"],
        )
        assert again.status_code == 409

        # Очередь пуста, статистика сошлась
        stats = await client.get("/hitl/stats", headers=owner["headers"])
        assert stats.json() == {"pending": 0, "confirmed": 1, "rejected": 0}

        # Эталонная пара легла в БЗ с пометкой hitl
        async with test_session_maker() as session:
            kb_rows = (
                await session.execute(
                    select(KnowledgeBase).where(
                        KnowledgeBase.source == "hitl"
                    )
                )
            ).scalars().all()
            assert len(kb_rows) == 1
            assert kb_rows[0].target_summary == CORRECTED

    async def test_reject_flow(
        self, client, monkeypatch, test_session_maker
    ):
        owner = await _make_hitl_entry(client, monkeypatch)
        await _set_role(test_session_maker, owner["user_id"], "editor")

        queue = await client.get("/hitl/queue", headers=owner["headers"])
        example_id = queue.json()[0]["id"]

        reject = await client.post(
            f"/hitl/queue/{example_id}/reject",
            json={"reason": "резюме нерелевантно"},
            headers=owner["headers"],
        )
        assert reject.status_code == 200
        assert reject.json()["status"] == "rejected"

        stats = await client.get("/hitl/stats", headers=owner["headers"])
        assert stats.json()["pending"] == 0
        assert stats.json()["rejected"] == 1

        # В БЗ ничего не попало
        async with test_session_maker() as session:
            kb_rows = (
                await session.execute(select(KnowledgeBase))
            ).scalars().all()
            assert kb_rows == []

    async def test_missing_example_404(
        self, client, test_session_maker
    ):
        editor = await _register(client, "ed@example.com", "editor1")
        await _set_role(test_session_maker, editor["user_id"], "editor")

        response = await client.get(
            "/hitl/queue/00000000-0000-0000-0000-000000000000",
            headers=editor["headers"],
        )
        assert response.status_code == 404

    async def test_short_correction_rejected(
        self, client, monkeypatch, test_session_maker
    ):
        """Правка короче 50 символов не принимается схемой."""
        owner = await _make_hitl_entry(client, monkeypatch)
        await _set_role(test_session_maker, owner["user_id"], "editor")

        queue = await client.get("/hitl/queue", headers=owner["headers"])
        example_id = queue.json()[0]["id"]

        response = await client.post(
            f"/hitl/queue/{example_id}/confirm",
            json={"corrected_summary": "коротко"},
            headers=owner["headers"],
        )
        assert response.status_code == 422
