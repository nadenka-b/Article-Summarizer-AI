# tests/integration/test_summarize_api.py
"""Интеграционные тесты API суммаризации (этап 9).

LLM подменяется заглушкой — реальные вызовы LM Studio не выполняются.
"""

import json

import pytest

from backend.db.models import Article, Summary, UncertainExample


# ============================================================================
# Заглушка LLM
# ============================================================================
class FakeLLMClient:
    def __init__(self):
        self.default = ""
        self.handler = None
        self.calls = 0

    async def generate(self, prompt, model=None, temperature=0.3, task="generate"):
        self.calls += 1
        if self.handler is not None:
            text = self.handler(prompt)
        else:
            text = self.default
        return text, "mock", "mock-model"


def critic_json(score: float) -> str:
    return json.dumps({
        "completeness": score, "accuracy": score,
        "coherence": score, "style": score,
        "reasoning": "ok", "suggestions": [], "missing_terms": [],
    })


SUMMARY_TEXT = (
    "This test summary describes a novel scientific method "
    "for article summarization and reports improved quality metrics. "
) * 4  # 64 слова — проходит валидатор (мин. 50)

ARTICLE_TEXT = (
    "Научная статья про нейронные сети и методы обработки текста. "
    "Предложен новый подход к суммаризации. "
) * 60  # >200 символов


@pytest.fixture
def fake_llm(monkeypatch):
    client = FakeLLMClient()
    monkeypatch.setattr(
        "backend.utils.llm_client.get_llm_client", lambda: client
    )
    return client


@pytest.fixture
async def auth_headers(client):
    """Регистрирует пользователя и возвращает заголовки с токеном."""
    response = await client.post("/auth/register", json={
        "email": "sumer@example.com",
        "username": "sumer",
        "password": "Passw0rd123",
    })
    token = response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


# ============================================================================
# Тесты
# ============================================================================
class TestSummarizeText:
    async def test_requires_auth(self, client):
        response = await client.post(
            "/summarize/text", json={"text": ARTICLE_TEXT}
        )
        assert response.status_code == 401

    async def test_text_too_short_rejected(self, client, auth_headers):
        response = await client.post(
            "/summarize/text",
            json={"text": "коротко"},
            headers=auth_headers,
        )
        assert response.status_code == 422

    async def test_agents_mode_full_cycle(
        self, client, auth_headers, fake_llm, test_session_maker
    ):
        def handler(prompt: str) -> str:
            if "Оцени резюме" in prompt:  # Critic
                return critic_json(9.0)
            return SUMMARY_TEXT  # Summarizer / Editor

        fake_llm.handler = handler

        response = await client.post(
            "/summarize/text",
            json={"text": ARTICLE_TEXT, "mode": "agents"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["summary"] == SUMMARY_TEXT.strip()
        assert body["mode"] == "agents"
        assert body["critic_score"] == 9.0
        assert body["final_reason"] == "passed"
        assert body["cached"] is False
        assert body["hitl_created"] is False
        assert body["article_id"]
        assert body["summary_id"]
        assert len(body["reasoning_log"]) >= 3

        # Данные легли в БД
        async with test_session_maker() as session:
            summaries = (await session.execute(select_all(Summary))).scalars().all()
            assert len(summaries) == 1
            assert summaries[0].critic_score == 9.0

            articles = (await session.execute(select_all(Article))).scalars().all()
            assert len(articles) == 1
            assert articles[0].owner_id is not None
            assert articles[0].full_text == ARTICLE_TEXT

    async def test_baseline_mode_no_agents(
        self, client, auth_headers, fake_llm
    ):
        fake_llm.default = SUMMARY_TEXT

        response = await client.post(
            "/summarize/text",
            json={
                "text": ARTICLE_TEXT,
                "mode": "baseline",
                "use_cache": False,
            },
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["mode"] == "baseline"
        assert body["final_reason"] == "baseline"
        # Только один вызов LLM — Summarizer, без Critic
        assert fake_llm.calls == 1

    async def test_cache_hit_on_second_request(
        self, client, auth_headers, fake_llm
    ):
        fake_llm.default = SUMMARY_TEXT
        payload = {"text": ARTICLE_TEXT, "mode": "baseline"}

        first = await client.post(
            "/summarize/text", json=payload, headers=auth_headers
        )
        assert first.status_code == 200
        assert first.json()["cached"] is False

        calls_after_first = fake_llm.calls

        second = await client.post(
            "/summarize/text", json=payload, headers=auth_headers
        )
        assert second.status_code == 200
        assert second.json()["cached"] is True
        assert second.json()["summary"] == SUMMARY_TEXT.strip()
        # LLM не дёргался повторно
        assert fake_llm.calls == calls_after_first

    async def test_low_quality_goes_to_hitl(
        self, client, auth_headers, fake_llm, test_session_maker
    ):
        def handler(prompt: str) -> str:
            if "Оцени резюме" in prompt:
                return critic_json(5.0)  # ниже порога 7.0
            return SUMMARY_TEXT

        fake_llm.handler = handler

        response = await client.post(
            "/summarize/text",
            json={"text": ARTICLE_TEXT, "mode": "agents", "use_cache": False},
            headers=auth_headers,
        )

        assert response.status_code == 200
        assert response.json()["hitl_created"] is True

        async with test_session_maker() as session:
            queue = (
                await session.execute(select_all(UncertainExample))
            ).scalars().all()
            assert len(queue) == 1
            assert queue[0].status == "pending"

    async def test_invalid_mode_rejected(self, client, auth_headers):
        response = await client.post(
            "/summarize/text",
            json={"text": ARTICLE_TEXT, "mode": "wrong"},
            headers=auth_headers,
        )
        assert response.status_code == 422


class TestSummarizePdf:
    async def test_pdf_upload(
        self, client, auth_headers, fake_llm
    ):
        import pymupdf

        # Генерируем минимальный PDF с текстом
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((72, 72), ARTICLE_TEXT)
        pdf_bytes = doc.tobytes()
        doc.close()

        fake_llm.default = SUMMARY_TEXT

        response = await client.post(
            "/summarize/pdf",
            files={"file": ("article.pdf", pdf_bytes, "application/pdf")},
            params={"mode": "baseline", "use_cache": "false"},
            headers=auth_headers,
        )

        assert response.status_code == 200
        body = response.json()
        assert body["summary"] == SUMMARY_TEXT.strip()
        assert body["mode"] == "baseline"

    async def test_not_a_pdf_rejected(self, client, auth_headers):
        response = await client.post(
            "/summarize/pdf",
            files={"file": ("f.txt", b"hello", "text/plain")},
            headers=auth_headers,
        )
        assert response.status_code == 400


class TestGetSummary:
    async def test_get_own_summary(self, client, auth_headers, fake_llm):
        fake_llm.default = SUMMARY_TEXT

        created = await client.post(
            "/summarize/text",
            json={"text": ARTICLE_TEXT, "mode": "baseline", "use_cache": False},
            headers=auth_headers,
        )
        summary_id = created.json()["summary_id"]

        fetched = await client.get(
            f"/summaries/{summary_id}", headers=auth_headers
        )
        assert fetched.status_code == 200
        assert fetched.json()["summary_text"] == SUMMARY_TEXT.strip()

    async def test_get_missing_summary_404(self, client, auth_headers):
        response = await client.get(
            "/summaries/00000000-0000-0000-0000-000000000000",
            headers=auth_headers,
        )
        assert response.status_code == 404

    async def test_other_users_summary_forbidden(
        self, client, fake_llm
    ):
        fake_llm.default = SUMMARY_TEXT

        owner_headers = await _register(client, "owner@example.com", "owner1")
        created = await client.post(
            "/summarize/text",
            json={"text": ARTICLE_TEXT, "mode": "baseline", "use_cache": False},
            headers=owner_headers,
        )
        summary_id = created.json()["summary_id"]

        stranger_headers = await _register(
            client, "stranger@example.com", "stranger1"
        )
        stranger_view = await client.get(
            f"/summaries/{summary_id}", headers=stranger_headers
        )
        assert stranger_view.status_code == 403


# ============================================================================
# Хелперы
# ============================================================================
def select_all(model):
    from sqlalchemy import select

    return select(model)


async def _register(client, email: str, username: str) -> dict:
    response = await client.post("/auth/register", json={
        "email": email,
        "username": username,
        "password": "Passw0rd123",
    })
    return {"Authorization": f"Bearer {response.json()['access_token']}"}
