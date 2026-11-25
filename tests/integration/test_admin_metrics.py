# tests/integration/test_admin_metrics.py
"""Интеграционные тесты GET /admin/metrics (отчёты eval в UI, этап 12+).

Каталог отчётов подменяется на tmp_path через monkeypatch атрибута
singleton-настроек — реальные data/eval не затрагиваются.
"""

import json

from backend.core.config import get_settings
from backend.db.models import User  # noqa: F401 — регистрация моделей


async def _register(client, email: str, username: str) -> dict:
    response = await client.post("/auth/register", json={
        "email": email,
        "username": username,
        "password": "Passw0rd123",
    })
    token = response.json()["access_token"]
    return {"headers": {"Authorization": f"Bearer {token}"}}


async def _promote(test_session_maker, client, headers) -> None:
    """Повышает текущего пользователя до admin прямо в тестовой БД."""
    from jose import jwt

    from sqlalchemy import update

    token = headers["Authorization"].removeprefix("Bearer ")
    payload = jwt.decode(
        token, get_settings().JWT_SECRET_KEY,
        algorithms=[get_settings().JWT_ALGORITHM],
    )
    async with test_session_maker() as session:
        async with session.begin():
            await session.execute(
                update(User).where(User.id == payload["sub"]).values(role="admin")
            )


async def test_metrics_requires_admin(client):
    """Не-админ получает 403."""
    user = await _register(client, "user@example.com", "plainuser")
    response = await client.get("/admin/metrics", headers=user["headers"])
    assert response.status_code == 403


async def test_metrics_empty_dir(client, test_session_maker, monkeypatch, tmp_path):
    """Админ получает 200 и подсказку при пустом каталоге отчётов."""
    admin = await _register(client, "admin1@example.com", "admin1")
    await _promote(test_session_maker, client, admin["headers"])
    monkeypatch.setattr(get_settings(), "EVAL_REPORTS_DIR", str(tmp_path))
    response = await client.get("/admin/metrics", headers=admin["headers"])
    assert response.status_code == 200
    body = response.json()
    assert body["reports"] == []
    assert "run_eval" in body["hint"]


async def test_metrics_returns_reports(
    client, test_session_maker, monkeypatch, tmp_path
):
    """Отчёт из каталога возвращается с режимом и метриками."""
    admin = await _register(client, "admin2@example.com", "admin2")
    await _promote(test_session_maker, client, admin["headers"])
    monkeypatch.setattr(get_settings(), "EVAL_REPORTS_DIR", str(tmp_path))

    sample = {
        "metrics": {
            "n": 2, "mode": "baseline",
            "rouge_1": 0.42, "rouge_2": 0.11, "rouge_l": 0.31,
            "coverage": 0.7, "bleu": 5.0, "chrf": 30.2,
        },
        "per_article": [
            {"number": 1, "title": "Статья", "rouge_1": 0.4,
             "rouge_l": 0.3, "coverage": 0.7, "hypothesis": "…",
             "reference": "…"}
        ],
    }
    (tmp_path / "eval_baseline_2.json").write_text(
        json.dumps(sample, ensure_ascii=False), encoding="utf-8"
    )
    # битый файл должен быть молча пропущен
    (tmp_path / "eval_broken_1.json").write_text("{не json", encoding="utf-8")

    response = await client.get("/admin/metrics", headers=admin["headers"])
    assert response.status_code == 200
    reports = response.json()["reports"]
    assert len(reports) == 1
    assert reports[0]["mode"] == "baseline"
    assert reports[0]["metrics"]["bleu"] == 5.0
    assert len(reports[0]["per_article"]) == 1
