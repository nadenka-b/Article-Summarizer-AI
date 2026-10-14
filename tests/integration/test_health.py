# tests/integration/test_health.py
"""Интеграционные тесты для healthcheck-эндпоинтов."""

from fastapi.testclient import TestClient

from backend.api.main import app


client = TestClient(app)


class TestRoot:
    def test_root_returns_200(self):
        assert client.get("/").status_code == 200

    def test_root_contains_keys(self):
        data = client.get("/").json()
        assert "message" in data
        assert "docs" in data


class TestHealthz:
    def test_healthz_returns_200(self):
        assert client.get("/healthz").status_code == 200

    def test_healthz_returns_ok(self):
        assert client.get("/healthz").json() == {"status": "ok"}


class TestFields:
    def test_fields_returns_200(self):
        assert client.get("/fields").status_code == 200

    def test_fields_contains_cs(self):
        data = client.get("/fields").json()
        assert "fields" in data
        assert "cs" in data["fields"]

    def test_fields_contains_sections(self):
        data = client.get("/fields").json()
        assert "sections" in data
        assert "abstract" in data["sections"]

    def test_fields_contains_metrics(self):
        data = client.get("/fields").json()
        assert "eval_metrics" in data
        assert "rouge_1" in data["eval_metrics"]