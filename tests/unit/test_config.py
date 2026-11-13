# tests/unit/test_config.py
"""Юнит-тесты для backend.core.config."""

import pytest

from backend.core.config import Settings, get_settings


def make_settings(**overrides) -> Settings:
    """Settings без чтения .env."""
    return Settings(_env_file=None, **overrides)


class TestSettingsDefaults:
    def test_project_name_default(self):
        s = make_settings()
        assert s.PROJECT_NAME == "Article-Summarizer-AI"

    def test_admin_email_default(self):
        s = make_settings()
        assert s.ADMIN_EMAIL == "admin@example.com"

    def test_agent_models_default_none(self):
        s = make_settings()
        assert s.SUMMARIZER_MODEL is None
        assert s.CRITIC_MODEL is None
        assert s.EDITOR_MODEL is None

    def test_summary_mode_default(self):
        s = make_settings()
        assert s.SUMMARY_MODE in {"abstract", "section_wise"}

    def test_eval_mode_default(self):
        s = make_settings()
        assert s.EVAL_MODE in {"baseline", "extractor", "rag", "agents", "ensemble"}


class TestAsyncDatabaseUrl:
    def test_sqlite_to_aiosqlite(self):
        s = make_settings(DATABASE_URL="sqlite:///./test.db")
        assert s.ASYNC_DATABASE_URL == "sqlite+aiosqlite:///./test.db"

    def test_postgresql_to_asyncpg(self):
        s = make_settings(DATABASE_URL="postgresql://user:pass@host:5432/db")
        assert s.ASYNC_DATABASE_URL == "postgresql+asyncpg://user:pass@host:5432/db"


class TestUrls:
    def test_lmstudio_urls(self):
        s = make_settings(LMSTUDIO_URL="http://llm.local:1234/v1")
        assert s.lmstudio_chat_url == "http://llm.local:1234/v1/chat/completions"
        assert s.lmstudio_models_url == "http://llm.local:1234/v1/models"

    def test_qdrant_url(self):
        s = make_settings(QDRANT_HOST="qdrant.local", QDRANT_PORT=6334)
        assert s.qdrant_url == "http://qdrant.local:6334"


class TestValidation:
    def test_invalid_eval_mode_raises(self):
        with pytest.raises(ValueError, match="EVAL_MODE"):
            make_settings(EVAL_MODE="invalid")

    def test_invalid_summary_mode_raises(self):
        with pytest.raises(ValueError, match="SUMMARY_MODE"):
            make_settings(SUMMARY_MODE="invalid")


class TestGetSettingsSingleton:
    def test_returns_same_object(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2