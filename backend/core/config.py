# backend/core/config.py
"""
Конфигурация приложения Article-Summarizer-AI.

Все настройки читаются из .env файла и переменных окружения.
Порядок приоритета:
1. Переменные окружения (заданы в shell)
2. Файл .env в корне проекта
3. Значения по умолчанию (default) в этом классе
"""

from typing import List, Optional
from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """
    Все настройки приложения.

    Каждое поле соответствует переменной в .env.
    Если переменная не задана — используется default.
    """

    # ========================================================================
    # Application
    # ========================================================================
    PROJECT_NAME: str = Field(
        default="Article-Summarizer-AI",
        validation_alias="PROJECT_NAME",
    )
    DEBUG: bool = Field(default=True, validation_alias="DEBUG")
    EVAL_MODE: str = Field(default="agents", validation_alias="EVAL_MODE")
    ACTIVE_LEARNING_ENABLED: bool = Field(
        default=True, validation_alias="ACTIVE_LEARNING_ENABLED"
    )

    # ========================================================================
    # Summarization engine
    # ========================================================================
    SUMMARIZATION_ENGINE: str = Field(
        default="auto", validation_alias="SUMMARIZATION_ENGINE"
    )
    FALLBACK_TO_HF_ON_ERROR: bool = Field(
        default=True, validation_alias="FALLBACK_TO_HF_ON_ERROR"
    )
    SUMMARY_MODE: str = Field(
        default="abstract", validation_alias="SUMMARY_MODE"
    )  # abstract | section_wise
    MAX_ARTICLE_LENGTH: int = Field(
        default=50000, validation_alias="MAX_ARTICLE_LENGTH"
    )

    # ========================================================================
    # Database
    # ========================================================================
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./summarizer.db",
        validation_alias="DATABASE_URL",
    )

    # ========================================================================
    # Ollama (локальный LLM-сервер)
    # ========================================================================
    OLLAMA_HOST: str = Field(
        default="localhost", validation_alias="OLLAMA_HOST"
    )
    OLLAMA_PORT: int = Field(default=11434, validation_alias="OLLAMA_PORT")

    LLM_MODEL: str = Field(default="qwen2.5:7b", validation_alias="LLM_MODEL")
    SUMMARIZER_MODEL: Optional[str] = Field(
        default=None, validation_alias="SUMMARIZER_MODEL"
    )
    CRITIC_MODEL: Optional[str] = Field(
        default=None, validation_alias="CRITIC_MODEL"
    )
    EDITOR_MODEL: Optional[str] = Field(
        default=None, validation_alias="EDITOR_MODEL"
    )
    VALIDATOR_MODEL: Optional[str] = Field(
        default=None, validation_alias="VALIDATOR_MODEL"
    )
    ENSEMBLE_MODELS: str = Field(
        default="qwen2.5:7b,llama3.1:8b",
        validation_alias="ENSEMBLE_MODELS",
    )

    # ========================================================================
    # HuggingFace (запасной движок)
    # ========================================================================
    HF_TOKEN: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")
    HF_SUMMARIZATION_MODEL: str = Field(
        default="facebook/bart-large-cnn",
        validation_alias="HF_SUMMARIZATION_MODEL",
    )
    HF_CRITIC_MODEL: Optional[str] = Field(
        default=None, validation_alias="HF_CRITIC_MODEL"
    )
    HF_EDITOR_MODEL: Optional[str] = Field(
        default=None, validation_alias="HF_EDITOR_MODEL"
    )
    HF_VALIDATOR_MODEL: Optional[str] = Field(
        default=None, validation_alias="HF_VALIDATOR_MODEL"
    )

    # ========================================================================
    # Qdrant (векторная БД)
    # ========================================================================
    QDRANT_HOST: str = Field(
        default="localhost", validation_alias="QDRANT_HOST"
    )
    QDRANT_PORT: int = Field(default=6333, validation_alias="QDRANT_PORT")
    QDRANT_COLLECTION_NAME: str = Field(
        default="article_knowledge",
        validation_alias="QDRANT_COLLECTION_NAME",
    )
    QDRANT_API_KEY: Optional[str] = Field(
        default=None, validation_alias="QDRANT_API_KEY"
    )

    # ========================================================================
    # Redis (брокер для Celery)
    # ========================================================================
    REDIS_URL: str = Field(
        default="redis://localhost:6379/0", validation_alias="REDIS_URL"
    )

    # ========================================================================
    # Security (JWT)
    # ========================================================================
    JWT_SECRET_KEY: str = Field(
        default="change-me-in-production",
        validation_alias="JWT_SECRET_KEY",
    )
    JWT_ALGORITHM: str = Field(
        default="HS256", validation_alias="JWT_ALGORITHM"
    )
    JWT_EXPIRATION_MINUTES: int = Field(
        default=60 * 24 * 7, validation_alias="JWT_EXPIRATION_MINUTES"
    )

    # ========================================================================
    # Properties (вычисляемые поля)
    # ========================================================================

    @property
    def ASYNC_DATABASE_URL(self) -> str:
        """
        Приводит DATABASE_URL к формату, который понимает SQLAlchemy async.

        Примеры:
        - postgresql://user:pass@host:5432/db  → postgresql+asyncpg://...
        - sqlite:///./summarizer.db            → sqlite+aiosqlite:///...
        """
        url = self.DATABASE_URL
        if url.startswith("postgresql://"):
            return url.replace("postgresql://", "postgresql+asyncpg://", 1)
        if url.startswith("sqlite:///"):
            return url.replace("sqlite:///", "sqlite+aiosqlite:///", 1)
        return url

    @property
    def ollama_url(self) -> str:
        """Полный URL для API Ollama."""
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}/api/generate"

    @property
    def qdrant_url(self) -> str:
        """Полный URL Qdrant."""
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def ensemble_models_list(self) -> List[str]:
        """
        Список моделей для ансамбля, распарсенный из ENSEMBLE_MODELS.

        ENSEMBLE_MODELS="qwen2.5:7b,llama3.1:8b"
        → ["qwen2.5:7b", "llama3.1:8b"]
        """
        if not self.ENSEMBLE_MODELS:
            return []
        return [m.strip() for m in self.ENSEMBLE_MODELS.split(",") if m.strip()]

    # ========================================================================
    # Validators
    # ========================================================================

    @field_validator("SUMMARIZATION_ENGINE")
    @classmethod
    def validate_summarization_engine(cls, v: str) -> str:
        allowed = {"auto", "ollama", "huggingface"}
        if v not in allowed:
            raise ValueError(
                f"SUMMARIZATION_ENGINE должен быть одним из {allowed}, получено: {v!r}"
            )
        return v

    @field_validator("EVAL_MODE")
    @classmethod
    def validate_eval_mode(cls, v: str) -> str:
        allowed = {"baseline", "extractor", "rag", "agents", "ensemble"}
        if v not in allowed:
            raise ValueError(
                f"EVAL_MODE должен быть одним из {allowed}, получено: {v!r}"
            )
        return v

    @field_validator("SUMMARY_MODE")
    @classmethod
    def validate_summary_mode(cls, v: str) -> str:
        allowed = {"abstract", "section_wise"}
        if v not in allowed:
            raise ValueError(
                f"SUMMARY_MODE должен быть одним из {allowed}, получено: {v!r}"
            )
        return v

    # ========================================================================
    # Pydantic config
    # ========================================================================

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Возвращает singleton-экземпляр настроек.

    @lru_cache гарантирует, что .env будет прочитан один раз,
    а последующие вызовы вернут уже созданный объект.
    """
    return Settings()