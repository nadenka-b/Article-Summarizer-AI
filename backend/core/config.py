# backend/core/config.py
"""
Конфигурация приложения Article-Summarizer-AI.

Все настройки читаются из .env файла и переменных окружения.
Порядок приоритета:
1. Переменные окружения (заданы в shell)
2. Файл .env в корне проекта
3. Значения по умолчанию (default) в этом классе
"""

from typing import Optional
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
    # Каталог JSON-отчётов scripts/eval/run_eval.py (читается /admin/metrics):
    EVAL_REPORTS_DIR: str = Field(
        default="data/eval", validation_alias="EVAL_REPORTS_DIR"
    )
    ACTIVE_LEARNING_ENABLED: bool = Field(
        default=True, validation_alias="ACTIVE_LEARNING_ENABLED"
    )
    ADMIN_EMAIL: str = Field(
        default="admin@example.com",
        validation_alias="ADMIN_EMAIL",
    )
    # Порог направления резюме в очередь HITL: ниже этого балла Critic
    # (или провал валидации) — резюме уходит редактору. Подними до 9.5,
    # чтобы проверить HITL-контур вживую на любой генерации.
    HITL_CRITIC_THRESHOLD: float = Field(
        default=7.0, validation_alias="HITL_CRITIC_THRESHOLD"
    )

    # ========================================================================
    # Summarization
    # ========================================================================
    # Режим по умолчанию (используется на этапе 9 в API):
    SUMMARY_MODE: str = Field(
        default="abstract", validation_alias="SUMMARY_MODE"
    )  # abstract | section_wise
    # Чанкинг (этап 8): текст длиннее SUMMARIZER_DIRECT_CHARS разбивается
    # на чанки по предложениям с перекрытием:
    SUMMARIZER_DIRECT_CHARS: int = Field(
        default=6000, validation_alias="SUMMARIZER_DIRECT_CHARS"
    )
    CHUNK_SIZE_CHARS: int = Field(
        default=3500, validation_alias="CHUNK_SIZE_CHARS"
    )
    CHUNK_OVERLAP_CHARS: int = Field(
        default=300, validation_alias="CHUNK_OVERLAP_CHARS"
    )

    # ========================================================================
    # Database
    # ========================================================================
    DATABASE_URL: str = Field(
        default="sqlite+aiosqlite:///./summarizer.db",
        validation_alias="DATABASE_URL",
    )

    # ========================================================================
    # LLM-движок: LM Studio (локально) или OpenRouter (облако).
    # Оба говорят по OpenAI-совместимому API, выбор — переменной LLM_ENGINE.
    # ========================================================================
    LLM_ENGINE: str = Field(
        default="lmstudio", validation_alias="LLM_ENGINE"
    )
    LMSTUDIO_URL: str = Field(
        default="http://localhost:1234/v1",
        validation_alias="LMSTUDIO_URL",
    )
    LMSTUDIO_MODEL: str = Field(
        default="qwen2.5-7b-instruct-1m",
        validation_alias="LMSTUDIO_MODEL",
    )
    # Таймаут одного вызова LLM (локальная генерация может быть медленной):
    LLM_TIMEOUT_SECONDS: int = Field(
        default=600, validation_alias="LLM_TIMEOUT_SECONDS"
    )
    # Верхний предел генерации — защита от «бесконечного размышления»
    # thinking-моделей (6000 токенов ≈ 8 минут при 12 ток/с):
    LLM_MAX_TOKENS: int = Field(
        default=6000, validation_alias="LLM_MAX_TOKENS"
    )
    # Необязательные переопределения модели для отдельных агентов
    # (этап 8, тюнинг движка; None → модель активного движка).
    # Validator не указан: он rule-based и LLM не вызывает.
    SUMMARIZER_MODEL: Optional[str] = Field(
        default=None, validation_alias="SUMMARIZER_MODEL"
    )
    CRITIC_MODEL: Optional[str] = Field(
        default=None, validation_alias="CRITIC_MODEL"
    )
    EDITOR_MODEL: Optional[str] = Field(
        default=None, validation_alias="EDITOR_MODEL"
    )

    # ========================================================================
    # OpenRouter (облачный OpenAI-совместимый агрегатор моделей)
    # ========================================================================
    OPENROUTER_API_KEY: Optional[str] = Field(
        default=None, validation_alias="OPENROUTER_API_KEY"
    )
    OPENROUTER_BASE_URL: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    # Модель по умолчанию в облаке: та же Qwen2.5-7B, что и локально, —
    # поведение конвейера не зависит от места исполнения.
    OPENROUTER_MODEL: str = Field(
        default="qwen/qwen-2.5-7b-instruct",
        validation_alias="OPENROUTER_MODEL",
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
    # Локальный файловый режим Qdrant (без сервера): путь к папке хранения.
    # Пусто → обычный HTTP-клиент к QDRANT_HOST:QDRANT_PORT.
    QDRANT_LOCAL_PATH: Optional[str] = Field(
        default=None, validation_alias="QDRANT_LOCAL_PATH"
    )

    # ========================================================================
    # Эмбеддинги (гибридный RAG)
    # ========================================================================
    # Токен Hugging Face (необязателен): без него Hub пускает анонимом —
    # ниже лимиты, медленнее скачивание (BERTScore в eval, эмбеддинги).
    HF_TOKEN: Optional[str] = Field(default=None, validation_alias="HF_TOKEN")
    EMBEDDING_MODEL: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    EMBEDDING_DIM: int = Field(
        default=384, validation_alias="EMBEDDING_DIM"
    )

    # ========================================================================
    # Гибридный поиск (BM25 + Qdrant + RRF)
    # ========================================================================
    HYBRID_TOP_K: int = Field(default=5, validation_alias="HYBRID_TOP_K")
    RRF_K: int = Field(default=60, validation_alias="RRF_K")
    # Сколько символов исходного текста пары БЗ использовать для поиска
    # (эмбеддинг и BM25 индексируют урезанное представление, пара целиком
    # хранится в БД).
    KB_SEARCH_TEXT_CHARS: int = Field(
        default=2000, validation_alias="KB_SEARCH_TEXT_CHARS"
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
    def qdrant_url(self) -> str:
        """Полный URL Qdrant."""
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def llm_base_url(self) -> str:
        """Базовый URL активного LLM-движка (OpenAI-совместимый)."""
        if self.LLM_ENGINE == "openrouter":
            return self.OPENROUTER_BASE_URL.rstrip("/")
        return self.LMSTUDIO_URL.rstrip("/")

    @property
    def default_model(self) -> str:
        """Модель по умолчанию активного движка."""
        if self.LLM_ENGINE == "openrouter":
            return self.OPENROUTER_MODEL
        return self.LMSTUDIO_MODEL

    @property
    def lmstudio_chat_url(self) -> str:
        return f"{self.LMSTUDIO_URL.rstrip('/')}/chat/completions"

    @property
    def lmstudio_models_url(self) -> str:
        return f"{self.LMSTUDIO_URL.rstrip('/')}/models"

    # ========================================================================
    # Validators
    # ========================================================================

    @field_validator("LLM_ENGINE")
    @classmethod
    def validate_llm_engine(cls, v: str) -> str:
        allowed = {"lmstudio", "openrouter"}
        if v not in allowed:
            raise ValueError(
                f"LLM_ENGINE должен быть одним из {allowed}, получено: {v!r}"
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

