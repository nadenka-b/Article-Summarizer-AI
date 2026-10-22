# backend/db/models.py
"""
SQLAlchemy-модели для Article-Summarizer-AI.

Таблицы:
- users                  — пользователи (user/editor/admin)
- articles               — загруженные научные статьи
- summaries              — сгенерированные резюме
- knowledge_base         — эталонные пары (текст → резюме) для RAG
- glossary               — научные термины и аббревиатуры
- uncertain_examples     — очередь HITL (низкое качество)
- cache                  — кэш результатов LLM
- reasoning_logs         — пошаговый лог агентов
- llm_calls              — трассировка вызовов LLM
"""

import uuid
from datetime import datetime

from sqlalchemy import (
    String,
    Text,
    Integer,
    Float,
    Boolean,
    DateTime,
    ForeignKey,
    UniqueConstraint,
    Index,
)
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from backend.db.base import Base


# ============================================================================
# Helpers
# ============================================================================
def _uuid_col():
    """
    Универсальная UUID-колонка.

    На PostgreSQL — нативный тип UUID.
    На SQLite (для тестов) — строка CHAR(36).
    """
    return mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )


# ============================================================================
# 1. users
# ============================================================================
class User(Base):
    """Пользователь системы."""

    __tablename__ = "users"

    id: Mapped[str] = _uuid_col()
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String(255), nullable=False)
    full_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(
        String(20), default="user", nullable=False
    )  # user | editor | admin
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), onupdate=func.now(), nullable=True
    )

    # Связи
    articles: Mapped[list["Article"]] = relationship(
        back_populates="owner", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<User {self.username} ({self.role})>"


# ============================================================================
# 2. articles
# ============================================================================
class Article(Base):
    """Загруженная научная статья."""

    __tablename__ = "articles"

    id: Mapped[str] = _uuid_col()
    owner_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    source: Mapped[str] = mapped_column(
        String(50), default="upload", nullable=False
    )  # upload | arxiv | pubmed | semantic_scholar
    source_id: Mapped[str | None] = mapped_column(
        String(100), nullable=True
    )  # arXiv ID, PubMed ID и т.п.
    field: Mapped[str | None] = mapped_column(
        String(20), nullable=True, index=True
    )  # cs, bio, physics, ...
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)

    # Текст
    full_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    sections: Mapped[str | None] = mapped_column(
        Text, nullable=True
    )  # JSON со секциями: {"abstract": "...", "methods": "..."}
    char_count: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Метаданные
    pdf_url: Mapped[str | None] = mapped_column(String(500), nullable=True)
    authors: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON список
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Служебное
    status: Mapped[str] = mapped_column(
        String(20), default="uploaded", nullable=False
    )  # uploaded | extracted | summarized | failed
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Связи
    owner: Mapped["User"] = relationship(back_populates="articles")
    summaries: Mapped[list["Summary"]] = relationship(
        back_populates="article", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Article {self.title[:50]}>"


# ============================================================================
# 3. summaries
# ============================================================================
class Summary(Base):
    """Сгенерированное резюме."""

    __tablename__ = "summaries"

    id: Mapped[str] = _uuid_col()
    article_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("articles.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Результат
    summary_text: Mapped[str] = mapped_column(Text, nullable=False)
    summary_mode: Mapped[str] = mapped_column(
        String(20), default="abstract", nullable=False
    )  # abstract | section_wise

    # Оценки
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    critic_reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Модель
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)
    engine_used: Mapped[str | None] = mapped_column(String(20), nullable=True)  # ollama  | api

    # Итерации агентов
    supervisor_iterations: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # Связи
    article: Mapped["Article"] = relationship(back_populates="summaries")

    def __repr__(self) -> str:
        return f"<Summary of {self.article_id} ({self.summary_mode})>"


# ============================================================================
# 4. knowledge_base
# ============================================================================
class KnowledgeBase(Base):
    """
    База знаний: эталонные пары «текст статьи → резюме».
    Используется RAG для few-shot примеров.
    """

    __tablename__ = "knowledge_base"
    __table_args__ = (
        UniqueConstraint("field", "source_text_hash", name="uq_kb_field_hash"),
        Index("ix_kb_field_language", "field", "language"),
    )

    id: Mapped[str] = _uuid_col()
    field: Mapped[str] = mapped_column(String(20), nullable=False, index=True)  # cs, bio, ...
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)

    # Эталонная пара
    source_text: Mapped[str] = mapped_column(Text, nullable=False)  # фрагмент статьи
    source_text_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # sha256 для дедупликации
    target_summary: Mapped[str] = mapped_column(Text, nullable=False)  # эталонное резюме

    # Метаданные
    source: Mapped[str | None] = mapped_column(String(50), nullable=True)  # откуда взято
    quality_score: Mapped[float | None] = mapped_column(Float, nullable=True)  # оценка качества

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<KB {self.field} ({self.language})>"


# ============================================================================
# 5. glossary
# ============================================================================
class GlossaryDB(Base):
    """
    Глоссарий: научные термины, аббревиатуры, шаблонные фразы.
    """

    __tablename__ = "glossary"
    __table_args__ = (
        UniqueConstraint("field", "term", name="uq_glossary_field_term"),
    )

    id: Mapped[str] = _uuid_col()
    field: Mapped[str] = mapped_column(String(20), nullable=False, index=True)
    term: Mapped[str] = mapped_column(String(200), nullable=False, index=True)
    definition: Mapped[str] = mapped_column(Text, nullable=False)  # определение/расшифровка
    category: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )  # abbreviation | concept | template
    language: Mapped[str] = mapped_column(String(10), default="en", nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Glossary {self.term}>"


# ============================================================================
# 6. uncertain_examples (HITL)
# ============================================================================
class UncertainExample(Base):
    """
    Очередь HITL: резюме с низким качеством, требующие проверки редактором.
    """

    __tablename__ = "uncertain_examples"

    id: Mapped[str] = _uuid_col()
    summary_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("summaries.id", ondelete="CASCADE"), nullable=False, index=True
    )

    # Что было сгенерировано
    model_output: Mapped[str] = mapped_column(Text, nullable=False)
    critic_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    validation_passed: Mapped[bool | None] = mapped_column(Boolean, nullable=True)

    # Что исправил редактор
    corrected_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )

    # Статус
    status: Mapped[str] = mapped_column(
        String(20), default="pending", nullable=False, index=True
    )  # pending | confirmed | rejected

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    reviewed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    def __repr__(self) -> str:
        return f"<UncertainExample {self.id} ({self.status})>"


# ============================================================================
# 7. cache
# ============================================================================
class SummaryCache(Base):
    """
    Кэш результатов суммаризации.

    Ключ: (source_text_hash, summary_mode, engine_used, model_used).
    """

    __tablename__ = "cache"
    __table_args__ = (
        UniqueConstraint(
            "source_text_hash", "summary_mode", "engine_used", "model_used",
            name="uq_cache_key",
        ),
    )

    id: Mapped[str] = _uuid_col()
    source_text_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    summary_mode: Mapped[str] = mapped_column(String(20), nullable=False)
    engine_used: Mapped[str | None] = mapped_column(String(20), nullable=True)
    model_used: Mapped[str | None] = mapped_column(String(100), nullable=True)

    # Результат
    output: Mapped[str] = mapped_column(Text, nullable=False)
    hits: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    def __repr__(self) -> str:
        return f"<Cache {self.source_text_hash[:8]} hits={self.hits}>"


# ============================================================================
# 8. reasoning_logs
# ============================================================================
class ReasoningLog(Base):
    """
    Лог шагов рассуждения агентов (Supervisor, Validator, Critic, Editor).

    Записывается по одной строке на каждый шаг агента.
    """

    __tablename__ = "reasoning_logs"

    id: Mapped[str] = _uuid_col()
    request_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True
    )  # один request_id = один запрос пользователя
    agent_name: Mapped[str] = mapped_column(String(50), nullable=False, index=True)

    # Что было на входе/выходе
    input_summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    output_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Модель (если агент вызывал LLM)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    parameters: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Тайминги
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    status: Mapped[str] = mapped_column(
        String(20), default="success", nullable=False
    )  # success | error

    def __repr__(self) -> str:
        return f"<ReasoningLog {self.agent_name} ({self.status})>"


# ============================================================================
# 9. llm_calls
# ============================================================================
class LLMCall(Base):
    """
    Трассировка всех вызовов LLM.

    Используется для мониторинга (сколько вызовов, какая latency, сколько ошибок).
    """

    __tablename__ = "llm_calls"

    id: Mapped[str] = _uuid_col()
    request_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    agent_name: Mapped[str | None] = mapped_column(String(50), nullable=True, index=True)

    # Модель
    model_name: Mapped[str] = mapped_column(String(100), nullable=False)
    provider: Mapped[str] = mapped_column(
        String(20), nullable=False
    )  # ollama  | openai | anthropic | ...
    parameters: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON

    # Хэши промпта/ответа (сами тексты не храним — слишком много)
    prompt_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    response_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)

    # Метрики
    tokens_used: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)

    status: Mapped[str] = mapped_column(
        String(20), default="success", nullable=False
    )  # success | error
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self) -> str:
        return f"<LLMCall {self.provider}/{self.model_name} ({self.status})>"