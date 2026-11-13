# backend/core/models.py
"""
Pydantic-модели для API: запросы и ответы.

Отдельный namespace от SQLAlchemy-моделей (backend.db.models).
SQLAlchemy — про хранение, Pydantic — про HTTP-контракт.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


# ============================================================================
# Auth — Requests
# ============================================================================
class RegisterRequest(BaseModel):
    """Тело запроса на регистрацию."""

    email: EmailStr = Field(..., description="Email пользователя")
    username: str = Field(
        ..., min_length=3, max_length=100,
        description="Уникальный username",
    )
    password: str = Field(
        ..., min_length=8, max_length=128,
        description="Пароль (минимум 8 символов, цифра и заглавная буква)",
    )
    full_name: Optional[str] = Field(
        default=None, max_length=255,
        description="Полное имя (опционально)",
    )

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """
        Простая проверка стойкости пароля.

        Требования: хотя бы одна цифра и хотя бы одна заглавная буква.
        """
        if not any(c.isdigit() for c in v):
            raise ValueError("Пароль должен содержать хотя бы одну цифру")
        if not any(c.isupper() for c in v):
            raise ValueError("Пароль должен содержать хотя бы одну заглавную букву")
        return v

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Username — только буквы, цифры, _ и -."""
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "Username может содержать только буквы, цифры, '_' и '-'"
            )
        return v


class LoginRequest(BaseModel):
    """Тело запроса на логин. Логинимся по email."""

    email: EmailStr
    password: str


# ============================================================================
# Auth — Responses
# ============================================================================
class UserResponse(BaseModel):
    """
    Публичное представление пользователя.

    ВАЖНО: hashed_password здесь НЕТ. Никогда не показываем хеш наружу.
    """

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    username: str
    full_name: Optional[str] = None
    role: str
    is_active: bool
    created_at: datetime


class TokenResponse(BaseModel):
    """Ответ на успешный логин или регистрацию."""

    access_token: str
    token_type: str = "bearer"
    expires_in: int = Field(..., description="Срок жизни токена в секундах")


# ============================================================================
# Summarization — Requests (этап 9)
# ============================================================================
_ALLOWED_MODES = {"baseline", "extractor", "rag", "agents", "ensemble"}
_ALLOWED_SUMMARY_MODES = {"abstract", "section_wise"}


def _validate_mode(v: Optional[str]) -> Optional[str]:
    if v is not None and v not in _ALLOWED_MODES:
        raise ValueError(f"mode должен быть одним из {sorted(_ALLOWED_MODES)}")
    return v


def _validate_summary_mode(v: Optional[str]) -> Optional[str]:
    if v is not None and v not in _ALLOWED_SUMMARY_MODES:
        raise ValueError(
            f"summary_mode должен быть одним из {sorted(_ALLOWED_SUMMARY_MODES)}"
        )
    return v


class SummarizeBaseRequest(BaseModel):
    """Общая часть запросов на суммаризацию."""

    title: Optional[str] = Field(default=None, max_length=500)
    field: Optional[str] = Field(default=None, max_length=20)
    mode: Optional[str] = None
    summary_mode: Optional[str] = None
    use_cache: bool = True

    _mode_check = field_validator("mode")(_validate_mode)
    _summary_mode_check = field_validator("summary_mode")(_validate_summary_mode)


class SummarizeTextRequest(SummarizeBaseRequest):
    """Суммаризация сырого текста статьи."""

    text: str = Field(..., min_length=200, description="Текст статьи")


class SummarizeArxivRequest(SummarizeBaseRequest):
    """Суммаризация статьи с arXiv по идентификатору."""

    arxiv_id: str = Field(
        ...,
        pattern=r"^\d{4}\.\d{4,5}(v\d+)?$",
        description="arXiv ID, например 2401.12345",
    )


# ============================================================================
# Summarization — Responses
# ============================================================================
class SummaryResponse(BaseModel):
    """Результат суммаризации."""

    article_id: Optional[str] = None
    summary_id: Optional[str] = None
    summary: str = ""
    mode: str
    summary_mode: str
    strategy: str = "none"
    critic_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    supervisor_iterations: int = 0
    final_reason: str = ""
    cached: bool = False
    hitl_created: bool = False
    reasoning_log: List[dict] = []
    error: Optional[str] = None


class SummaryOut(BaseModel):
    """Резюме из БД (GET /summaries/{id})."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    article_id: str
    summary_text: str
    summary_mode: str
    critic_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    supervisor_iterations: int
    created_at: datetime


# ============================================================================
# HITL — очередь модерации (этап 10)
# ============================================================================
class HITLQueueItem(BaseModel):
    """Пункт очереди модерации."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    summary_id: str
    model_output: str
    critic_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    status: str
    created_at: datetime


class HITLDetail(HITLQueueItem):
    """Пункт очереди с текстом статьи (для интерфейса редактора)."""

    article_title: Optional[str] = None
    article_text: Optional[str] = None
    corrected_summary: Optional[str] = None
    reviewed_at: Optional[datetime] = None


class HITLConfirmRequest(BaseModel):
    """Правка редактора: исправленное резюме."""

    corrected_summary: str = Field(
        ..., min_length=50, description="Исправленный текст резюме"
    )


class HITLRejectRequest(BaseModel):
    """Отклонение с причиной."""

    reason: Optional[str] = Field(default=None, max_length=2000)


class HITLReviewResponse(BaseModel):
    """Результат обработки пункта очереди."""

    id: str
    status: str
    kb_inserted: Optional[bool] = None
    message: str


class HITLStats(BaseModel):
    """Статистика очереди."""

    pending: int
    confirmed: int
    rejected: int


# ============================================================================
# Admin — управление пользователями (этап 10)
# ============================================================================
class UserRoleRequest(BaseModel):
    """Смена роли пользователя (только admin)."""

    role: str = Field(..., description="Новая роль: user | editor | admin")


class UserAdminOut(BaseModel):
    """Пользователь для админ-списка."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    email: EmailStr
    username: str
    role: str
    is_active: bool
    created_at: datetime