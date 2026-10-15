# backend/core/models.py
"""
Pydantic-модели для API: запросы и ответы.

Отдельный namespace от SQLAlchemy-моделей (backend.db.models).
SQLAlchemy — про хранение, Pydantic — про HTTP-контракт.
"""

from datetime import datetime
from typing import Optional

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