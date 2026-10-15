# backend/utils/security.py
"""
Утилиты безопасности: хеширование паролей и работа с JWT.

- Пароли хранятся в виде bcrypt-хеша.
- JWT подписывается HS256 с секретным ключом из .env.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from backend.core.config import get_settings
from backend.core.exceptions import ConfigurationError


logger = logging.getLogger(__name__)
settings = get_settings()


# ============================================================================
# Хеширование паролей
# ============================================================================
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def get_password_hash(password: str) -> str:
    """
    Хеширует пароль через bcrypt.

    bcrypt имеет ограничение в 72 байта на входе. Если пароль длиннее —
    обрезаем, чтобы не получить ошибку от библиотеки.
    """
    if len(password.encode("utf-8")) > 72:
        password = password[:72]
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Проверяет, соответствует ли пароль хешу.

    Возвращает True, если пароль верный. False — если нет.
    Если хеш повреждён — тоже вернёт False (внутри passlib ловит исключение).
    """
    if len(plain_password.encode("utf-8")) > 72:
        plain_password = plain_password[:72]
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        logger.warning("Password verification failed: %s", e)
        return False


# ============================================================================
# JWT
# ============================================================================
def _check_secret_key() -> None:
    """
    Проверяет, что JWT_SECRET_KEY задан и не дефолтный.

    Если оставить 'change-me-in-production', любой сможет подделать токен.
    """
    if not settings.JWT_SECRET_KEY or settings.JWT_SECRET_KEY == "change-me-in-production":
        if not settings.DEBUG:
            raise ConfigurationError(
                "JWT_SECRET_KEY не задан или используется дефолтное значение. "
                "Сгенерируйте ключ: python -c \"import secrets; print(secrets.token_urlsafe(64))\""
            )
        logger.warning(
            "JWT_SECRET_KEY — дефолтный. Это допустимо только в DEBUG-режиме."
        )


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """
    Создаёт JWT-токен.

    data — payload, обычно {"sub": user_id, "role": "user"}.
    expires_delta — кастомный срок жизни. Если None — берём из настроек.
    """
    _check_secret_key()

    to_encode = data.copy()
    now = datetime.now(timezone.utc)

    if expires_delta is not None:
        expire = now + expires_delta
    else:
        expire = now + timedelta(minutes=settings.JWT_EXPIRATION_MINUTES)

    to_encode.update({"exp": expire, "iat": now})

    return jwt.encode(
        to_encode,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )


def decode_token(token: str) -> Dict[str, Any]:
    """
    Декодирует и валидирует JWT.

    Возвращает payload. Если токен невалиден или истёк — бросает JWTError.
    """
    _check_secret_key()

    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM],
        )
        return payload
    except JWTError as e:
        logger.debug("JWT decode failed: %s", e)
        raise