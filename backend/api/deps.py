# backend/api/deps.py
"""
FastAPI-зависимости: сессия БД, текущий пользователь, роли.

Использование в роутах:
    async def route(user: User = Depends(get_current_user)):
        ...

    async def admin_route(admin: User = Depends(require_admin)):
        ...
"""

import logging
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.db.models import User
from backend.db.session import get_async_session
from backend.utils.security import decode_token


logger = logging.getLogger(__name__)


# ============================================================================
# Сессия БД
# ============================================================================
async def get_db() -> AsyncSession:
    """
    Возвращает сессию БД для использования в роутах.

    Просто переиспользует get_async_session из db.session — так роуты
    могут импортировать любую из них.
    """
    async for session in get_async_session():
        yield session


# ============================================================================
# Bearer-токен
# ============================================================================
# auto_error=False — потому что нам нужно разное поведение:
# - get_current_user: 401, если токена нет
# - optional_user:    None, если токена нет
bearer_scheme = HTTPBearer(auto_error=False)


# ============================================================================
# Текущий пользователь
# ============================================================================
async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Достаёт текущего пользователя из JWT.

    Алгоритм:
    1. Если заголовка Authorization нет — 401.
    2. Декодируем JWT. Если невалиден/истёк — 401.
    3. Ищем пользователя в БД по sub (user_id).
    4. Если не найден — 401.
    5. Если is_active=False — 403.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Требуется авторизация",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = decode_token(credentials.credentials)
    except JWTError as e:
        logger.debug("Invalid JWT: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалидный или истёкший токен",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Токен не содержит идентификатор пользователя",
        )

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Пользователь не найден",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Пользователь заблокирован",
        )

    return user


async def optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: AsyncSession = Depends(get_db),
) -> Optional[User]:
    """
    Мягкая версия get_current_user.

    Если токена нет или он невалиден — возвращает None вместо 401.
    Полезно для эндпоинтов, которые работают и для гостя, и для юзера.
    """
    if credentials is None:
        return None
    try:
        return await get_current_user(credentials, db)
    except HTTPException:
        return None


# ============================================================================
# Роли
# ============================================================================
async def require_admin(
    user: User = Depends(get_current_user),
) -> User:
    """Пускает только admin. Иначе — 403."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права администратора",
        )
    return user


async def require_editor(
    user: User = Depends(get_current_user),
) -> User:
    """Пускает editor или admin. Иначе — 403."""
    if user.role not in ("editor", "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Требуются права редактора",
        )
    return user