# backend/api/routes/auth.py
"""
Эндпоинты аутентификации.

- POST /auth/register — регистрация нового пользователя.
- POST /auth/login    — вход по email + password.
- GET  /auth/me       — данные текущего пользователя.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_current_user, get_db
from backend.core.config import get_settings
from backend.core.models import (
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserResponse,
)
from backend.db.models import User
from backend.utils.security import (
    create_access_token,
    get_password_hash,
    verify_password,
)


logger = logging.getLogger(__name__)
settings = get_settings()
router = APIRouter(prefix="/auth", tags=["auth"])


# ============================================================================
# POST /auth/register
# ============================================================================
@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Регистрация нового пользователя",
)
async def register(
    payload: RegisterRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Регистрирует нового пользователя.

    1. Проверяет уникальность email и username.
    2. Хеширует пароль через bcrypt.
    3. Создаёт пользователя с ролью 'user'.
    4. Возвращает JWT-токен, чтобы клиент сразу был авторизован.
    """
    # Проверка: email или username заняты?
    stmt = select(User).where(
        or_(User.email == payload.email, User.username == payload.username)
    )
    result = await db.execute(stmt)
    existing = result.scalar_one_or_none()

    if existing is not None:
        if existing.email == payload.email:
            detail = "Пользователь с таким email уже существует"
        else:
            detail = "Этот username уже занят"
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=detail,
        )

    user = User(
        email=payload.email,
        username=payload.username,
        hashed_password=get_password_hash(payload.password),
        full_name=payload.full_name,
        role="user",
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    logger.info("New user registered: %s (id=%s)", user.username, user.id)

    token = create_access_token({"sub": user.id, "role": user.role})
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRATION_MINUTES * 60,
    )


# ============================================================================
# POST /auth/login
# ============================================================================
@router.post(
    "/login",
    response_model=TokenResponse,
    summary="Вход по email и паролю",
)
async def login(
    payload: LoginRequest,
    db: AsyncSession = Depends(get_db),
) -> TokenResponse:
    """
    Проверяет email + password и возвращает JWT.

    Соображения безопасности:
    - Не говорим, что именно неверно (email или пароль) — единое сообщение.
    - Не логируем введённый пароль.
    - 403, если пользователь деактивирован.
    """
    stmt = select(User).where(User.email == payload.email)
    result = await db.execute(stmt)
    user = result.scalar_one_or_none()

    if user is None or not verify_password(payload.password, user.hashed_password):
        # Единое сообщение — не даём перебором выяснить, какие email есть в системе.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверный email или пароль",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Пользователь заблокирован",
        )

    logger.info("User logged in: %s", user.username)

    token = create_access_token({"sub": user.id, "role": user.role})
    return TokenResponse(
        access_token=token,
        expires_in=settings.JWT_EXPIRATION_MINUTES * 60,
    )


# ============================================================================
# GET /auth/me
# ============================================================================
@router.get(
    "/me",
    response_model=UserResponse,
    summary="Данные текущего пользователя",
)
async def me(user: User = Depends(get_current_user)) -> UserResponse:
    """Возвращает профиль текущего пользователя. Требует Bearer-токен."""
    return UserResponse.model_validate(user)