# backend/api/routes/admin.py
"""
Админ-роуты (этап 10).

Минимальное управление пользователями: назначение ролей.
Без этого назначить редактора для HITL-очереди можно только из БД.
"""

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db, require_admin
from backend.core.models import UserAdminOut, UserRoleRequest
from backend.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/admin", tags=["admin"])

ALLOWED_ROLES = {"user", "editor", "admin"}


@router.get(
    "/users",
    response_model=list[UserAdminOut],
    summary="Список пользователей (admin)",
)
async def list_users(
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> list[UserAdminOut]:
    result = await db.execute(select(User).order_by(User.created_at))
    return [UserAdminOut.model_validate(u) for u in result.scalars().all()]


@router.patch(
    "/users/{user_id}/role",
    response_model=UserAdminOut,
    summary="Сменить роль пользователя (admin)",
)
async def change_role(
    user_id: str,
    payload: UserRoleRequest,
    admin: User = Depends(require_admin),
    db: AsyncSession = Depends(get_db),
) -> UserAdminOut:
    if payload.role not in ALLOWED_ROLES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"role должен быть одним из {sorted(ALLOWED_ROLES)}",
        )

    user = await db.get(User, user_id)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Пользователь не найден",
        )

    if user.id == admin.id and payload.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Администратор не может снять роль admin с себя",
        )

    old_role = user.role
    user.role = payload.role

    logger.info(
        "Admin %s: роль пользователя %s изменена %s → %s",
        admin.username, user.username, old_role, payload.role,
    )
    return UserAdminOut.model_validate(user)
