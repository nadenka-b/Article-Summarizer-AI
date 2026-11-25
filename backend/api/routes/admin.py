# backend/api/routes/admin.py
"""
Админ-роуты (этап 10+).

Управление пользователями (назначение ролей) и просмотр метрик качества:
GET /admin/metrics отдаёт отчёты scripts/eval/run_eval.py из data/eval/.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.api.deps import get_db, require_admin
from backend.core.config import get_settings
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


@router.get(
    "/metrics",
    summary="Отчёты метрик качества по режимам (admin)",
)
async def eval_metrics(admin: User = Depends(require_admin)) -> dict:
    """Отдаёт JSON-отчёты scripts/eval/run_eval.py из EVAL_REPORTS_DIR.

    Один файл = один прогон eval_<mode>_<n>.json. Если режим прогоняли
    несколько раз, клиенту нужны все версии (свежая — по mtime).
    """
    reports_dir = Path(get_settings().EVAL_REPORTS_DIR)
    if not reports_dir.is_dir():
        return {
            "reports": [],
            "hint": (
                "Каталог отчётов не найден. Запустите "
                "python scripts/eval/run_eval.py --mode <baseline|rag|agents> "
                "--limit <N> и обновите страницу."
            ),
        }

    reports = []
    for path in sorted(reports_dir.glob("eval_*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            logger.warning("Пропущен битый отчёт %s: %s", path.name, e)
            continue
        metrics = data.get("metrics", {})
        reports.append({
            "file": path.name,
            "mode": metrics.get("mode") or path.stem.split("_")[1],
            "mtime": datetime.fromtimestamp(
                path.stat().st_mtime
            ).isoformat(timespec="seconds"),
            "metrics": metrics,
            "per_article": data.get("per_article", []),
        })

    return {
        "reports": reports,
        "hint": "" if reports else (
            "Отчётов пока нет. Запустите "
            "python scripts/eval/run_eval.py --mode <baseline|rag|agents> "
            "--limit <N> и обновите страницу."
        ),
    }
