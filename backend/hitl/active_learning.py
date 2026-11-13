# backend/hitl/active_learning.py
"""
Контур Human-in-the-Loop и активного обучения.

Резюме с низким качеством (Critic < порога или провал валидации)
попадает в очередь uncertain_examples. Редактор (человек) исправляет
его; подтверждённое резюме становится новой эталонной парой базы
знаний — система учится на правках (активное обучение).
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.data.knowledge_base import KBManager
from backend.db.models import Article, Summary, UncertainExample

logger = logging.getLogger(__name__)


class ActiveLearningManager:
    """Очередь HITL: добавление, просмотр, подтверждение, отклонение."""

    # ------------------------------------------------------------------------
    # Добавление в очередь
    # ------------------------------------------------------------------------
    @staticmethod
    async def add_uncertain_example(
        session: AsyncSession,
        *,
        summary_id: str,
        model_output: str,
        critic_score: Optional[float] = None,
        validation_passed: Optional[bool] = None,
    ) -> str:
        """
        Добавляет резюме в очередь модерации.

        Идемпотентно: если для этого резюме уже есть запись со статусом
        pending — возвращается id существующей записи (дублей нет).
        """
        existing = await session.execute(
            select(UncertainExample).where(
                UncertainExample.summary_id == summary_id,
                UncertainExample.status == "pending",
            )
        )
        duplicate = existing.scalars().first()
        if duplicate is not None:
            logger.debug(
                "HITL: pending-запись для summary %s уже есть", summary_id
            )
            return duplicate.id

        example = UncertainExample(
            summary_id=summary_id,
            model_output=model_output,
            critic_score=critic_score,
            validation_passed=validation_passed,
            status="pending",
        )
        session.add(example)
        await session.flush()

        logger.info(
            "HITL: резюме %s в очереди модерации (score=%s)",
            summary_id, critic_score,
        )
        return example.id

    # ------------------------------------------------------------------------
    # Просмотр очереди
    # ------------------------------------------------------------------------
    @staticmethod
    async def get_pending_examples(
        session: AsyncSession,
        limit: int = 20,
        offset: int = 0,
    ) -> List[UncertainExample]:
        stmt = (
            select(UncertainExample)
            .where(UncertainExample.status == "pending")
            .order_by(UncertainExample.created_at)
            .limit(limit)
            .offset(offset)
        )
        result = await session.execute(stmt)
        return list(result.scalars().all())

    @staticmethod
    async def get_example(
        session: AsyncSession, example_id: str
    ) -> Optional[UncertainExample]:
        result = await session.execute(
            select(UncertainExample).where(UncertainExample.id == example_id)
        )
        return result.scalars().first()

    @staticmethod
    async def count_pending(session: AsyncSession) -> int:
        result = await session.execute(
            select(func.count(UncertainExample.id)).where(
                UncertainExample.status == "pending"
            )
        )
        return result.scalar_one()

    @staticmethod
    async def get_stats(session: AsyncSession) -> Dict[str, int]:
        """{pending, confirmed, rejected} одним запросом."""
        result = await session.execute(
            select(UncertainExample.status, func.count(UncertainExample.id))
            .group_by(UncertainExample.status)
        )
        counts = {status: 0 for status in ("pending", "confirmed", "rejected")}
        for status, count in result.all():
            counts[status] = count
        return counts

    # ------------------------------------------------------------------------
    # Обработка редактором
    # ------------------------------------------------------------------------
    @staticmethod
    async def confirm_correction(
        session: AsyncSession,
        *,
        example_id: str,
        corrected_summary: str,
        reviewer_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Подтверждает правку редактора — атомарно в транзакции вызывшего:

        1. INSERT эталонной пары «текст статьи → исправленное резюме»
           в knowledge_base (ON CONFLICT DO NOTHING);
        2. UPDATE примера: status=confirmed, corrected_summary, reviewer,
           reviewed_at;
        3. UPDATE исходного Summary финальным текстом.

        Возвращает {example, kb_inserted}. kb_inserted=False, если пара
        уже была в БЗ — пример при этом всё равно подтверждается.
        """
        example = await ActiveLearningManager.get_example(session, example_id)
        if example is None:
            raise ValueError("Пример очереди не найден")

        if example.status != "pending":
            raise ValueError(
                f"Пример уже обработан (status={example.status})"
            )

        # Статья исходного резюме — источник текста для эталонной пары.
        summary = await session.get(Summary, example.summary_id)
        if summary is None:
            raise ValueError("Исходное резюме не найдено")

        article = await session.get(Article, summary.article_id)
        if article is None:
            raise ValueError("Статья резюме не найдена")

        kb_manager = KBManager()
        kb_id = await kb_manager.insert_in_session(
            session,
            field=article.field or "general",
            language=article.language or "ru",
            source_text=article.full_text or "",
            target_summary=corrected_summary,
            source="hitl",
            quality_score=example.critic_score,
        )
        kb_inserted = kb_id is not None

        # Финальная версия резюме — исправленная.
        summary.summary_text = corrected_summary

        example.status = "confirmed"
        example.corrected_summary = corrected_summary
        example.reviewer_id = reviewer_id
        example.reviewed_at = datetime.now(timezone.utc)

        await session.flush()

        logger.info(
            "HITL: пример %s подтверждён (kb_inserted=%s)", example_id, kb_inserted
        )
        return {"example": example, "kb_inserted": kb_inserted}

    @staticmethod
    async def reject_example(
        session: AsyncSession,
        example_id: str,
        reviewer_id: Optional[str] = None,
        reason: Optional[str] = None,
    ) -> UncertainExample:
        """
        Отклоняет пример: status=rejected, в базу знаний НЕ пишем.
        Исправленное резюме не становится эталоном.
        """
        example = await ActiveLearningManager.get_example(session, example_id)
        if example is None:
            raise ValueError("Пример очереди не найден")

        if example.status != "pending":
            raise ValueError(
                f"Пример уже обработан (status={example.status})"
            )

        example.status = "rejected"
        example.corrected_summary = None
        example.reviewer_id = reviewer_id
        example.reviewed_at = datetime.now(timezone.utc)
        if reason:
            # Причину отказа сохраняем в свободной форме рядом с model_output.
            example.model_output = (
                f"{example.model_output}\n[REJECT REASON] {reason}"
            )

        await session.flush()

        logger.info("HITL: пример %s отклонён", example_id)
        return example
