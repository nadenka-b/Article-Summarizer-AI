# backend/agents/supervisor.py
"""
Supervisor — оркестратор мультиагентного пайплайна.

Цикл:
1. Summarizer (один раз) — создаёт первичное резюме.
2. Validator → Critic.
3. Если качество OK — выход.
4. Иначе — Editor, и повтор с шага 2 (не более max_iterations раз).

Если после всех итераций лучшей оказалась не последняя версия —
Supervisor откатывает state к лучшей (снапшот по critic_score).
"""

import logging
from typing import Any, Dict, Optional

from backend.agents.base import Agent
from backend.agents.critic import CriticAgent
from backend.agents.editor import EditorAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.validator import ValidatorAgent

logger = logging.getLogger(__name__)


class Supervisor(Agent):
    """Оркестратор цикла агентов."""

    name = "supervisor"

    def __init__(
        self,
        max_iterations: int = 3,
        critic_threshold: float = 7.0,
    ) -> None:
        super().__init__()
        self.max_iterations = max_iterations
        self.critic_threshold = critic_threshold

        # Ленивая инициализация
        self._summarizer: SummarizerAgent | None = None
        self._validator: ValidatorAgent | None = None
        self._critic: CriticAgent | None = None
        self._editor: EditorAgent | None = None

    # ------------------------------------------------------------------------
    # Lazy properties
    # ------------------------------------------------------------------------
    @property
    def summarizer(self) -> SummarizerAgent:
        if self._summarizer is None:
            self._summarizer = SummarizerAgent()
        return self._summarizer

    @property
    def validator(self) -> ValidatorAgent:
        if self._validator is None:
            self._validator = ValidatorAgent()
        return self._validator

    @property
    def critic(self) -> CriticAgent:
        if self._critic is None:
            self._critic = CriticAgent()
        return self._critic

    @property
    def editor(self) -> EditorAgent:
        if self._editor is None:
            self._editor = EditorAgent()
        return self._editor

    # ------------------------------------------------------------------------
    # Main pipeline
    # ------------------------------------------------------------------------
    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Основной цикл."""
        # === 1. Summarizer (один раз) ===
        state = await self.summarizer.run(state)

        if not state.get("summary"):
            state["supervisor_final_reason"] = "no_summary"
            state["supervisor_iterations"] = 0
            state["supervisor_restored_best"] = False
            logger.warning("Supervisor: Summarizer вернул пустой результат")
            return state

        # === 2. Цикл Validator → Critic → Editor ===
        best: Optional[Dict[str, Any]] = None
        best_score = float("-inf")

        for iteration in range(1, self.max_iterations + 1):
            state["supervisor_iteration"] = iteration
            logger.info("=== Supervisor iteration %d/%d ===", iteration, self.max_iterations)

            state = await self.validator.run(state)
            state = await self.critic.run(state)

            score = self._quality(state)
            if score is not None and score > best_score:
                best_score = score
                best = self._snapshot(state)

            if self._is_good_enough(state):
                logger.info("Supervisor: quality OK on iteration %d", iteration)
                return self._finish(state, "passed", iteration, best, best_score)

            # На последней итерации Editor не вызываем
            if iteration < self.max_iterations:
                state = await self.editor.run(state)
                if not state.get("editor_changed", False):
                    logger.info("Editor ничего не изменил → выходим")
                    return self._finish(
                        state, "editor_no_change", iteration, best, best_score
                    )

        logger.info("Supervisor: max iterations reached")
        return self._finish(
            state, "max_iterations", self.max_iterations, best, best_score
        )

    # ------------------------------------------------------------------------
    # Quality / best iteration
    # ------------------------------------------------------------------------
    @staticmethod
    def _quality(state: Dict[str, Any]) -> Optional[float]:
        """
        Численная оценка текущей версии резюме или None.

        None — когда сравнивать нельзя:
        - валидация не пройдена;
        - критик не смог оценить (critic_failed / score is None).
        """
        if not state.get("validation_passed", False):
            return None
        if state.get("critic_failed", False):
            return None
        score = state.get("critic_score")
        if score is None:
            return None
        return float(score)

    @staticmethod
    def _snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
        """Копия результата итерации для возможного отката."""
        return {
            "summary": state.get("summary", ""),
            "section_summaries": dict(state.get("section_summaries") or {}),
            "critic_score": state.get("critic_score"),
            "critic_reasoning": state.get("critic_reasoning"),
            "validation_passed": state.get("validation_passed"),
        }

    @staticmethod
    def _restore(state: Dict[str, Any], snapshot: Dict[str, Any]) -> None:
        """Откат результата к лучшей итерации.

        Важно: reasoning_log не откатываем — это история работы агентов,
        а не результат.
        """
        state.update(snapshot)

    def _finish(
        self,
        state: Dict[str, Any],
        reason: str,
        iteration: int,
        best: Optional[Dict[str, Any]],
        best_score: float,
    ) -> Dict[str, Any]:
        """Завершение: фиксирует причину выхода и при необходимости
        откатывает state к лучшей итерации (если финал хуже лучшей)."""
        state["supervisor_final_reason"] = reason
        state["supervisor_iterations"] = iteration

        current = self._quality(state)
        if best is not None and (current is None or best_score > current):
            self._restore(state, best)
            state["supervisor_restored_best"] = True
            logger.info(
                "Supervisor: restored best iteration (score=%s > %s)",
                best_score, current,
            )
        else:
            state["supervisor_restored_best"] = False
        return state

    # ------------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------------
    def _is_good_enough(self, state: Dict[str, Any]) -> bool:
        """
        Критерий выхода:
        - validation_passed == True, И
        - критик оценил (не critic_failed, score не None), И
        - critic_score >= threshold.
        """
        validation_ok = bool(state.get("validation_passed", False))
        if not validation_ok:
            return False
        if state.get("critic_failed", False):
            return False
        score = state.get("critic_score")
        if score is None:
            return False
        return float(score) >= self.critic_threshold