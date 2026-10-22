# backend/agents/supervisor.py
"""
Supervisor — оркестратор мультиагентного пайплайна.

Цикл:
1. Summarizer (один раз) — создаёт первичное резюме.
2. Validator → Critic.
3. Если качество OK — выход.
4. Иначе — Editor, и повтор с шага 2 (не более max_iterations раз).
"""

import logging
from typing import Any, Dict

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
            logger.warning("Supervisor: Summarizer вернул пустой результат")
            return state

        # === 2. Цикл Validator → Critic → Editor ===
        for iteration in range(1, self.max_iterations + 1):
            state["supervisor_iteration"] = iteration
            logger.info("=== Supervisor iteration %d/%d ===", iteration, self.max_iterations)

            state = await self.validator.run(state)
            state = await self.critic.run(state)

            if self._is_good_enough(state):
                state["supervisor_final_reason"] = "passed"
                state["supervisor_iterations"] = iteration
                logger.info("Supervisor: quality OK on iteration %d", iteration)
                return state

            # На последней итерации Editor не вызываем
            if iteration < self.max_iterations:
                state = await self.editor.run(state)
                if not state.get("editor_changed", False):
                    logger.info("Editor ничего не изменил → выходим")
                    state["supervisor_final_reason"] = "editor_no_change"
                    state["supervisor_iterations"] = iteration
                    return state

        state["supervisor_final_reason"] = "max_iterations"
        state["supervisor_iterations"] = self.max_iterations
        logger.info("Supervisor: max iterations reached")
        return state

    # ------------------------------------------------------------------------
    # Quality check
    # ------------------------------------------------------------------------
    def _is_good_enough(self, state: Dict[str, Any]) -> bool:
        """
        Критерий выхода:
        - validation_passed == True, И
        - critic_score >= threshold (или critic_score is None).
        """
        validation_ok = bool(state.get("validation_passed", False))
        if not validation_ok:
            return False
        score = state.get("critic_score")
        if score is None:
            return True
        return float(score) >= self.critic_threshold