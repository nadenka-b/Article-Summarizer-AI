# backend/agents/editor.py
"""Editor — переписывает резюме по замечаниям."""

import logging
import re
from typing import Any, Dict, List

from backend.agents.base import Agent

logger = logging.getLogger(__name__)


class EditorAgent(Agent):
    """Улучшает резюме с учётом замечаний Validator и Critic."""

    name = "editor"
    MAX_SUMMARY_CHARS: int = 4000

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Логика:
        1. Собираем issues = errors + warnings + suggestions + missing_terms.
        2. Если issues пуст → ничего не делаем (editor_changed=False).
        3. Иначе — вызываем LLM для переписывания.
        """
        summary = (state.get("summary") or "").strip()
        if not summary:
            state["editor_changed"] = False
            state["editor_reasoning"] = "Резюме пустое — нечего редактировать"
            return state

        issues = self._collect_issues(state)

        if not issues:
            logger.info("Editor: no issues → skipping LLM")
            state["editor_changed"] = False
            state["editor_reasoning"] = "Нет замечаний"
            return state

        prompt = self._build_prompt(summary, issues)
        try:
            raw = await self._call_llm(prompt)
        except Exception as e:
            logger.warning("Editor LLM failed: %s", e)
            state["editor_changed"] = False
            state["editor_reasoning"] = f"LLM error: {e}"
            return state

        new_summary = self._clean(raw)
        if not new_summary or new_summary == summary:
            state["editor_changed"] = False
            state["editor_reasoning"] = "LLM не изменила текст"
            return state

        state["summary"] = new_summary
        state["editor_changed"] = True
        state["editor_reasoning"] = f"Переписано с учётом {len(issues)} замечаний"
        logger.info("Editor: changed, issues=%d", len(issues))
        return state

    def _collect_issues(self, state: Dict[str, Any]) -> List[str]:
        """Собирает все замечания в один список."""
        issues: List[str] = []
        issues.extend(state.get("validator_errors") or [])
        issues.extend(state.get("validator_warnings") or [])
        issues.extend(state.get("critic_suggestions") or [])
        issues.extend(
            f"Не хватает термина: {t}"
            for t in (state.get("critic_missing_terms") or [])
        )
        return issues

    def _build_prompt(self, summary: str, issues: List[str]) -> str:
        issues_str = "\n".join(f"- {i}" for i in issues)
        return f"""Ты — научный редактор. Перепиши резюме, устранив замечания.

ЗАМЕЧАНИЯ:
{issues_str}

ИСХОДНОЕ РЕЗЮМЕ:
---
{summary[: self.MAX_SUMMARY_CHARS]}
---

ТРЕБОВАНИЯ:
- Устрани все замечания.
- Сохрани смысл, не добавляй галлюцинаций.
- Стиль — научный.
- Только текст нового резюме, без пояснений.

НОВОЕ РЕЗЮМЕ:"""

    

    @staticmethod
    def _clean(raw: str) -> str:
        if not raw:
            return ""
        text = raw.strip()
        md = re.match(r"^```(?:text|markdown)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
        if md:
            text = md.group(1).strip()
        prefixes = ["новое резюме:", "резюме:", "edited summary:", "new summary:"]
        lower = text.lower()
        for p in prefixes:
            if lower.startswith(p):
                text = text[len(p):].lstrip(" :\n\t")
                break
        return text.strip('"\' \n\t')