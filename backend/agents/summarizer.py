# backend/agents/summarizer.py
"""Summarizer — создаёт первичное резюме научной статьи."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.base import Agent

logger = logging.getLogger(__name__)


class SummarizerAgent(Agent):
    """Генерация резюме: abstract или section_wise."""

    name = "summarizer"
    SECTION_KEYS: List[str] = ["abstract", "introduction", "methods", "results", "conclusion"]
    MAX_INPUT_CHARS: int = 6000

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode = state.get("summary_mode") or "abstract"
        article_text = (state.get("article_text") or "").strip()
        sections: Optional[Dict[str, str]] = state.get("sections")

        if not article_text and not sections:
            state["summary"] = ""
            state["section_summaries"] = {}
            state["summarizer_error"] = "Нет входных данных"
            return state

        if mode == "section_wise":
            summary, section_summaries = await self._summarize_sections(
                sections or {}, article_text
            )
        else:
            summary = await self._summarize_abstract(article_text, sections)
            section_summaries = {}

        state["summary"] = summary
        state["section_summaries"] = section_summaries
        state["summary_mode"] = mode
        state.setdefault("summarizer_model", None)

        logger.info("Summarizer: mode=%s, words=%d", mode, len(summary.split()))
        return state

    async def _summarize_abstract(
        self,
        article_text: str,
        sections: Optional[Dict[str, str]],
    ) -> str:
        context = self._build_abstract_context(article_text, sections)
        prompt = self._build_abstract_prompt(context)
        try:
            raw = await self._call_llm(prompt, task="summarize")
        except Exception as e:
            logger.warning("Summarizer LLM failed: %s", e)
            return ""
        return self._clean_summary(raw)

    def _build_abstract_context(
        self,
        article_text: str,
        sections: Optional[Dict[str, str]],
    ) -> str:
        if sections:
            parts = []
            for key in self.SECTION_KEYS:
                if sections.get(key):
                    parts.append(f"[{key.upper()}]\n{sections[key]}")
            return "\n\n".join(parts)[: self.MAX_INPUT_CHARS]
        return article_text[: self.MAX_INPUT_CHARS]

    def _build_abstract_prompt(self, context: str) -> str:
        return f"""Ты — научный редактор. Напиши аннотацию (abstract) к статье.
Объём 150-250 слов. Отрази: проблему, метод, ключевые результаты, вывод.
Только текст аннотации.

ТЕКСТ СТАТЬИ:
---
{context}
---

АННОТАЦИЯ:"""

    async def _summarize_sections(
        self,
        sections: Dict[str, str],
        article_text: str,
    ) -> Tuple[str, Dict[str, str]]:
        section_summaries: Dict[str, str] = {}
        for key in self.SECTION_KEYS:
            text = (sections.get(key) or "").strip()
            if not text:
                continue
            prompt = self._build_section_prompt(key, text)
            try:
                raw = await self._call_llm(prompt)
                section_summaries[key] = self._clean_summary(raw)
            except Exception as e:
                logger.warning("Summarizer failed on %s: %s", key, e)
                section_summaries[key] = ""

        combined_parts = [
            f"**{k.capitalize()}**: {section_summaries[k]}"
            for k in self.SECTION_KEYS
            if section_summaries.get(k)
        ]
        return "\n\n".join(combined_parts), section_summaries

    def _build_section_prompt(self, section: str, text: str) -> str:
        instructions = {
            "abstract": "Кратко переформулируй суть в 50-80 слов.",
            "introduction": "Опиши проблему, важность, новизну (40-60 слов).",
            "methods": "Опиши методы, данные, архитектуру (60-100 слов).",
            "results": "Перечисли ключевые результаты с числами (60-100 слов).",
            "conclusion": "Что доказали, ограничения, future work (40-60 слов).",
        }
        instruction = instructions.get(section, "Кратко изложи содержание (40-80 слов).")
        return f"""Сделай резюме секции "{section.upper()}" научной статьи.
{instruction}
Только текст резюме.

ТЕКСТ:
---
{text[: self.MAX_INPUT_CHARS]}
---

РЕЗЮМЕ:"""

    
    @staticmethod
    def _clean_summary(raw: str) -> str:
        if not raw:
            return ""
        text = raw.strip()
        md = re.match(r"^```(?:text|markdown)?\s*(.*?)\s*```$", text, flags=re.DOTALL)
        if md:
            text = md.group(1).strip()
        prefixes = [
            "вот резюме:", "вот аннотация:", "summary:", "abstract:",
            "аннотация:", "резюме:", "the summary is:",
        ]
        lower = text.lower()
        for p in prefixes:
            if lower.startswith(p):
                text = text[len(p):].lstrip(" :\n\t")
                break
        return text.strip('"\' \n\t')