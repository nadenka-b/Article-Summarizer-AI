# backend/agents/summarizer.py
"""Summarizer — создаёт первичное резюме научной статьи."""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from backend.agents.base import Agent
from backend.data.chunker import chunk_text, needs_chunking

logger = logging.getLogger(__name__)


class SummarizerAgent(Agent):
    """Генерация резюме: abstract или section_wise.

    Стратегии abstract-режима (этап 8):
    - sections   — article задан секциями, суммаризируем их;
    - direct     — короткий текст помещается в промпт целиком;
    - hierarchical — длинный текст режется на чанки, резюме каждого
      чанка агрегируются в финальную аннотацию (без потерь контента).
    """

    name = "summarizer"
    SECTION_KEYS: List[str] = ["abstract", "introduction", "methods", "results", "conclusion"]
    MAX_INPUT_CHARS: int = 6000

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        mode = state.get("summary_mode") or "abstract"
        article_text = (state.get("article_text") or "").strip()
        sections: Optional[Dict[str, str]] = state.get("sections")
        few_shot = state.get("few_shot_examples") or ""

        if not article_text and not sections:
            state["summary"] = ""
            state["section_summaries"] = {}
            state["summarizer_error"] = "Нет входных данных"
            state["summarizer_strategy"] = "none"
            return state

        if mode == "section_wise":
            summary, section_summaries = await self._summarize_sections(
                sections or {}, article_text
            )
            strategy = "section_wise"
        else:
            section_summaries = {}
            if sections:
                context = self._build_abstract_context(article_text, sections)
                summary = await self._summarize_direct(context, few_shot)
                strategy = "sections"
            elif not needs_chunking(article_text):
                summary = await self._summarize_direct(article_text, few_shot)
                strategy = "direct"
            else:
                summary = await self._summarize_hierarchical(article_text, few_shot)
                strategy = "hierarchical"

        chunks_count = len(chunk_text(article_text)) if article_text else 0

        state["summary"] = summary
        state["section_summaries"] = section_summaries
        state["summary_mode"] = mode
        state["summarizer_strategy"] = strategy
        state["chunks_count"] = chunks_count if strategy == "hierarchical" else 0
        state.setdefault("summarizer_model", None)

        logger.info(
            "Summarizer: mode=%s, strategy=%s, words=%d",
            mode, strategy, len(summary.split()),
        )
        return state

    # ------------------------------------------------------------------------
    # Abstract-режим: стратегии
    # ------------------------------------------------------------------------
    async def _summarize_direct(self, context: str, few_shot: str = "") -> str:
        prompt = self._build_abstract_prompt(context, few_shot)
        try:
            raw = await self._call_llm(prompt, task="summarize")
        except Exception as e:
            logger.warning("Summarizer LLM failed: %s", e)
            return ""
        return self._clean_summary(raw)

    async def _summarize_hierarchical(
        self, article_text: str, few_shot: str = ""
    ) -> str:
        """
        Длинная статья: чанк → резюме чанка → агрегация.

        Сбой резюме отдельного чанка не рушит всё — chunk пропускается.
        """
        chunks = chunk_text(article_text)
        logger.info("Hierarchical summarization: %d chunks", len(chunks))

        partial_summaries: List[str] = []
        for chunk in chunks:
            prompt = self._build_chunk_prompt(chunk["text"], len(chunks))
            try:
                raw = await self._call_llm(prompt, task="summarize")
                cleaned = self._clean_summary(raw)
                if cleaned:
                    partial_summaries.append(cleaned)
            except Exception as e:
                logger.warning(
                    "Summarizer failed on chunk %d/%d: %s",
                    chunk["index"] + 1, len(chunks), e,
                )

        if not partial_summaries:
            logger.warning("Hierarchical summarization: all chunks failed")
            return ""

        combined = "\n\n".join(
            f"[Часть {i + 1}] {partial}"
            for i, partial in enumerate(partial_summaries)
        )
        final_prompt = self._build_aggregation_prompt(combined, few_shot)
        try:
            raw = await self._call_llm(final_prompt, task="summarize")
            return self._clean_summary(raw)
        except Exception as e:
            logger.warning("Aggregation failed: %s", e)
            # Фолбэк: склейка частичных резюме лучше, чем пустота.
            return "\n\n".join(partial_summaries)

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

    # ------------------------------------------------------------------------
    # Промпты
    # ------------------------------------------------------------------------
    @staticmethod
    def _few_shot_block(few_shot: str) -> str:
        if not few_shot:
            return ""
        return (
            "ПРИМЕРЫ ХОРОШИХ АННОТАЦИЙ К ПОХОЖИМ СТАТЬЯМ "
            "(используй как ориентир стиля и структуры):\n"
            f"{few_shot}\n\n"
        )

    def _build_abstract_prompt(self, context: str, few_shot: str = "") -> str:
        return f"""Ты — научный редактор. Напиши аннотацию (abstract) к статье.
Объём 150-250 слов. Отрази: проблему, метод, ключевые результаты, вывод.
Только текст аннотации.

{self._few_shot_block(few_shot)}ТЕКСТ СТАТЬИ:
---
{context}
---

АННОТАЦИЯ:"""

    def _build_chunk_prompt(self, chunk_text_value: str, total: int) -> str:
        return f"""Сделай краткое резюме фрагмента {total}-частной научной статьи.
Объём 60-100 слов: метод, данные, ключевые результаты фрагмента.
Только текст резюме.

ТЕКСТ ФРАГМЕНТА:
---
{chunk_text_value}
---

РЕЗЮМЕ ФРАГМЕНТА:"""

    def _build_aggregation_prompt(self, combined: str, few_shot: str = "") -> str:
        return f"""Ты — научный редактор. Перед тобой частичные резюме фрагментов
одной статьи. Объедини их в цельную аннотацию (abstract): 150-250 слов,
проблема → метод → ключевые результаты → вывод. Без повторов и без
нумерации частей. Только текст аннотации.

{self._few_shot_block(few_shot)}ЧАСТИЧНЫЕ РЕЗЮМЕ:
---
{combined}
---

ИТОГОВАЯ АННОТАЦИЯ:"""

    # ------------------------------------------------------------------------
    # Section-wise режим
    # ------------------------------------------------------------------------
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
            "аннотация:", "резюме:", "the summary is:", "итоговая аннотация:",
        ]
        lower = text.lower()
        for p in prefixes:
            if lower.startswith(p):
                text = text[len(p):].lstrip(" :\n\t")
                break
        return text.strip('"\' \n\t')
