# backend/agents/critic.py
"""Critic — LLM-as-a-Judge для оценки качества резюме."""

import json
import logging
import re
from typing import Any, Dict, List, Optional

from backend.agents.base import Agent

logger = logging.getLogger(__name__)


class CriticAgent(Agent):
    """Оценка по 4 критериям через LLM (заглушка на этапе 5)."""

    name = "critic"
    CRITIC_PASS_THRESHOLD: float = 7.0

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        summary = (state.get("summary") or "").strip()
        article_text = (state.get("article_text") or "").strip()
        key_terms: List[str] = state.get("key_terms") or []

        if not summary:
            return self._fallback(state, "Резюме пустое")

        prompt = self._build_prompt(summary, article_text, key_terms)
        try:
            raw = await self._call_llm(prompt)
        except Exception as e:
            return self._fallback(state, f"LLM error: {e}")

        parsed = self._parse_json(raw)
        if parsed is None:
            return self._fallback(state, "Не удалось разобрать JSON")

        return self._write(state, parsed)

    def _build_prompt(
        self, summary: str, article_text: str, key_terms: List[str]
    ) -> str:
        excerpt = article_text[:4000] if article_text else "(нет)"
        terms_str = ", ".join(key_terms[:20]) if key_terms else "(нет)"
        return f"""Ты — эксперт-рецензент научных статей. Оцени резюме по 4 критериям (0..10):
- completeness: отражены ли методы, результаты, выводы
- accuracy: нет ли искажений и галлюцинаций
- coherence: логичность, отсутствие повторов
- style: научный стиль

ПРИМЕР ОТВЕТА:
{{"completeness": 9, "accuracy": 9, "coherence": 8, "style": 9,
  "reasoning": "точно передаёт суть", "suggestions": ["добавить числа"],
  "missing_terms": ["F1-score"]}}

СТАТЬЯ (фрагмент):
---
{excerpt}
---

КЛЮЧЕВЫЕ ТЕРМИНЫ: {terms_str}

РЕЗЮМЕ:
---
{summary}
---

Ответь ТОЛЬКО JSON с полями: completeness, accuracy, coherence, style,
reasoning, suggestions, missing_terms."""

    

    def _parse_json(self, raw: str) -> Optional[Dict[str, Any]]:
        if not raw:
            return None
        # 1. Прямой JSON
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                return d
        except json.JSONDecodeError:
            pass
        # 2. Markdown
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(1))
                if isinstance(d, dict):
                    return d
            except json.JSONDecodeError:
                pass
        # 3. Первый {...}
        m = re.search(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", raw, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(0))
                if isinstance(d, dict):
                    return d
            except json.JSONDecodeError:
                pass
        return None

    def _write(self, state, parsed) -> Dict[str, Any]:
        c = self._f(parsed.get("completeness"))
        a = self._f(parsed.get("accuracy"))
        co = self._f(parsed.get("coherence"))
        s = self._f(parsed.get("style"))
        avg = (c + a + co + s) / 4.0
        state["critic_completeness"] = c
        state["critic_accuracy"] = a
        state["critic_coherence"] = co
        state["critic_style"] = s
        state["critic_score"] = round(avg, 2)
        state["critic_reasoning"] = str(parsed.get("reasoning", ""))[:2000]
        state["critic_suggestions"] = self._list(parsed.get("suggestions"))
        state["critic_missing_terms"] = self._list(parsed.get("missing_terms"))
        state["critic_passed"] = avg >= self.CRITIC_PASS_THRESHOLD
        logger.info("Critic score=%.2f passed=%s", avg, state["critic_passed"])
        return state

    def _fallback(self, state, reason: str) -> Dict[str, Any]:
        state["critic_completeness"] = 5.0
        state["critic_accuracy"] = 5.0
        state["critic_coherence"] = 5.0
        state["critic_style"] = 5.0
        state["critic_score"] = 5.0
        state["critic_reasoning"] = f"Fallback: {reason}"
        state["critic_suggestions"] = []
        state["critic_missing_terms"] = []
        state["critic_passed"] = False
        logger.warning("Critic fallback: %s", reason)
        return state

    @staticmethod
    def _f(v: Any, default: float = 5.0) -> float:
        try:
            f = float(v)
        except (TypeError, ValueError):
            return default
        return max(0.0, min(10.0, f))

    @staticmethod
    def _list(v: Any) -> List[str]:
        if not isinstance(v, list):
            return []
        return [str(x) for x in v if x is not None]