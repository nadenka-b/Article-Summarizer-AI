# backend/agents/validator.py
"""Validator — формальная проверка резюме."""

import logging
import re
from typing import Any, Dict, List

from backend.agents.base import Agent

logger = logging.getLogger(__name__)


class ValidatorAgent(Agent):
    """Проверка длины, плагиата, покрытия терминов."""

    name = "validator"
    MIN_WORDS: int = 50
    MAX_WORDS: int = 500
    MIN_PLAGIARISM_SENTENCE_WORDS: int = 15
    MIN_KEY_TERMS: int = 2

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        summary = (state.get("summary") or "").strip()
        article_text = (state.get("article_text") or "").strip()
        key_terms: List[str] = state.get("key_terms") or []

        errors: List[str] = []
        warnings: List[str] = []

        if not summary:
            errors.append("Резюме пустое")
            return self._write(state, errors, warnings)

        wc = self._count_words(summary)
        if wc < self.MIN_WORDS:
            errors.append(f"Резюме слишком короткое: {wc} слов (мин. {self.MIN_WORDS})")
        if wc > self.MAX_WORDS:
            warnings.append(f"Резюме длинное: {wc} слов (макс. {self.MAX_WORDS})")

        if article_text:
            copied = self._find_copied(summary, article_text)
            if copied:
                errors.append(f"Найдено {len(copied)} дословных копий из исходника")

        if key_terms:
            missing = self._missing_terms(summary, key_terms)
            covered = len(key_terms) - len(missing)
            # Не требуем больше терминов, чем вообще было передано
            required = min(self.MIN_KEY_TERMS, len(key_terms))
            if covered < required:
                errors.append(
                    f"Покрыто {covered} терминов из {len(key_terms)} "
                    f"(мин. {required})"
                )
            elif missing:
                warnings.append(f"Не хватает терминов: {', '.join(missing[:5])}")
        
        return self._write(state, errors, warnings)

    def _write(self, state, errors, warnings) -> Dict[str, Any]:
        state["validator_errors"] = errors
        state["validator_warnings"] = warnings
        state["validation_passed"] = len(errors) == 0
        logger.info(
            "Validation %s: %d errors, %d warnings",
            "PASSED" if not errors else "FAILED", len(errors), len(warnings),
        )
        return state

    @staticmethod
    def _count_words(text: str) -> int:
        return len(re.findall(r"\w+", text, flags=re.UNICODE))

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [p.strip() for p in re.split(r"(?<=[.!?])\s+", text) if p.strip()]

    def _find_copied(self, summary: str, article: str) -> List[str]:
        article_norm = self._normalize(article)
        return [
            s for s in self._split_sentences(summary)
            if self._count_words(s) >= self.MIN_PLAGIARISM_SENTENCE_WORDS
            and self._normalize(s) in article_norm
        ]

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"\s+", " ", text.lower()).strip()

    @staticmethod
    def _missing_terms(summary: str, terms: List[str]) -> List[str]:
        s = summary.lower()
        return [t for t in terms if t.lower() not in s]