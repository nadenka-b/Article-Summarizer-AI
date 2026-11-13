# backend/retrieval/bm25_index.py
"""
Лексический поиск BM25 (rank_bm25.BM25Okapi).

Токенизация рассчитана на русский и английский:
lowercase → выделение словоформ → отбрасывание стоп-слов и коротких токенов.
"""

import logging
import re
from typing import Any, Callable, List, Optional

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{2,}")

# Минимальный набор стоп-слов: русский + английский.
# Полные списки не нужны — BM25 сам понижает вес частых токенов (IDF).
STOPWORDS = {
    # русские
    "и", "в", "во", "не", "на", "с", "со", "по", "для", "от", "при", "к",
    "у", "как", "что", "это", "этом", "этого", "или", "но", "а", "же",
    "бы", "был", "была", "были", "быть", "есть", "так", "также", "то",
    "о", "об", "из", "за", "до", "после", "между", "чем", "когда", "где",
    "они", "он", "она", "мы", "вы", "т", "д", "п", "е",
    # английские
    "the", "a", "an", "of", "in", "on", "for", "to", "and", "or", "is",
    "are", "was", "were", "be", "been", "with", "as", "by", "at", "from",
    "this", "that", "these", "those", "it", "its", "we", "our", "their",
}


def tokenize(text: str) -> List[str]:
    """Простая токенизация для BM25: RU/EN словоформы без стоп-слов."""
    return [
        token
        for token in _TOKEN_RE.findall(text.lower())
        if token not in STOPWORDS
    ]


class BM25Index:
    """
    In-memory BM25-индекс над записями.

    entry_getter — функция entry -> текст для индексации
    (по умолчанию str(entry)).
    """

    def __init__(
        self,
        entry_getter: Callable[[Any], str] = str,
    ) -> None:
        self._entry_getter = entry_getter
        self._entries: List[Any] = []
        self._bm25: Optional[BM25Okapi] = None

    # ------------------------------------------------------------------------
    # Построение
    # ------------------------------------------------------------------------
    def initialize(self, entries: List[Any]) -> None:
        """Строит индекс по записям (полная перестройка)."""
        self._entries = list(entries)
        corpus = [tokenize(self._entry_getter(e)) for e in self._entries]

        # BM25Okapi падает на пустом корпусе или корпусе из пустых документов
        if corpus and any(corpus):
            self._bm25 = BM25Okapi(corpus)
        else:
            self._bm25 = None
            logger.warning("BM25Index: пустой корпус, индекс не построен")

    def __len__(self) -> int:
        return len(self._entries)

    # ------------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> List[tuple[Any, float]]:
        """
        Возвращает [(entry, score)] отсортированные по убыванию score.

        Только score > 0: записи без лексического совпадения не возвращаем.
        """
        if self._bm25 is None or not self._entries:
            return []

        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(
            (
                (self._entries[i], float(score))
                for i, score in enumerate(scores)
                if score > 0
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        return ranked[:top_k]
