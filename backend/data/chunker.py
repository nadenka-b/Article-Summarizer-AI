# backend/data/chunker.py
"""
Чанкинг длинных текстов для иерархической суммаризации.

Стратегия (этап 8): текст длиннее прямого лимита разбивается на чанки
по границам предложений с перекрытием; Summarizer сначала делает
резюме каждого чанка, затем агрегирует их в финальную аннотацию.
"""

import hashlib
import logging
import re
from typing import Any, Dict, List

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Конец предложения (включая русские кавычки-хвосты и скобки).
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?…])\s+")


def split_into_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения, сохраняя порядок."""
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]


def needs_chunking(text: str) -> bool:
    """Нужен ли чанкинг: текст длиннее прямого лимита суммаризатора."""
    return len(text) > settings.SUMMARIZER_DIRECT_CHARS


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[Dict[str, Any]]:
    """
    Пакует предложения в чанки размером ~chunk_size символов
    с перекрытием overlap символов (последние предложения чанка
    повторяются в начале следующего — контекст не рвётся).

    Возвращает список словарей:
        [{"index": 0, "text": ..., "char_count": n, "text_hash": sha256}, ...]

    Гарантии:
    - текст покрыт полностью, без потерь и дублей предложений
      (перекрытие — единственное намеренное исключение);
    - слишком длинное предложение (без точек) режется жёстко.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE_CHARS
    overlap = overlap or settings.CHUNK_OVERLAP_CHARS
    overlap = max(0, min(overlap, chunk_size // 2))

    text = text.strip()
    if not text:
        return []

    if len(text) <= chunk_size:
        return [_make_chunk(0, text)]

    sentences = split_into_sentences(text)

    # Предложения-гиганты (без знаков конца) режем жёстко.
    normalized: List[str] = []
    for sentence in sentences:
        while len(sentence) > chunk_size:
            normalized.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        if sentence:
            normalized.append(sentence)

    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in normalized:
        sentence_len = len(sentence) + (1 if current else 0)
        if current and current_len + sentence_len > chunk_size:
            chunks.append(" ".join(current))
            # Перекрытие: хвост текущего чанка переезжает в следующий.
            tail: List[str] = []
            tail_len = 0
            for part in reversed(current):
                if tail_len + len(part) > overlap:
                    break
                tail.insert(0, part)
                tail_len += len(part) + 1
            current = list(tail)
            current_len = sum(len(p) + 1 for p in current)
        current.append(sentence)
        current_len += len(sentence) + 1

    if current:
        chunks.append(" ".join(current))

    return [
        _make_chunk(index, chunk) for index, chunk in enumerate(chunks)
    ]


def _make_chunk(index: int, text: str) -> Dict[str, Any]:
    return {
        "index": index,
        "text": text,
        "char_count": len(text),
        "text_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }
