# backend/retrieval/hybrid_retriever.py
"""
Гибридный поиск: BM25 (лексический) + Qdrant (семантический) + RRF.

Используется на Этапе 9 для подстановки few-shot примеров в промпт LLM;
здесь — только поиск, без вызова LLM.

Если Qdrant недоступен, hybrid_search деградирует до чистого BM25
(warning в логе).
"""

import logging
from typing import Any, List, Optional, Sequence, Tuple

from backend.core.config import get_settings
from backend.data.knowledge_base import KBEntry, KBManager
from backend.retrieval.vector_index import VectorIndex

logger = logging.getLogger(__name__)
settings = get_settings()


def reciprocal_rank_fusion(
    rankings: Sequence[List[Tuple[Any, float]]],
    k: int = 60,
    top_k: int = 5,
    dedup_key=lambda entry: entry.id,
) -> List[Tuple[Any, float]]:
    """
    Слияние нескольких ранжирований по формуле RRF:

        score(d) = sum over спискам 1 / (k + rank + 1)

    - ранги с нуля;
    - дубликаты схлопываются по dedup_key (запись берётся из первого
      списка, где она встретилась);
    - возвращается top_k записей по сумме RRF-баллов.
    """
    scores = {}
    first_seen: dict = {}

    for ranking in rankings:
        for rank, (entry, _original_score) in enumerate(ranking):
            key = dedup_key(entry)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank + 1)
            if key not in first_seen:
                first_seen[key] = entry

    fused = sorted(
        scores.items(), key=lambda pair: pair[1], reverse=True
    )[:top_k]

    return [(first_seen[key], score) for key, score in fused]


def hybrid_search(
    kb_manager: KBManager,
    vector_index: VectorIndex,
    query: str,
    field: Optional[str] = None,
    top_k: Optional[int] = None,
) -> List[Tuple[KBEntry, float]]:
    """
    Гибридный поиск пар БЗ: BM25 + вектора, слияние через RRF.

    Деградация: если Qdrant недоступен (search вернул []), результат —
    только BM25 с warning'ом в логе.
    """
    top_k = top_k or settings.HYBRID_TOP_K

    # Каждый источник отдаёт больше кандидатов, чем нужно, —
    # чтобы RRF было что сливать.
    candidates_k = max(top_k * 3, 10)

    bm25_results = kb_manager.search(query, field=field, top_k=candidates_k)
    vector_results = vector_index.search(
        query, top_k=candidates_k, field=field
    )

    rankings = [bm25_results]
    if vector_results:
        rankings.append(vector_results)
    else:
        logger.warning(
            "hybrid_search: векторный поиск пуст (Qdrant недоступен?), "
            "работает только BM25"
        )

    return reciprocal_rank_fusion(
        rankings,
        k=settings.RRF_K,
        top_k=top_k,
    )


def format_few_shot_examples(
    results: List[Tuple[KBEntry, float]],
    max_examples: int = 3,
    summary_chars: int = 700,
) -> str:
    """
    Форматирует найденные пары в блок few-shot примеров для промпта.

    Этап 9 подключит это к промптам Summarizer/Critic.
    """
    if not results:
        return ""

    blocks = []
    for entry, score in results[:max_examples]:
        summary = entry.target_summary[:summary_chars]
        blocks.append(
            f"ПРИМЕР (схожесть {score:.3f}):\n"
            f"Фрагмент статьи: {entry.search_text[:300]}...\n"
            f"Эталонная аннотация: {summary}"
        )
    return "\n\n".join(blocks)
