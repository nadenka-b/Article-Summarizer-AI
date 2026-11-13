# backend/data/embeddings.py
"""
Синглтон модели эмбеддингов (SentenceTransformer).

Модель загружается один раз — при первом обращении (лениво).
Все последующие вызовы переиспользуют тот же экземпляр.
"""

import logging
from typing import List, Optional

import numpy as np

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_model = None


def get_embedding_model():
    """Возвращает singleton SentenceTransformer (ленивая загрузка)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        logger.info(
            "Loading embedding model %s (dim=%s)...",
            settings.EMBEDDING_MODEL,
            settings.EMBEDDING_DIM,
        )
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Embedding model loaded")
    return _model


def encode_texts(
    texts: List[str],
    batch_size: int = 32,
) -> Optional[np.ndarray]:
    """
    Кодирует тексты в нормированные векторы.

    Возвращает None, если список пуст. Векторы L2-нормированы —
    скалярное произведение = косинусная близость.
    """
    if not texts:
        return None
    model = get_embedding_model()
    return model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
