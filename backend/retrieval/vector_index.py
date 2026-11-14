# backend/retrieval/vector_index.py
"""
Векторный индекс на Qdrant.

Особенности:
- детерминированные id точек: uuid5 от «field|language|source_text_hash» —
  повторный upsert той же пары не создаёт дубликат;
- batch-upsert пачками;
- два режима: HTTP-сервер (QDRANT_HOST:QDRANT_PORT) или локальное
  файловое хранилище (QDRANT_LOCAL_PATH) без сервера;
- Qdrant недоступен → warning и пустой результат, без исключений
  (гибридный поиск деградирует до BM25).
"""

import logging
import uuid
from typing import List, Optional, Sequence, Tuple

from backend.core.config import get_settings
from backend.data.embeddings import encode_texts
from backend.data.knowledge_base import KBEntry, sha256_text

logger = logging.getLogger(__name__)
settings = get_settings()

# Namespace для детерминированных id точек.
_POINT_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

BATCH_SIZE = 64


class VectorIndex:
    """Семантический поиск пар БЗ через Qdrant."""

    def __init__(self, collection: Optional[str] = None) -> None:
        self._collection = collection or settings.QDRANT_COLLECTION_NAME
        self._client = None
        self._unavailable = False  # флаг: сервер недоступен, не спамим логами

    # ------------------------------------------------------------------------
    # Клиент
    # ------------------------------------------------------------------------
    def _get_client(self):
        """Ленивое создание Qdrant-клиента. None, если режим не задан."""
        if self._unavailable:
            return None
        if self._client is None:
            try:
                from qdrant_client import QdrantClient

                if settings.QDRANT_LOCAL_PATH:
                    self._client = QdrantClient(
                        path=settings.QDRANT_LOCAL_PATH,
                    )
                else:
                    self._client = QdrantClient(
                        host=settings.QDRANT_HOST,
                        port=settings.QDRANT_PORT,
                        api_key=settings.QDRANT_API_KEY,
                        timeout=5,
                    )
            except Exception as e:
                self._mark_unavailable(e)
        return self._client

    def _mark_unavailable(self, error: Exception) -> None:
        if not self._unavailable:
            logger.warning(
                "Qdrant недоступен (%s); векторный поиск отключён, "
                "гибридный поиск деградирует до BM25",
                error,
            )
        self._unavailable = True

    # ------------------------------------------------------------------------
    # Коллекция
    # ------------------------------------------------------------------------
    def ensure_collection(self) -> bool:
        """Создаёт коллекцию, если её нет. False, если Qdrant недоступен."""
        client = self._get_client()
        if client is None:
            return False
        try:
            from qdrant_client.models import Distance, VectorParams

            if not client.collection_exists(self._collection):
                client.create_collection(
                    collection_name=self._collection,
                    vectors_config=VectorParams(
                        size=settings.EMBEDDING_DIM,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "Qdrant: создана коллекция %s (dim=%d)",
                    self._collection,
                    settings.EMBEDDING_DIM,
                )
            return True
        except Exception as e:
            self._mark_unavailable(e)
            return False

    # ------------------------------------------------------------------------
    # Идентификаторы точек
    # ------------------------------------------------------------------------
    @staticmethod
    def _point_id(field: str, language: str, source_hash: str) -> str:
        """Детерминированный uuid5: одна пара — одна точка."""
        key = f"{field}|{language}|{source_hash}"
        return str(uuid.uuid5(_POINT_NAMESPACE, key))

    # ------------------------------------------------------------------------
    # Добавление
    # ------------------------------------------------------------------------
    def add_entries(self, entries: Sequence[KBEntry]) -> List[str]:
        """
        Считает эмбеддинги search_text и upsert-ит точки пачками.

        Возвращает список добавленных id точек; [] при недоступном Qdrant.
        """
        if not entries:
            return []

        client = self._get_client()
        if client is None:
            return []

        try:
            from qdrant_client.models import PointStruct

            self.ensure_collection()

            vectors = encode_texts([e.search_text for e in entries])
            if vectors is None:
                return []

            point_ids: List[str] = []
            for start in range(0, len(entries), BATCH_SIZE):
                batch_entries = entries[start : start + BATCH_SIZE]
                batch_vectors = vectors[start : start + BATCH_SIZE]

                points = []
                for entry, vector in zip(batch_entries, batch_vectors):
                    point_id = self._point_id(
                        entry.field,
                        entry.language,
                        sha256_text(entry.source_text),
                    )
                    point_ids.append(point_id)
                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector.tolist(),
                            payload=self._entry_payload(entry),
                        )
                    )

                client.upsert(
                    collection_name=self._collection,
                    points=points,
                    wait=True,
                )

            logger.info(
                "Qdrant: добавлено/обновлено %d точек в %s",
                len(point_ids),
                self._collection,
            )
            return point_ids
        except Exception as e:
            self._mark_unavailable(e)
            return []

    @staticmethod
    def _entry_payload(entry: KBEntry) -> dict:
        """Точка хранит всё нужное для few-shot без обращения к БД."""
        return {
            "kb_id": entry.id,
            "field": entry.field,
            "language": entry.language,
            "target_summary": entry.target_summary,
            "source_text": entry.search_text,  # урезанное представление
            "source": entry.source,
        }

    @staticmethod
    def _entry_from_payload(payload: dict) -> KBEntry:
        return KBEntry(
            id=payload.get("kb_id") or "",
            field=payload.get("field") or "general",
            language=payload.get("language") or "ru",
            source_text=payload.get("source_text") or "",
            target_summary=payload.get("target_summary") or "",
            search_text=payload.get("source_text") or "",
            source=payload.get("source"),
        )

    # ------------------------------------------------------------------------
    # Поиск
    # ------------------------------------------------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        field: Optional[str] = None,
    ) -> List[Tuple[KBEntry, float]]:
        """Семантический поиск. [] при недоступном Qdrant."""
        client = self._get_client()
        if client is None or not query.strip():
            return []

        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            query_vector = encode_texts([query])
            if query_vector is None:
                return []

            query_filter = None
            if field:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="field",
                            match=MatchValue(value=field),
                        )
                    ]
                )

            result = client.query_points(
                collection_name=self._collection,
                query=query_vector[0].tolist(),
                query_filter=query_filter,
                limit=top_k,
                with_payload=True,
            )

            return [
                (
                    self._entry_from_payload(hit.payload or {}),
                    float(hit.score),
                )
                for hit in result.points
            ]
        except Exception as e:
            self._mark_unavailable(e)
            return []

    # ------------------------------------------------------------------------
    # Обслуживание
    # ------------------------------------------------------------------------
    def delete_by_field(self, field: str) -> bool:
        """Удаляет точки области. False при недоступном Qdrant."""
        client = self._get_client()
        if client is None:
            return False
        try:
            from qdrant_client.models import FieldCondition, Filter, MatchValue

            client.delete(
                collection_name=self._collection,
                points_selector=Filter(
                    must=[
                        FieldCondition(
                            key="field",
                            match=MatchValue(value=field),
                        )
                    ]
                ),
                wait=True,
            )
            return True
        except Exception as e:
            self._mark_unavailable(e)
            return False

    def count(self) -> Optional[int]:
        """Количество точек в коллекции; None при недоступном Qdrant."""
        client = self._get_client()
        if client is None:
            return None
        try:
            info = client.get_collection(self._collection)
            return info.points_count
        except Exception as e:
            self._mark_unavailable(e)
            return None


# Singleton
_vector_index: Optional[VectorIndex] = None


def get_vector_index() -> VectorIndex:
    global _vector_index
    if _vector_index is None:
        _vector_index = VectorIndex()
    return _vector_index
