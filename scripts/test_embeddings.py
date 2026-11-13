
import sys
import asyncio
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.db.models import ArticleChunk
from backend.db.session import AsyncSessionLocal


MODEL_NAME = "BAAI/bge-m3"
TEST_CHUNKS_COUNT = 5
EXPECTED_DIMENSION = 1024


async def get_test_chunks():
    """Получает несколько чанков из SQLite."""
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(ArticleChunk)
            .order_by(
                ArticleChunk.article_id,
                ArticleChunk.chunk_index
            )
            .limit(TEST_CHUNKS_COUNT)
        )

        chunks = list(result.scalars().all())

    if not chunks:
        raise ValueError("В таблице article_chunks нет чанков.")

    return chunks


def main():
    print("Загрузка модели:", MODEL_NAME)

    model = SentenceTransformer(MODEL_NAME)

    print("Модель загружена.")

    chunks = asyncio.run(get_test_chunks())

    print(f"\nПолучено чанков: {len(chunks)}")

    texts = [chunk.chunk_text for chunk in chunks]

    print("\nСоздание эмбеддингов...")

    embeddings = model.encode(
        texts,
        batch_size=5,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    print("\nРезультаты:")

    print("Форма массива:", embeddings.shape)
    print("Количество эмбеддингов:", len(embeddings))
    print("Размерность вектора:", embeddings.shape[1])

    norms = np.linalg.norm(embeddings, axis=1)

    print("\nНормы векторов:")
    for i, norm in enumerate(norms):
        print(f"Чанк {i + 1}: {norm:.6f}")

    print("\nПроверки:")

    print(
        "Размерность 1024:",
        embeddings.shape[1] == EXPECTED_DIMENSION
    )

    print(
        "Все значения конечные:",
        np.isfinite(embeddings).all()
    )

    print(
        "Векторы нормализованы:",
        np.allclose(norms, 1.0, atol=1e-5)
    )

    print("\nПервые 10 значений первого эмбеддинга:")
    print(embeddings[0][:10])

    print("\nТест завершён. Данные в БД не изменялись.")


if __name__ == "__main__":
    main()
