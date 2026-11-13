"""
Переиндексация базы знаний: BM25 (in-memory) + Qdrant (векторы).

- Загружает пары «текст → аннотация» из таблицы knowledge_base;
- строит BM25-индекс в памяти KBManager (лениво, при первом поиске);
- считает эмбеддинги и upsert-ит точки в Qdrant
  (HTTP-режим QDRANT_HOST:QDRANT_PORT или локальный QDRANT_LOCAL_PATH).

Если Qdrant недоступен — печатает warning и завершается: BM25 работает,
векторы можно доделать позже повторным запуском (идемпотентно, uuid5).

Запуск:
    python scripts/retrieval/reindex_kb.py [--limit N] [--local PATH] [--dry-run]

Пример локального режима без Docker:
    python scripts/retrieval/reindex_kb.py --local ./data/qdrant_storage
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Определяем корень проекта, чтобы пути не зависели от места запуска скрипта.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


async def run(args: argparse.Namespace) -> None:
    # ВАЖНО: backend-модули импортируются ПОСЛЕ установки env-переменных —
    # настройки (QDRANT_LOCAL_PATH и др.) синглтонятся при первом импорте.
    from backend.core.config import get_settings

    from backend.data.knowledge_base import KBManager
    from backend.db.session import AsyncSessionLocal, engine
    from backend.retrieval.hybrid_retriever import hybrid_search
    from backend.retrieval.vector_index import VectorIndex

    settings = get_settings()

    kb_manager = KBManager()
    vector_index = VectorIndex()

    try:
        async with AsyncSessionLocal() as session:
            total = await kb_manager.initialize(session)
            print(f"Загружено пар из БД: {total}")

            if total == 0:
                print(
                    "База знаний пуста. Сначала запустите "
                    "scripts/import/import_articles.py."
                )
                return
    finally:
        await engine.dispose()

    entries = kb_manager.entries
    if args.limit:
        entries = entries[: args.limit]
        print(f"Ограничение индексации: --limit {args.limit}")

    mode = (
        f"локальный ({settings.QDRANT_LOCAL_PATH})"
        if settings.QDRANT_LOCAL_PATH
        else f"HTTP {settings.QDRANT_HOST}:{settings.QDRANT_PORT}"
    )
    print(f"Режим Qdrant: {mode}")
    print(f"Модель эмбеддингов: {settings.EMBEDDING_MODEL}")

    if args.dry_run:
        print("Dry-run: векторный индекс не заполняется.")
    else:
        print("Заполнение векторного индекса...")
        point_ids = vector_index.add_entries(entries)
        print(f"Точек добавлено/обновлено: {len(point_ids)}")

        points_count = vector_index.count()
        if points_count is not None:
            print(f"Всего точек в коллекции: {points_count}")
        else:
            print("Qdrant недоступен — векторный индекс не заполнен.")

    # Смоук-тест гибридного поиска.
    if args.query:
        print(f"\nСмоук-тест поиска: {args.query!r}")
        results = hybrid_search(kb_manager, vector_index, args.query, top_k=3)

        if not results:
            print("Ничего не найдено.")
        for entry, score in results:
            print(
                f"  [{score:.4f}] {entry.field}/{entry.language}: "
                f"{entry.target_summary[:100]}..."
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Переиндексация БЗ: BM25 + Qdrant."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Проиндексировать только первые N пар (для смоук-теста).",
    )
    parser.add_argument(
        "--local",
        default=None,
        help="Локальный файловый режим Qdrant (путь к папке хранения).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Не заполнять векторный индекс (только BM25 в памяти).",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Смоук-тест гибридного поиска по этому запросу.",
    )
    args = parser.parse_args()

    # Env для локального режима Qdrant — до импортов backend-модулей.
    if args.local:
        os.environ["QDRANT_LOCAL_PATH"] = args.local

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
