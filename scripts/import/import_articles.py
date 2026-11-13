"""
Импорт обработанных статей в БД.

- количество статей не захардкожено: обрабатываются все NNNN.json,
  найденные в data/processed (работает и для 100, и для 1000);
- идемпотентность: существующие статьи по умолчанию ПРОПУСКАЮТСЯ
  (повторный запуск → added=0); флаг --update обновляет записи;
- статьи с извлечённой аннотацией дополнительно попадают в базу знаний
  (knowledge_base) как эталонные пары «текст → аннотация» для RAG;
  повторный запуск не создаёт дублей (ON CONFLICT DO NOTHING).

Запуск:
    python scripts/import/import_articles.py [--update] [--no-kb]
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from sqlalchemy import func, select

# Определяем корень проекта, чтобы пути не зависели от места запуска скрипта.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.core.config import get_settings
from backend.data.knowledge_base import KBManager
from backend.db.models import Article, KnowledgeBase, User
from backend.db.session import AsyncSessionLocal, engine


ARTICLES_DIR = PROJECT_ROOT / "data" / "processed"
CATEGORIES_FILE = ARTICLES_DIR / "article_categories.json"


def load_json(path: Path) -> dict[str, Any]:
    """Загружает JSON-файл и проверяет, что в нём находится объект."""
    try:
        with path.open("r", encoding="utf-8") as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Не удалось прочитать {path.name}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"{path.name}: ожидался JSON-объект.")

    return data


def find_article_numbers() -> list[int]:
    """Определяет номера статей по найденным NNNN.json (без хардкода)."""
    if not ARTICLES_DIR.is_dir():
        raise FileNotFoundError(
            f"Папка с JSON-файлами не найдена: {ARTICLES_DIR}"
        )

    numbers = [
        int(path.stem)
        for path in ARTICLES_DIR.glob("[0-9]" * 4 + ".json")
        if path.stem.isdigit()
    ]

    if not numbers:
        raise ValueError(
            f"В {ARTICLES_DIR} не найдено файлов вида NNNN.json. "
            "Сначала запустите prepare_articles.py."
        )

    return sorted(numbers)


def validate_source_data(numbers: list[int]) -> list[dict[str, Any]]:
    """Проверяет все JSON-файлы до начала импорта в БД."""
    errors: list[str] = []
    articles: list[dict[str, Any]] = []

    if not CATEGORIES_FILE.is_file():
        raise FileNotFoundError(
            f"Файл классификации не найден: {CATEGORIES_FILE}"
        )

    categories_data = load_json(CATEGORIES_FILE)
    classifications = categories_data.get("articles")

    if not isinstance(classifications, dict):
        raise ValueError(
            "В article_categories.json отсутствует объект 'articles'."
        )

    for number in numbers:
        filename = f"{number:04d}.json"
        path = ARTICLES_DIR / filename

        if not path.is_file():
            errors.append(f"Не найден файл {filename}")
            continue

        try:
            data = load_json(path)
        except ValueError as exc:
            errors.append(str(exc))
            continue

        if data.get("number") != number:
            errors.append(
                f"{filename}: номер внутри JSON не совпадает "
                f"с ожидаемым ({number})."
            )

        title = data.get("title")
        if not isinstance(title, str) or not title.strip():
            errors.append(f"{filename}: отсутствует название статьи.")

        full_text = data.get("full_text")
        if not isinstance(full_text, str) or not full_text.strip():
            errors.append(f"{filename}: отсутствует полный текст статьи.")

        if data.get("extraction_status") != "success":
            errors.append(
                f"{filename}: extraction_status не равен 'success'."
            )

        language = data.get("language")
        if not isinstance(language, str) or not language.strip():
            errors.append(f"{filename}: отсутствует язык статьи.")

        classification = classifications.get(str(number))
        if not isinstance(classification, dict):
            errors.append(
                f"{filename}: отсутствует запись классификации "
                f"в article_categories.json."
            )
            continue

        field = classification.get("field")
        subfield = classification.get("subfield")

        if not isinstance(field, str) or not field.strip():
            errors.append(f"{filename}: некорректное поле field.")

        if not isinstance(subfield, str) or not subfield.strip():
            errors.append(f"{filename}: некорректное поле subfield.")

        if isinstance(field, str) and len(field) > 20:
            errors.append(f"{filename}: field длиннее 20 символов.")

        if isinstance(subfield, str) and len(subfield) > 100:
            errors.append(f"{filename}: subfield длиннее 100 символов.")

        if not isinstance(full_text, str) or not full_text.strip():
            continue

        abstract = data.get("abstract")
        has_abstract = (
            isinstance(abstract, str)
            and bool(abstract.strip())
            and data.get("abstract_extraction_status") == "success"
        )

        articles.append({
            "number": number,
            "title": title.strip() if isinstance(title, str) else "",
            "full_text": full_text,
            "language": language.strip().lower(),
            "field": field.strip() if isinstance(field, str) else "",
            "subfield": subfield.strip() if isinstance(subfield, str) else "",
            "authors": data.get("authors"),
            "abstract": abstract.strip() if has_abstract else None,
        })

    if errors:
        print("\nОбнаружены ошибки при проверке исходных данных:")
        for error in errors:
            print(f"  - {error}")

        raise ValueError(
            f"Проверка не пройдена. Количество ошибок: {len(errors)}. "
            "Импорт не выполнялся."
        )

    return articles


def normalize_authors(authors: Any) -> str | None:
    """Приводит авторов к строковому формату поля Article.authors."""
    if authors is None:
        return None

    if isinstance(authors, str):
        return authors.strip() or None

    if isinstance(authors, (list, dict)):
        return json.dumps(authors, ensure_ascii=False)

    return str(authors)


async def import_articles(update: bool, with_kb: bool) -> None:
    settings = get_settings()

    admin_email = settings.ADMIN_EMAIL
    if not admin_email or not admin_email.strip():
        raise ValueError(
            "Не задан ADMIN_EMAIL. Добавьте его в файл .env."
        )

    admin_email = admin_email.strip().lower()

    print("Проверка исходных JSON-файлов...")
    numbers = find_article_numbers()
    print(f"Найдено файлов статей: {len(numbers)}")

    articles_data = validate_source_data(numbers)

    kb_candidates = [a for a in articles_data if a["abstract"]]
    print(
        f"Проверка пройдена: {len(articles_data)} статей, "
        f"из них с аннотацией для БЗ: {len(kb_candidates)}"
    )

    inserted = 0
    updated = 0
    skipped = 0
    kb_added = 0
    kb_skipped = 0

    kb_manager = KBManager()

    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():

                # Ищем существующего активного администратора.
                admin_result = await session.execute(
                    select(User).where(
                        func.lower(User.email) == admin_email
                    )
                )
                admin = admin_result.scalar_one_or_none()

                if admin is None:
                    raise ValueError(
                        f"Пользователь с email {admin_email} не найден. "
                        "Создайте администратора (register + смена роли)."
                    )

                if admin.role != "admin":
                    raise ValueError(
                        "Указанный пользователь не имеет роль admin."
                    )

                if not admin.is_active:
                    raise ValueError(
                        "Указанный пользователь неактивен."
                    )

                print(f"\nВладелец корпуса найден: {admin.email}")

                # Загружаем уже существующие записи корпуса.
                existing_result = await session.execute(
                    select(Article).where(
                        Article.source == "corpus"
                    )
                )
                existing_articles = existing_result.scalars().all()

                existing_by_source_id: dict[str, Article] = {}

                for article in existing_articles:
                    if article.source_id in existing_by_source_id:
                        raise ValueError(
                            "В БД обнаружены дубликаты статей корпуса "
                            f"с source_id={article.source_id}."
                        )

                    existing_by_source_id[article.source_id] = article

                # Создаём или обновляем записи.
                for data in articles_data:
                    source_id = str(data["number"])

                    sections = (
                        json.dumps(
                            {"abstract": data["abstract"]},
                            ensure_ascii=False,
                        )
                        if data["abstract"]
                        else None
                    )

                    article_values = {
                        "owner_id": admin.id,
                        "title": data["title"],
                        "source": "corpus",
                        "source_id": source_id,
                        "field": data["field"],
                        "subfield": data["subfield"],
                        "language": data["language"],
                        "full_text": data["full_text"],
                        "sections": sections,
                        "char_count": len(data["full_text"]),
                        "authors": normalize_authors(data["authors"]),
                        "status": "extracted",
                    }

                    existing = existing_by_source_id.get(source_id)

                    if existing is not None:
                        if update:
                            for key, value in article_values.items():
                                setattr(existing, key, value)
                            updated += 1
                        else:
                            skipped += 1
                    else:
                        session.add(Article(**article_values))
                        inserted += 1

                # Эталонные пары для базы знаний (RAG).
                if with_kb:
                    print(
                        "\nЗаполнение базы знаний парами "
                        "«текст → аннотация»..."
                    )

                    kb_pairs = [
                        {
                            "field": data["field"],
                            "language": data["language"],
                            "source_text": data["full_text"],
                            "target_summary": data["abstract"],
                            "source": "corpus",
                            "quality_score": None,
                        }
                        for data in kb_candidates
                    ]

                    kb_added, kb_skipped = await kb_manager.add_many(
                        session, kb_pairs
                    )

                # Проверяем, что изменения отправлены в БД
                # до завершения транзакции.
                await session.flush()

        print("\nТранзакция успешно завершена.")
        print(f"Новых статей: {inserted}")
        print(f"Обновлено статей: {updated}")
        print(f"Пропущено (уже в БД): {skipped}")
        if with_kb:
            print(
                f"БЗ: добавлено пар: {kb_added}, "
                f"пропущено (уже есть): {kb_skipped}"
            )

    finally:
        await engine.dispose()

    # Дополнительная проверка результата после commit.
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(func.count(Article.id)).where(
                Article.source == "corpus"
            )
        )
        total = result.scalar_one()
        print(f"\nВсего статей корпуса в БД: {total}")

        kb_result = await session.execute(
            select(func.count(KnowledgeBase.id))
        )
        kb_total = kb_result.scalar_one()
        print(f"Всего пар в базе знаний: {kb_total}")

        if total == 0:
            raise RuntimeError("После импорта в БД нет статей корпуса.")

    print("\nИмпорт успешно завершён.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Импорт статей в БД (идемпотентно)."
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Обновлять существующие статьи (по умолчанию пропускаются).",
    )
    parser.add_argument(
        "--no-kb",
        action="store_true",
        help="Не заполнять базу знаний парами «текст → аннотация».",
    )
    args = parser.parse_args()

    asyncio.run(import_articles(update=args.update, with_kb=not args.no_kb))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nОШИБКА ИМПОРТА: {exc}")
        raise
