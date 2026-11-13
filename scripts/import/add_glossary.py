"""
Импорт глоссария в БД (идемпотентно).

Источники записей (можно комбинировать):
- встроенный набор backend/data/glossary.py (510+ русских записей) — по умолчанию;
- CSV: колонки term;definition;category;field (разделитель ; или ,);
- JSON: список объектов с ключами term, definition, category, field;
- --mine-corpus: добыча частотных терминов из data/processed/*.json
  (названия и аннотации статей) — набор автоматически растёт вместе с корпусом.

Повторный запуск ничего не добавляет (added=0): уникальность (field, term).

Запуск:
    python scripts/import/add_glossary.py [--csv file] [--json file] [--mine-corpus]
"""

import argparse
import asyncio
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# Определяем корень проекта, чтобы пути не зависели от места запуска скрипта.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.data.glossary import BUILTIN_GLOSSARY, GlossaryManager
from backend.db.session import AsyncSessionLocal, engine


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Стоп-слова майнера: служебная лексика, не попадающая в глоссарий.
_MINER_STOPWORDS = {
    "статья", "работы", "работа", "результаты", "результат", "результатов",
    "метод", "методы", "методов", "модель", "модели", "моделей",
    "задачи", "задача", "данные", "данных", "исследования", "исследование",
    "также", "которые", "который", "которая", "всех", "все", "для",
    "или", "это", "что", "как", "при", "на", "в", "и", "с", "по", "из",
    "не", "были", "был", "была", "может", "можно", "разных",
    "новых", "новый", "новая", "различных", "основных",
    "данной", "данного", "этой", "этого", "таким", "такими", "них",
    "т", "д", "п", "г", "вопросы", "проблемы", "подход", "подходы",
    "анализ", "анализа", "объектов", "объект", "системы", "система",
    "the", "of", "and", "in", "for", "to", "on", "with", "a", "an",
    "is", "are", "based", "using", "by", "from", "this", "we",
}

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё]{3,}")


def load_csv_entries(path: Path) -> list[dict[str, Any]]:
    """Читает CSV с колонками term;definition;category;field."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        sample = file.read(4096)
        file.seek(0)

        dialect = csv.Sniffer().sniff(sample, delimiters=";,")
        reader = csv.DictReader(file, dialect=dialect)

        entries = []
        for row in reader:
            term = (row.get("term") or "").strip()
            definition = (row.get("definition") or "").strip()

            if not term or not definition:
                continue

            entries.append({
                "term": term,
                "definition": definition,
                "category": (row.get("category") or "concept").strip(),
                "field": (row.get("field") or "general").strip(),
                "language": "ru",
            })

        return entries


def load_json_entries(path: Path) -> list[dict[str, Any]]:
    """Читает JSON: список объектов {term, definition, category, field}."""
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError("Ожидался JSON-список записей глоссария.")

    entries = []
    for item in data:
        term = str(item.get("term", "")).strip()
        definition = str(item.get("definition", "")).strip()

        if not term or not definition:
            continue

        entries.append({
            "term": term,
            "definition": definition,
            "category": str(item.get("category", "concept")),
            "field": str(item.get("field", "general")),
            "language": str(item.get("language", "ru")),
        })

    return entries


def mine_corpus_terms(
    limit: int = 150,
    min_frequency: int = 3,
) -> list[dict[str, Any]]:
    """
    Добывает частотные термины из названий и аннотаций корпуса.

    Биграммы и униграммы из data/processed/*.json, отсечённые по частоте.
    Определение термина формируется автоматически (частота + источник).
    """
    if not PROCESSED_DIR.is_dir():
        print(f"Папка корпуса не найдена: {PROCESSED_DIR}")
        return []

    unigrams: Counter = Counter()
    bigrams: Counter = Counter()

    for path in PROCESSED_DIR.glob("[0-9]" * 4 + ".json"):
        try:
            with path.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError):
            continue

        text = " ".join(
            part
            for part in (data.get("title"), data.get("abstract"))
            if isinstance(part, str) and part
        )
        if not text:
            continue

        words = _WORD_RE.findall(text.lower())

        for word in words:
            if word not in _MINER_STOPWORDS:
                unigrams[word] += 1

        for first, second in zip(words, words[1:]):
            if (
                first not in _MINER_STOPWORDS
                and second not in _MINER_STOPWORDS
            ):
                bigrams[f"{first} {second}"] += 1

    # Биграммы информативнее униграмм — берём сначала их.
    candidates: list[tuple[str, int]] = []

    for term, frequency in bigrams.most_common():
        if frequency >= min_frequency:
            candidates.append((term, frequency))

    for term, frequency in unigrams.most_common():
        if frequency >= min_frequency * 3 and len(term) >= 6:
            candidates.append((term, frequency))

    entries = []
    for term, frequency in candidates[:limit]:
        entries.append({
            "term": term,
            "definition": (
                f"Термин, встречающийся в корпусе статей "
                f"(частота: {frequency})"
            ),
            "category": "concept",
            "field": "general",
            "language": "ru",
        })

    return entries


async def run(args: argparse.Namespace) -> None:
    manager = GlossaryManager()

    entries: list[dict[str, Any]] = []

    if not args.builtin_only:
        entries.extend(BUILTIN_GLOSSARY)
        print(f"Встроенный глоссарий: {len(BUILTIN_GLOSSARY)} записей")

    if args.csv:
        csv_entries = load_csv_entries(Path(args.csv))
        print(f"Из CSV {args.csv}: {len(csv_entries)} записей")
        entries.extend(csv_entries)

    if args.json:
        json_entries = load_json_entries(Path(args.json))
        print(f"Из JSON {args.json}: {len(json_entries)} записей")
        entries.extend(json_entries)

    if args.mine_corpus:
        mined = mine_corpus_terms(limit=args.mine_limit)
        print(f"Добыто из корпуса: {len(mined)} записей")
        entries.extend(mined)

    if not entries:
        print("Нет записей для импорта.")
        return

    try:
        async with AsyncSessionLocal() as session:
            async with session.begin():
                added, skipped = await manager.add_many(session, entries)
                total = await manager.count(session)
    finally:
        await engine.dispose()

    print(f"\nДобавлено: {added}")
    print(f"Пропущено (уже есть): {skipped}")
    print(f"Всего терминов в глоссарии: {total}")

    if total < 500:
        print(
            "Внимание: в глоссарии меньше 500 записей — цель этапа 6 "
            "не достигнута."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Импорт глоссария в БД (идемпотентно)."
    )
    parser.add_argument("--csv", help="CSV-файл (term;definition;category;field)")
    parser.add_argument("--json", help="JSON-файл со списком записей")
    parser.add_argument(
        "--mine-corpus",
        action="store_true",
        help="Добывать частотные термины из корпуса data/processed.",
    )
    parser.add_argument(
        "--mine-limit",
        type=int,
        default=150,
        help="Максимум добытых из корпуса терминов (по умолчанию 150).",
    )
    parser.add_argument(
        "--builtin-only",
        action="store_true",
        help="Импортировать только встроенный набор.",
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
