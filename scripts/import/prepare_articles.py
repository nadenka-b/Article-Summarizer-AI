import argparse
import json
import re
from pathlib import Path

import pymupdf
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException


# --------------------------------------------------
# НАСТРОЙКИ
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

LICENSES_FILE = PROJECT_ROOT / "data" / "import" / "licenses.md"
PDF_DIR = PROJECT_ROOT / "data" / "raw" / "pdfs"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Фиксируем seed, чтобы результаты определения языка
# были воспроизводимыми при повторном запуске.
DetectorFactory.seed = 0

MIN_TEXT_LENGTH_FOR_LANGUAGE = 100
MAX_LANGUAGE_TEXT_LENGTH = 20000

# Заголовки аннотаций. Поддерживаются как отдельные строки,
# так и заголовки с текстом после двоеточия/точки.
ABSTRACT_HEADING_PATTERN = re.compile(
    r"(?i)(?<!\w)"
    r"(?P<header>аннотация|резюме|abstract|summary)"
    r"(?=[ \t]*[.:]|[ \t]{2,}|\r?\n|$)"
)


# Заголовки ключевых слов — конец соответствующей аннотации.
KEYWORDS_PATTERN = re.compile(
    r"(?i)(?<!\w)(?:ключевые\s+слова|keywords)\b"
    r"[ \t]*\**[ \t]*:?"
)

RUSSIAN_ABSTRACT_HEADINGS = {"аннотация", "резюме"}
ENGLISH_ABSTRACT_HEADINGS = {"abstract", "summary"}


# --------------------------------------------------
# ОБРАБОТКА MARKDOWN
# --------------------------------------------------

def split_markdown_row(line: str) -> list[str]:
    """
    Разбивает строку Markdown-таблицы на ячейки.

    Учитывает:
    - экранированные символы \\|;
    - символы | внутри фрагментов кода в обратных кавычках;
    - начальный и конечный разделители таблицы.
    """
    line = line.strip()

    if line.startswith("|"):
        line = line[1:]

    if line.endswith("|") and not line.endswith("\\|"):
        line = line[:-1]

    cells = []
    current_cell = []
    in_code = False
    i = 0

    while i < len(line):
        char = line[i]

        if char == "\\" and i + 1 < len(line):
            current_cell.append(line[i + 1])
            i += 2
            continue

        if char == "`":
            in_code = not in_code
            current_cell.append(char)
            i += 1
            continue

        if char == "|" and not in_code:
            cells.append("".join(current_cell).strip())
            current_cell = []
        else:
            current_cell.append(char)

        i += 1

    cells.append("".join(current_cell).strip())
    return cells


def is_markdown_separator_row(cells: list[str]) -> bool:
    """Проверяет, является ли строка разделителем Markdown-таблицы."""
    if not cells:
        return False

    return all(
        re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) is not None
        for cell in cells
    )


def extract_markdown_link(text: str) -> tuple[str, str | None]:
    """
    Извлекает подпись и URL из Markdown-ссылки.

    Например:
    [https://example.com](https://example.com)

    Возвращает:
    ("https://example.com", "https://example.com")
    """
    match = re.search(
        r"\[([^\]]+)\]\((https?://[^)\s]+)\)",
        text
    )

    if match:
        return match.group(1).strip(), match.group(2).strip()

    return text.strip(), None


def clean_metadata_value(value: str) -> str:
    """Удаляет лишнее форматирование из значения метаданных."""
    value, _ = extract_markdown_link(value)

    value = value.strip()
    value = value.replace("<br>", " ")
    value = value.replace("<br/>", " ")

    return re.sub(r"\s+", " ", value).strip()


def normalize_doi(value: str) -> str | None:
    """Извлекает DOI из обычного текста или URL."""
    value, link_url = extract_markdown_link(value)
    search_text = link_url if link_url else value

    match = re.search(
        r"10\.\d{4,9}/[-._;()/:A-Z0-9]+",
        search_text,
        flags=re.IGNORECASE
    )

    if not match:
        return None

    return match.group(0).rstrip(".,;:")


def normalize_source_url(value: str) -> str | None:
    """Извлекает URL из Markdown-ссылки или текста."""
    _, link_url = extract_markdown_link(value)

    if link_url:
        return link_url

    value = value.strip()

    if re.match(r"^https?://", value, flags=re.IGNORECASE):
        return value

    return None


def extract_article_number(value: str) -> int | None:
    """Извлекает номер статьи из первой колонки таблицы."""
    match = re.search(r"\d+", value)

    if not match:
        return None

    return int(match.group(0))


def parse_license_table(file_path: Path) -> dict[int, dict]:
    """
    Читает Markdown-файл с таблицей лицензий.

    Ожидается 10 колонок:
    1. №
    2. Название
    3. Авторы
    4. Год
    5. Журнал
    6. DOI
    7. Источник
    8. Лицензия
    9. Использование
    10. Примечания об обработке
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Не найден файл с метаданными: {file_path}"
        )

    articles = {}

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line or "|" not in line:
                continue

            cells = split_markdown_row(line)

            if is_markdown_separator_row(cells):
                continue

            if cells and cells[0].strip().lower() in {
                "№", "номер", "no.", "no"
            }:
                continue

            if len(cells) != 10:
                continue

            article_number = extract_article_number(cells[0])

            if article_number is None:
                continue

            if article_number in articles:
                print(
                    f"Предупреждение: повторяется номер статьи "
                    f"{article_number}, строка {line_number}. "
                    f"Будет использована последняя запись."
                )

            doi = normalize_doi(cells[5])
            source_url = normalize_source_url(cells[6])

            articles[article_number] = {
                "number": article_number,
                "title": clean_metadata_value(cells[1]),
                "authors": clean_metadata_value(cells[2]),
                "year": clean_metadata_value(cells[3]),
                "journal": clean_metadata_value(cells[4]),
                "doi": doi,
                "source_url": source_url,
                "license": clean_metadata_value(cells[7]),
                "usage_notes": clean_metadata_value(cells[8]),
                "processing_notes": clean_metadata_value(cells[9]),
            }

    return articles


# --------------------------------------------------
# ПОИСК PDF
# --------------------------------------------------

def find_pdf_files(pdf_dir: Path) -> dict[int, Path]:
    """
    Находит PDF-файлы и сопоставляет их с номерами статей.

    Например:
    0028_статья.pdf -> 28
    0054_статья.pdf -> 54
    """
    if not pdf_dir.exists():
        raise FileNotFoundError(
            f"Не найдена папка с PDF: {pdf_dir}"
        )

    pdf_files = {}

    for pdf_path in pdf_dir.glob("*.pdf"):
        match = re.match(
            r"^0*(\d+)(?:[_\s.-]|$)",
            pdf_path.stem
        )

        if not match:
            print(
                f"Предупреждение: не удалось определить номер "
                f"статьи из имени файла {pdf_path.name}"
            )
            continue

        article_number = int(match.group(1))

        if article_number in pdf_files:
            raise ValueError(
                f"Найдено несколько PDF для статьи "
                f"№{article_number}: "
                f"{pdf_files[article_number].name} и {pdf_path.name}"
            )

        pdf_files[article_number] = pdf_path

    return pdf_files


# --------------------------------------------------
# ИЗВЛЕЧЕНИЕ ТЕКСТА ИЗ PDF
# --------------------------------------------------

def normalize_text_for_abstract(text: str) -> str:
    """Нормализация с сохранением структуры строк."""
    text = text.replace("\xa0", " ")

    # Склеиваем слова, разорванные мягким дефисом.
    text = re.sub(
        r"(?<=[A-Za-zА-Яа-яЁё0-9])\u00ad[ \t]*\r?\n[ \t]*(?=[A-Za-zА-Яа-яЁё])",
        "",
        text,
    )

    text = text.replace("\u00ad", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def normalize_text(text: str) -> str:
    """Нормализация для основного текста статьи."""
    text = normalize_text_for_abstract(text)
    text = text.replace("_", "")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    return text.strip()


def extract_pdf_text(pdf_path: Path) -> dict:
    """
    Извлекает текст из PDF постранично.

    Дополнительно сохраняет:
    - количество изображений на странице;
    - количество графических объектов;
    - количество символов на странице.

    drawing_count — это количество векторных объектов,
    а не количество рисунков или формул.
    """
    pages = []
    page_info = []

    with pymupdf.open(pdf_path) as document:
        page_count = len(document)

        for page_index, page in enumerate(document):
            page_text = page.get_text("text", sort=True)
            page_text = normalize_text(page_text)

            pages.append({
                "page_number": page_index + 1,
                "text": page_text
            })

            page_info.append({
                "page_number": page_index + 1,
                "char_count": len(page_text),
                "image_count": len(page.get_images(full=True)),
                "drawing_count": len(page.get_drawings())
            })

    full_text = "\n\n".join(
        page["text"] for page in pages if page["text"]
    )

    return {
        "page_count": page_count,
        "pages": pages,
        "page_info": page_info,
        "full_text": full_text,
        "char_count": len(full_text)
    }


# --------------------------------------------------
# ОПРЕДЕЛЕНИЕ ЯЗЫКА
# --------------------------------------------------

def detect_text_language(text: str) -> tuple[str | None, float | None]:
    """
    Определяет язык фрагмента.

    Для короткого текста возвращает (None, None),
    поскольку результат для него может быть ненадёжным.
    """
    text = text.strip()

    if len(text) < MIN_TEXT_LENGTH_FOR_LANGUAGE:
        return None, None

    sample = text[:MAX_LANGUAGE_TEXT_LENGTH]

    try:
        predictions = detect_langs(sample)

        if not predictions:
            return None, None

        return predictions[0].lang, round(predictions[0].prob, 4)

    except LangDetectException:
        return None, None


def detect_article_language(text: str) -> dict:
    """
    Определяет язык статьи по извлечённому тексту.

    Возвращает:
    - language;
    - language_confidence;
    - language_candidates;
    - language_detection_status.
    """
    text = text.strip()

    if len(text) < MIN_TEXT_LENGTH_FOR_LANGUAGE:
        return {
            "language": "unknown",
            "language_confidence": None,
            "language_candidates": [],
            "language_detection_status": "insufficient_text"
        }

    sample = text[:MAX_LANGUAGE_TEXT_LENGTH]

    try:
        predictions = detect_langs(sample)

        if not predictions:
            raise LangDetectException("No language predictions", sample)

        candidates = [
            {
                "language": prediction.lang,
                "probability": round(prediction.prob, 4)
            }
            for prediction in predictions[:3]
        ]

        best_prediction = predictions[0]

        return {
            "language": best_prediction.lang,
            "language_confidence": round(best_prediction.prob, 4),
            "language_candidates": candidates,
            "language_detection_status": "success"
        }

    except LangDetectException:
        return {
            "language": "unknown",
            "language_confidence": None,
            "language_candidates": [],
            "language_detection_status": "failed"
        }


# --------------------------------------------------
# ИЗВЛЕЧЕНИЕ АННОТАЦИИ
# --------------------------------------------------

def extract_abstract_candidates(full_text: str) -> list[dict]:
    """
    Находит блоки аннотаций в начале статьи.

    Аннотация должна:
    - начинаться с отдельной строки-заголовка;
    - заканчиваться перед строкой с ключевыми словами;
    - не объединяться с другой языковой версией.
    """

    # Ограничиваем поиск начальной частью статьи.
    # Основные разделы статьи не должны попадать в поиск.
    section_pattern = re.compile(
        r"(?im)^[ \t]*(?:"
        r"введение|"
        r"1[.)]?[ \t]+введение|"
        r"introduction|"
        r"1[.)]?[ \t]+introduction"
        r")[ \t]*$"
    )

    section_match = section_pattern.search(full_text)

    search_text = (
        full_text[:section_match.start()]
        if section_match
        else full_text
    )

    headings = list(
        ABSTRACT_HEADING_PATTERN.finditer(search_text)
    )

    candidates = []

    for index, heading in enumerate(headings):
        start = heading.end()

        next_heading = (
            headings[index + 1]
            if index + 1 < len(headings)
            else None
        )

        keywords_match = KEYWORDS_PATTERN.search(
            search_text,
            start
        )

        if keywords_match and (
            next_heading is None
            or keywords_match.start() < next_heading.start()
        ):
            end = keywords_match.start()
            boundary_found = True

        elif next_heading:
            end = next_heading.start()
            boundary_found = False

        else:
            # Не берём весь остаток статьи за аннотацию.
            continue

        abstract_text = normalize_text_for_abstract(
            search_text[start:end]
        )

        if not abstract_text:
            continue

        header = heading.group("header").strip().lower()

        if header in RUSSIAN_ABSTRACT_HEADINGS:
            heading_language = "ru"
        elif header in ENGLISH_ABSTRACT_HEADINGS:
            heading_language = "en"
        else:
            heading_language = None

        detected_language, confidence = detect_text_language(
            abstract_text
        )

        candidates.append({
            "heading": header,
            "heading_language": heading_language,
            "language": detected_language,
            "confidence": confidence,
            "text": abstract_text,
            "boundary_found": boundary_found,
        })

    return candidates

def extract_article_abstract(full_text: str, article_language: str) -> dict:
    """
    Выбирает аннотацию на языке основной статьи.

    Если язык определить не удалось или границы сомнительны,
    сохраняет найденный текст со статусом needs_review.
    Другую языковую версию вместо нужной не подставляет.
    """
    candidates = extract_abstract_candidates(full_text)

    empty_result = {
        "abstract": None,
        "abstract_language": None,
        "abstract_heading": None,
        "abstract_extraction_status": "not_found",
        "abstract_candidates_count": len(candidates),
        "abstract_needs_review_reason": None,
    }

    if not candidates:
        return empty_result

    if article_language == "unknown":
        empty_result["abstract_extraction_status"] = "needs_review"
        empty_result["abstract_needs_review_reason"] = (
            "Не удалось надёжно определить основной язык статьи."
        )
        return empty_result

    # Сначала используем язык, определённый по самому тексту.
    # Заголовок служит дополнительной подсказкой, но не заменяет
    # определение языка содержимого.
    matching = [
        candidate for candidate in candidates
        if candidate["language"] == article_language
    ]

    if not matching:
        empty_result["abstract_extraction_status"] = "language_mismatch"
        empty_result["abstract_needs_review_reason"] = (
            "Не найдена аннотация, язык которой совпадает "
            "с основным языком статьи."
        )
        return empty_result

    # Если есть несколько кандидатов на нужном языке,
    # выбираем самый длинный, но обязательно помечаем на проверку.
    matching.sort(key=lambda item: len(item["text"]), reverse=True)
    selected = matching[0]

    review_reasons = []

    if len(matching) > 1:
        review_reasons.append(
            "Найдено несколько аннотаций на языке статьи."
        )

    if not selected["boundary_found"]:
        review_reasons.append(
            "Не найден заголовок «Ключевые слова» или Keywords "
            "перед следующим блоком."
        )

    if selected["confidence"] is None:
        review_reasons.append(
            "Текст аннотации слишком короткий для определения языка."
        )
    elif selected["confidence"] < 0.8:
        review_reasons.append(
            "Низкая уверенность определения языка аннотации."
        )

    if (
        selected["heading_language"] is not None
        and selected["heading_language"] != article_language
    ):
        review_reasons.append(
            "Язык заголовка не совпадает с основным языком статьи."
        )

    status = "needs_review" if review_reasons else "success"

    return {
        "abstract": selected["text"],
        "abstract_language": selected["language"],
        "abstract_heading": selected["heading"],
        "abstract_extraction_status": status,
        "abstract_candidates_count": len(candidates),
        "abstract_needs_review_reason": (
            " ".join(review_reasons) if review_reasons else None
        ),
    }

def debug_abstract_extraction(full_text: str, article_number: int):
    print(f"\n{'=' * 60}")
    print(f"СТАТЬЯ №{article_number}")
    print(f"{'=' * 60}")

    print("\n1. Первые 3000 символов текста:")
    print(repr(full_text[:3000]))

    headings = list(ABSTRACT_HEADING_PATTERN.finditer(full_text))
    print(f"\n2. Найдено заголовков: {len(headings)}")

    for heading in headings:
        print(
            f"  {heading.group('header')!r} "
            f"на позиции {heading.start()}"
        )

    keywords = list(KEYWORDS_PATTERN.finditer(full_text))
    print(f"\n3. Найдено заголовков ключевых слов: {len(keywords)}")

    for keyword in keywords:
        print(
            f"  {keyword.group(0)!r} "
            f"на позиции {keyword.start()}"
        )

    print("\n3.1. Проверка ключевых слов без регулярного выражения:")

    for line in full_text.splitlines():
        if "ключевые" in line.lower() or "keywords" in line.lower():
            print(repr(line))

    print("\n4. Первые 300 символов после нормализации:")
    print(repr(normalize_text_for_abstract(full_text)[:300]))

# --------------------------------------------------
# ОБРАБОТКА ОДНОЙ СТАТЬИ
# --------------------------------------------------

def process_article(
    article_number: int,
    metadata: dict,
    pdf_path: Path
) -> dict:
    """Извлекает текст, метаданные и аннотацию одной статьи."""
    print(f"\nОбработка статьи №{article_number}: {pdf_path.name}")

    pdf_data = extract_pdf_text(pdf_path)

    language_data = detect_article_language(pdf_data["full_text"])

    abstract_data = extract_article_abstract(
        pdf_data["full_text"],
        language_data["language"]
    )

    article_data = {
        **metadata,
        "pdf_filename": pdf_path.name,
        **pdf_data,
        **language_data,
        **abstract_data,
        "extraction_status": (
            "success" if pdf_data["full_text"] else "empty_text"
        )
    }

    return article_data


# --------------------------------------------------
# СОХРАНЕНИЕ РЕЗУЛЬТАТОВ
# --------------------------------------------------

def save_article(article_data: dict) -> Path:
    """Сохраняет статью в JSON."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    article_number = article_data["number"]
    output_path = OUTPUT_DIR / f"{article_number:04d}.json"

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            article_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    return output_path


# --------------------------------------------------
# ОСНОВНОЙ ЗАПУСК
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Извлечение текста и аннотаций из PDF (инкрементально)."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Переобработать все статьи, включая уже обработанные.",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("ПОДГОТОВКА СТАТЕЙ ДЛЯ ARTICLE SUMMARIZER AI")
    print("=" * 60)

    # 1. Читаем метаданные.
    print("\nЧтение Markdown-файла с лицензиями...")
    articles_metadata = parse_license_table(LICENSES_FILE)
    print(f"Найдено записей в таблице: {len(articles_metadata)}")

    # 2. Ищем PDF-файлы.
    print("\nПоиск PDF-файлов...")
    pdf_files = find_pdf_files(PDF_DIR)
    print(f"Найдено PDF-файлов: {len(pdf_files)}")

    # 3. Сопоставляем PDF и метаданные.
    matched_numbers = sorted(
        set(articles_metadata) & set(pdf_files)
    )

    metadata_without_pdf = sorted(
        set(articles_metadata) - set(pdf_files)
    )

    pdf_without_metadata = sorted(
        set(pdf_files) - set(articles_metadata)
    )

    print(f"\nНайдено совпадений: {len(matched_numbers)}")

    if metadata_without_pdf:
        print(
            "Для следующих записей нет PDF: "
            + ", ".join(map(str, metadata_without_pdf))
        )

    if pdf_without_metadata:
        print(
            "Для следующих PDF нет записи в таблице: "
            + ", ".join(map(str, pdf_without_metadata))
        )

    # 4. Обрабатываем статьи.
    successful = 0
    failed = 0
    skipped = 0
    abstract_statistics = {
        "success": 0,
        "needs_review": 0,
        "not_found": 0,
        "language_mismatch": 0,
    }
    report = []

    for article_number in matched_numbers:
        # Инкрементальность: пропускаем уже обработанные статьи,
        # чтобы при пополнении корпуса обрабатывались только новые PDF.
        output_path = OUTPUT_DIR / f"{article_number:04d}.json"

        if not args.force and output_path.is_file():
            skipped += 1
            print(
                f"\nСтатья №{article_number}: уже обработана, "
                f"пропускаем ({output_path.name})"
            )
            continue

        try:
            article_data = process_article(
                article_number,
                articles_metadata[article_number],
                pdf_files[article_number]
            )

            output_path = save_article(article_data)

            if article_data["extraction_status"] == "success":
                successful += 1
            else:
                failed += 1

            abstract_status = article_data["abstract_extraction_status"]
            abstract_statistics[abstract_status] += 1

            report.append({
                "number": article_number,
                "pdf_filename": pdf_files[article_number].name,
                "language": article_data["language"],
                "language_confidence": article_data["language_confidence"],
                "extraction_status": article_data["extraction_status"],
                "abstract_extraction_status": abstract_status,
                "abstract_language": article_data["abstract_language"],
                "abstract_candidates_count": (
                    article_data["abstract_candidates_count"]
                ),
                "abstract_needs_review_reason": (
                    article_data["abstract_needs_review_reason"]
                ),
                "output_file": str(output_path.relative_to(PROJECT_ROOT))
            })

            print(
                f"  Язык статьи: {article_data['language']} "
                f"(уверенность: {article_data['language_confidence']})"
            )
            print(f"  Извлечено символов: {article_data['char_count']}")
            print(
                f"  Аннотация: {abstract_status} "
                f"(язык: {article_data['abstract_language']})"
            )
            if article_data["abstract_needs_review_reason"]:
                print(
                    "  Причина проверки: "
                    f"{article_data['abstract_needs_review_reason']}"
                )
            print(f"  Сохранено: {output_path.name}")

        except Exception as error:
            failed += 1

            print(
                f"Ошибка при обработке статьи "
                f"№{article_number}: {error}"
            )

            report.append({
                "number": article_number,
                "pdf_filename": pdf_files[article_number].name,
                "extraction_status": "error",
                "error": str(error)
            })

    # 5. Формируем общий отчёт.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    report_data = {
        "total_metadata_records": len(articles_metadata),
        "total_pdf_files": len(pdf_files),
        "matched_articles": len(matched_numbers),
        "successful": successful,
        "failed": failed,
        "skipped_already_processed": skipped,
        "forced": bool(args.force),
        "abstract_statistics": abstract_statistics,
        "metadata_without_pdf": metadata_without_pdf,
        "pdf_without_metadata": pdf_without_metadata,
        "articles": report
    }

    report_path = OUTPUT_DIR / "extraction_report.json"

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(
            report_data,
            file,
            ensure_ascii=False,
            indent=4
        )

    # 6. Итог.
    print("\n" + "=" * 60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print("=" * 60)

    print(f"Всего совпадений: {len(matched_numbers)}")
    print(f"Успешно обработано: {successful}")
    print(f"Ошибок или пустого текста: {failed}")

    print("\nСтатистика извлечения аннотаций:")
    for status, count in abstract_statistics.items():
        print(f"  {status}: {count}")

    print(f"\nПропущено (уже обработаны): {skipped}")

    print(f"\nОтчёт: {report_path}")

    


if __name__ == "__main__":
    main()

