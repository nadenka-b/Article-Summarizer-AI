

"""Extractor — извлекает текст и структуру научной статьи из PDF."""

import asyncio
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from backend.agents.base import Agent
from backend.core.exceptions import ExtractionError

logger = logging.getLogger(__name__)


class ExtractorAgent(Agent):
    """Извлечение текста, метаданных и основных секций PDF."""

    name = "extractor"

    SECTION_ALIASES = {
        "abstract": [
            "abstract",
            "summary",
        ],
        "introduction": [
            "introduction",
            "background",
            "1 introduction",
        ],
        "methods": [
            "methods",
            "methodology",
            "materials and methods",
            "method",
            "experimental setup",
        ],
        "results": [
            "results",
            "results and discussion",
            "findings",
        ],
        "conclusion": [
            "conclusion",
            "conclusions",
            "discussion",
            "concluding remarks",
        ],
    }

    def __init__(self) -> None:
        super().__init__()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Принимает:
          pdf_bytes — содержимое PDF в bytes;
          pdf_path — путь к PDF-файлу.

        Возвращает:
          article_text — полный извлечённый текст;
          sections — найденные секции;
          article_metadata — метаданные PDF.
        """

        pdf_bytes = state.get("pdf_bytes")
        pdf_path = state.get("pdf_path")

        # Если текст уже был извлечён ранее — повторно PDF не обрабатываем.
        if state.get("article_text"):
            return state

        if not pdf_bytes and not pdf_path:
            raise ExtractionError(
                "Не передан PDF-файл: укажите pdf_bytes или pdf_path"
            )

        try:
            result = await asyncio.to_thread(
                self._extract_pdf,
                pdf_bytes,
                pdf_path,
            )
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                f"Не удалось обработать PDF: {e}"
            ) from e

        article_text = result["article_text"]

        if not article_text.strip():
            raise ExtractionError(
                "В PDF не найден текст. Возможно, документ состоит "
                "из сканированных изображений и требует OCR."
            )

        state["article_text"] = article_text
        state["sections"] = self._split_sections(article_text)
        state["article_metadata"] = result["metadata"]
        state["page_count"] = result["page_count"]
        state["char_count"] = len(article_text)

        logger.info(
            "Extractor: pages=%d, chars=%d, sections=%d",
            result["page_count"],
            len(article_text),
            len(state["sections"]),
        )

        return state

    @staticmethod
    def _extract_pdf(
        pdf_bytes: Optional[bytes],
        pdf_path: Optional[str],
    ) -> Dict[str, Any]:
        """Синхронное извлечение текста с помощью PyMuPDF."""

        try:
            import fitz
        except ImportError as e:
            raise ExtractionError(
                "Не установлена библиотека PyMuPDF. "
                "Установите её командой: pip install pymupdf"
            ) from e

        try:
            if pdf_bytes:
                document = fitz.open(
                    stream=pdf_bytes,
                    filetype="pdf",
                )
            else:
                document = fitz.open(str(Path(pdf_path)))

            with document:
                pages = [
                    page.get_text("text", sort=True)
                    for page in document
                ]

                full_text = "\n\n".join(pages)
                metadata = document.metadata or {}
                page_count = len(document)

        except Exception as e:
            raise ExtractionError(
                f"Ошибка чтения PDF: {e}"
            ) from e

        full_text = ExtractorAgent._clean_text(full_text)

        return {
            "article_text": full_text,
            "metadata": {
                "title": metadata.get("title"),
                "author": metadata.get("author"),
                "subject": metadata.get("subject"),
                "keywords": metadata.get("keywords"),
            },
            "page_count": page_count,
        }

    @staticmethod
    def _clean_text(text: str) -> str:
        """Удаляет лишние переносы и пробелы, исправляет переносы слов."""

        # Соединяем слова, перенесённые на следующую строку через дефис.
        text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)

        # Убираем лишние пробелы.
        text = re.sub(r"[ \t]+", " ", text)

        # Убираем слишком большое количество пустых строк.
        text = re.sub(r"\n{3,}", "\n\n", text)

        return text.strip()

    @classmethod
    def _split_sections(cls, text: str) -> Dict[str, str]:
        """
        Разделяет текст на основные секции статьи.

        Ищет распространённые английские названия разделов.
        Если заголовок не найден, соответствующая секция
        просто не добавляется.
        """

        aliases = {
            alias.lower(): section
            for section, names in cls.SECTION_ALIASES.items()
            for alias in names
        }

        sections: Dict[str, list[str]] = {}
        current_section = None

        for line in text.splitlines():
            normalized = line.strip().lower()

            # Убираем номера разделов: например, "2. Methods".
            normalized = re.sub(
                r"^\d+(?:\.\d+)*\.?\s*",
                "",
                normalized,
            )

            normalized = normalized.strip(" :.-")

            if normalized in aliases:
                current_section = aliases[normalized]
                sections.setdefault(current_section, [])
                continue

            if current_section:
                sections[current_section].append(line.strip())

        return {
            key: "\n".join(lines).strip()
            for key, lines in sections.items()
            if "\n".join(lines).strip()
        }
