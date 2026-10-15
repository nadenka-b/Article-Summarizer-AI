# backend/utils/logging.py
"""
Настройка логирования для Article-Summarizer-AI.

Особенности:
- цветной вывод в консоль;
- запись в файл с ротацией (10 МБ × 10);
- отдельный файл для ERROR;
- приглушение логов сторонних библиотек.
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path

from backend.core.config import get_settings


# ============================================================================
# Директория и имена файлов
# ============================================================================
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

_DATE = datetime.now().strftime("%Y%m%d")
LOG_FILE = LOG_DIR / f"summarizer_{_DATE}.log"
ERROR_FILE = LOG_DIR / f"summarizer_errors_{_DATE}.log"


settings = get_settings()
DEFAULT_LEVEL = logging.DEBUG if settings.DEBUG else logging.INFO


# ============================================================================
# Кастомный форматтер с цветами
# ============================================================================
class CustomFormatter(logging.Formatter):
    """Форматтер с ANSI-цветами для консоли."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    def __init__(self, fmt: str = None, datefmt: str = None, use_colors: bool = True):
        super().__init__(fmt=fmt, datefmt=datefmt)
        self.use_colors = use_colors and sys.stdout.isatty()

    def format(self, record: logging.LogRecord) -> str:
        if self.use_colors:
            levelno = record.levelno
            if levelno >= logging.CRITICAL:
                prefix = self.bold_red
            elif levelno >= logging.ERROR:
                prefix = self.red
            elif levelno >= logging.WARNING:
                prefix = self.yellow
            else:
                prefix = self.grey
            original_msg = super().format(record)
            return f"{prefix}{original_msg}{self.reset}"
        return super().format(record)


# ============================================================================
# Конфигурация
# ============================================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "()": CustomFormatter,
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "DEBUG",
            "formatter": "default",
            "stream": sys.stdout,
        },
        "file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "DEBUG",
            "formatter": "default",
            "filename": str(LOG_FILE),
            "maxBytes": 10_485_760,
            "backupCount": 10,
            "encoding": "utf-8",
        },
        "error_file": {
            "class": "logging.handlers.RotatingFileHandler",
            "level": "ERROR",
            "formatter": "default",
            "filename": str(ERROR_FILE),
            "maxBytes": 10_485_760,
            "backupCount": 5,
            "encoding": "utf-8",
        },
    },
    "loggers": {
        "sqlalchemy.engine": {"level": "WARNING", "propagate": False},
        "httpx": {"level": "WARNING", "propagate": False},
        "httpcore": {"level": "WARNING", "propagate": False},
        "urllib3": {"level": "WARNING", "propagate": False},
        "transformers": {"level": "WARNING", "propagate": False},
        "sentence_transformers": {"level": "WARNING", "propagate": False},
        "huggingface_hub": {"level": "WARNING", "propagate": False},
        "passlib": {"level": "WARNING", "propagate": False},
        "aiosqlite": {"level": "WARNING", "propagate": False},
        "watchfiles": {"level": "WARNING", "propagate": False},
        "watchfiles.main": {"level": "WARNING", "propagate": False},
    },
    "root": {
        "level": "DEBUG",
        "handlers": ["console", "file", "error_file"],
    },
}


def setup_logging(level: int = None) -> logging.Logger:
    """Инициализирует логирование."""
    if level is None:
        level = DEFAULT_LEVEL
    logging.config.dictConfig(LOGGING_CONFIG)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    return root_logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


logger = setup_logging()

__all__ = ["setup_logging", "get_logger", "logger"]