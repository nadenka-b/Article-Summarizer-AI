# run.py
"""
Скрипт запуска приложения Article-Summarizer-AI.

Использование:
    python run.py

Эквивалент:
    python -m poetry run uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
"""

import uvicorn

from backend.utils.logging import setup_logging


def main() -> None:
    """Точка входа."""
    logger = setup_logging()
    logger.info("Starting Article-Summarizer-AI via run.py")

    uvicorn.run(
        "backend.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="warning",
        reload_excludes=["logs/*", "*.log"],  # исключаем логи из watchfiles
    )


if __name__ == "__main__":
    main()