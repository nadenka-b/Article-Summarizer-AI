# backend/worker/celery_app.py
"""
Celery-приложение для фоновых задач (этап 11).

Брокер и бэкенд результатов — Redis (REDIS_URL из .env).

Запуск воркера (Windows, один процесс — --pool=solo):
    celery -A backend.worker.celery_app worker --pool=solo --loglevel=info

Запуск в Docker:
    docker compose -f infrastructure/docker-compose.yml up -d redis worker
"""

import logging

from celery import Celery
from celery.signals import worker_shutdown

from backend.core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

celery_app = Celery(
    "article_summarizer",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    # Без include воркер не импортирует модуль с задачами — они остаются
    # незарегистрированными, и выполнение падает с "unregistered task".
    include=["backend.worker.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    broker_connection_retry_on_startup=True,
    result_expires=60 * 60 * 24,  # результаты живут сутки
    task_default_queue="default",
)


@worker_shutdown.connect
def _on_worker_shutdown(**kwargs):
    """Аккуратно закрывает персистентный event loop воркера."""
    from backend.worker.tasks import close_worker_loop

    logger.info("Worker shutting down: closing event loop...")
    close_worker_loop()
    logger.info("Worker event loop closed")
