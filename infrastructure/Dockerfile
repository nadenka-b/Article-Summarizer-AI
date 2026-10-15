# infrastructure/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Системные зависимости
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install --no-cache-dir poetry==2.5.1 && \
    poetry config virtualenvs.create false

# Копируем файлы зависимостей
COPY pyproject.toml poetry.lock ./

# Устанавливаем зависимости (без dev-группы, без самого проекта)
RUN poetry install --no-root --only main --no-interaction

# Копируем код
COPY backend/ ./backend/
COPY run.py ./

# Порт
EXPOSE 8000

# Запуск
CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]