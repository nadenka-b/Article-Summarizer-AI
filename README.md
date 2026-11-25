# Article-Summarizer-AI — мультиагентная платформа суммаризации научных статей

## Описание
Article-Summarizer-AI — интеллектуальная система автоматического анализа и суммаризации научных публикаций. Пользователь загружает статью в PDF, указывает её идентификатор (arXiv ID) или вставляет текст. Система извлекает текст, очищает его от артефактов и генерирует краткое (abstract-like) или развёрнутое (по секциям) резюме с сохранением ключевой методологии, фактов и выводов.

Проект построен на мультиагентной архитектуре (Extractor, Summarizer, Validator, Critic, Editor + Supervisor), использует гибридный поиск по базе научных знаний (BM25 + Qdrant + RRF) для few-shot примеров, кэш результатов и контур Human-in-the-Loop (HITL) для экспертной модерации и активного обучения.

## Возможности

### По ролям пользователей

| Возможность | Гость | user | editor | admin |
| --- | :-: | :-: | :-: | :-: |
| Регистрация / вход (JWT) | ✅ | ✅ | ✅ | ✅ |
| Суммаризация: текст / PDF / arXiv ID | — | ✅ | ✅ | ✅ |
| Режимы конвейера: `agents` / `baseline` / `rag` / `extractor` | — | ✅ | ✅ | ✅ |
| Виды резюме: `abstract` / `section_wise` | — | ✅ | ✅ | ✅ |
| Фоновая суммаризация (Celery, по task_id) | — | ✅ | ✅ | ✅ |
| «Мои статьи и резюме» с метриками | — | ✅ | ✅ | ✅ |
| Поиск по базе знаний (демо гибридного RAG) | — | ✅ | ✅ | ✅ |
| Статус сервисов и статистика | — | ✅ | ✅ | ✅ |
| HITL-очередь: правка, подтверждение, отклонение | — | — | ✅ | ✅ |
| Администрирование пользователей и ролей | — | — | — | ✅ |
| Метрики качества (eval-отчёты по режимам) | — | — | — | ✅ |

### Ключевые сценарии

- **Суммаризация с контролем качества.** Полный цикл Supervisor: Summarizer → Validator (rule-based: длина, термины, плагиат) → Critic (LLM-as-a-Judge) → Editor, до 3 итераций с откатом к лучшей версии. Длинные статьи режутся на чанки по предложениям и суммаризируются иерархически.
- **RAG few-shot.** К запросу подтягиваются похожие пары «текст → аннотация» из базы знаний (гибридный BM25 + Qdrant + RRF) и подставляются в промпт как примеры.
- **Кэш.** Повторный запрос того же текста не дёргает LLM.
- **HITL / активное обучение.** Провал валидации или низкий балл Critic → резюме в очередь модерации. Эксперт правит и подтверждает → исправленное резюме становится новой эталонной парой БЗ и стартует фоновая переиндексация RAG.
- **Абляционные режимы** (`EVAL_MODE`): `baseline` (только LLM) / `rag` (+few-shot) / `agents` (полный цикл) — для честного сравнения в eval.
- **Метрики качества**: BLEU, ChrF, ROUGE-1/2/L, Coverage (+ BERTScore опционально) на эталонных аннотациях корпуса.

## Запуск проекта

### Предварительные требования
- Python 3.12 + Poetry 2.x;
- [LM Studio](https://lmstudio.ai) с моделью **Qwen2.5-7B-Instruct (Q4_K_M)** — локальный LLM-движок (или аккаунт OpenRouter — облако, выбор переменной `LLM_ENGINE`);
- (~32 ГБ RAM комфортно: LLM + эмбеддинг-модель грузятся одновременно);
- Docker Desktop — только для опций: Redis (фоновый режим) или полный стек.

### Вариант A — локальный запуск (основной, Windows)

```bash
# 1. Зависимости
poetry install
# Зависимости UI (если streamlit ещё не в окружении):
python -m poetry run pip install -r streamlit_app/requirements.txt

# 2. Конфигурация
cp .env.example .env
# В .env задать: ADMIN_EMAIL, LMSTUDIO_MODEL, JWT_SECRET_KEY

# 3. LLM-движок (LLM_ENGINE в .env):
#    a) lmstudio — локальный LM Studio: загрузить модель qwen2.5-7b-instruct
#       (Q4_K_M) и запустить сервер (GUI: Developer -> Start Server; CLI: lms
#       server start). Проверка (id = LMSTUDIO_MODEL):
curl http://localhost:1234/v1/models
#    b) openrouter — облако без локальной LLM (удобно для демо на слабом
#       железе): в .env задать LLM_ENGINE=openrouter, OPENROUTER_API_KEY и
#       OPENROUTER_MODEL=qwen/qwen-2.5-7b-instruct (та же модель, что
#       локально). Ключ не коммитить!

# 4. Первый администратор (только если БД пустая):
#    зарегистрировать пользователя с email = ADMIN_EMAIL через UI или
curl -X POST http://localhost:8000/auth/register -H "Content-Type: application/json" \
  -d '{"email":"admin@example.com","username":"admin","password":"Admin12345"}'
#    и назначить роль admin одной командой:
python -c "import sqlite3; c = sqlite3.connect('summarizer.db'); c.execute(\"UPDATE users SET role='admin' WHERE email='admin@example.com'\"); c.commit(); print('ok')"

# 5. Данные корпуса (в рабочей копии уже загружены; нужно при пустой БД):
#    ВАЖНО: import_articles требует существующего пользователя ADMIN_EMAIL
#    с ролью admin — сначала шаг 4.
python scripts/import/prepare_articles.py       # PDF -> JSON (инкрементально)
python scripts/import/import_articles.py        # статьи + БЗ + чанки (идемпотентно)
python scripts/import/add_glossary.py --mine-corpus   # глоссарий 660+
python scripts/retrieval/reindex_kb.py --local ./data/qdrant_storage  # индексы RAG

# 6. API
poetry run python run.py        # http://localhost:8000/docs

# 7. Веб-интерфейс (в отдельном терминале)
poetry run streamlit run streamlit_app/app.py --server.port 8501

# 8. (Опционально) фоновая суммаризация: Redis + Celery worker
docker compose -f infrastructure/docker-compose.yml up -d redis
poetry run celery -A backend.worker.celery_app worker --pool=solo --loglevel=info
```

### Вариант B — Docker Compose (полный стек)

```bash
docker compose -f infrastructure/docker-compose.yml up -d --build
# db (Postgres 16) + redis + qdrant + app (FastAPI) + worker (Celery) + nginx
curl http://localhost/health   # через nginx
```

Уточнения: LM Studio остаётся на хосте (контейнеры ходят на `host.docker.internal:1234`); модель в `docker-compose.yml` должна совпадать с загруженной в LM Studio; БД — Postgres в контейнере (отдельная от локального SQLite); Streamlit запускается на хосте как dev-инструмент.

## Проверка всех возможностей (чек-лист)

1. **Сервисы**: `curl http://localhost:8000/health` → `{"status":"ok","database":"healthy","llm_engine":"healthy"}`; в UI «Статистика» — все индикаторы 🟢.
2. **Регистрация и вход**: UI → вкладка «Регистрация» → новый пользователь; после входа — меню по роли. F5 не разлогинивает (сессия в `?token=`), «Выйти» в сайдбаре очищает её.
3. **Быстрая суммаризация** (Главная): текст ≥ 200 символов, режим `agents` → резюме + метрики (Critic, валидация, итерации, стратегия, HITL). Локальный LLM: секунды–минуты.
4. **Полная суммаризация** (Суммаризация): три входа — текст, PDF-файл, arXiv ID (например `2401.12345`); виды `abstract`/`section_wise`; режимы `baseline` (только LLM) и `extractor` (только извлечение текста).
5. **Кэш**: на странице «Суммаризация» прогнать один текст дважды — второй ответ приходит мгновенно (без вызова LLM; в ответе API `cached: true`).
6. **Фоновый режим** (нужны Redis + worker, шаг 8): чекбокс «Фоновый режим» → `task_id` → статусы PENDING/STARTED → SUCCESS с результатом.
7. **Мои резюме**: статьи с историей резюме, оценками Critic и числом итераций.
8. **Поиск по БЗ**: запрос («нейронные сети для обработки речи») → результаты с RRF-скорами; пары, подтверждённые через HITL, помечены `(hitl)`.
9. **Роли** (admin, «Администрирование»): повысить тестового пользователя до `editor` — у него в меню появится «HITL-очередь».
10. **HITL-контур**: поднять в `.env` `HITL_CRITIC_THRESHOLD=9.5` и перезапустить API (любая генерация уйдёт в очередь); сгенерировать резюме; под `editor` открыть «HITL-очередь»: правка → «Подтвердить» → сообщение о добавлении в БЗ (при включённом worker в фоне стартует переиндексация RAG); «Отклонить» для другой записи — в БЗ не попадает. Проверить: «Поиск по БЗ» находит подтверждённое резюме с меткой `(hitl)`.
11. **Оценка качества**: `python scripts/eval/run_eval.py --mode baseline --limit 5`, затем `--mode agents` (нужен LM Studio) → отчёты в `data/eval/`; в UI (admin) на странице «Метрики качества» — сравнение режимов таблицей и графиком.
12. **Тесты**: `poetry run pytest tests/ -q` → 171 passed.

## Архитектура

```
PDF / arXiv / текст
        │
        ▼
   Extractor ──► текст + секции
        │
        ▼
   Supervisor (цикл до 3 итераций, откат к лучшей версии)
   ├── Summarizer (direct / hierarchical-чанкинг / section_wise,
   │               few-shot примеры из RAG)
   ├── Validator  (длина, термины, плагиат — rule-based)
   ├── Critic     (LLM-as-a-Judge: полнота, точность, связность, стиль)
   └── Editor     (правки по замечаниям)
        │
        ├──► кэш (повторные запросы не дёргают LLM)
        ├──► БД: articles, article_chunks, summaries, reasoning_logs
        └──► низкое качество → HITL-очередь → эксперт правит →
             подтверждённое резюме = новый эталон БЗ (активное обучение)
```

**RAG**: база знаний (текст → аннотация) ищется гибридно — BM25 (лексика)
+ Qdrant (эмбеддинги paraphrase-multilingual-MiniLM-L12-v2) + RRF-фьюжн;
похожие пары подставляются в промпт как few-shot. Qdrant недоступен →
деградация до BM25.

**Фоновые задачи (Celery + Redis)**: `summarize` (асинхронная
суммаризация через `POST /summarize/async` + `GET /tasks/{id}`),
`reindex_knowledge_base` (запускается автоматически после HITL-confirm),
`clear_old_cache`.

## API

| Метод | Путь | Доступ | Описание |
| --- | --- | --- | --- |
| POST | `/auth/register`, `/auth/login`, `/auth/me` | все | авторизация (JWT) |
| GET | `/healthz`, `/health`, `/fields` | все | статус (БД + LM Studio) |
| POST | `/summarize/text` \| `/pdf` \| `/arxiv` | user+ | суммаризация |
| POST | `/summarize/async`, GET `/tasks/{id}` | user+ | фоновая суммаризация |
| GET | `/summaries/{id}`, `/articles/my` | владелец/админ | результаты |
| GET | `/kb/search?q=` | user+ | гибридный поиск по БЗ |
| GET/POST | `/hitl/queue`, `.../confirm`, `.../reject`, `/hitl/stats` | editor+ | контур HITL |
| GET/PATCH | `/admin/users`, `/admin/users/{id}/role` | admin | роли |

## Данные и лицензии

Источники, лицензии (преимущественно CC BY 4.0) и пайплайн пополнения
корпуса — в [data/import/README.md](data/import/README.md). Ручная
тематическая разметка — `data/import/article_categories.json`.
Сам корпус (PDF/JSON) в git не коммитится — воспроизводится из реестра
`licenses.md` скриптами импорта.

## Оценка качества

```bash
# BLEU, ChrF, ROUGE-1/2/L, Coverage (+ --use-bertscore опционально):
python scripts/eval/run_eval.py --mode baseline --limit 10
python scripts/eval/run_eval.py --mode agents --limit 10
```
Отчёты — в `data/eval/`. Режимы соответствуют EVAL_MODE (ablation:
baseline без RAG/агентов против полного agents-конвейера).

## Веб-интерфейс (Streamlit)

`poetry run streamlit run streamlit_app/app.py` — вход/регистрация,
суммаризация (текст/PDF/arXiv, синхронно или в фоне), мои резюме,
HITL-очередь редактора, администрирование ролей, статус сервисов,
поиск по базе знаний. Состав меню зависит от роли (см. таблицу выше);
сессия переживает перезагрузку страницы (токен в query params —
допустимо для локального dev-UI).

## Ограничения

- LLM — локальный LM Studio (качество и скорость зависят от железа,
  Qwen2.5-7B Q4 на CPU ≈ 30–90 с на резюме) или облачный OpenRouter
  (быстрее, но требует сети и платный);
- BM25 матчит точные словоформы (без стемминга) — семантическую
  часть закрывает векторный поиск;
- арХив-режим требует доступа в сеть; сканированные PDF без OCR не
  извлекаются;
- режим `ensemble` (EVAL_MODE) отложен — каркас eval готов;

## Стек
FastAPI · SQLAlchemy async · Celery + Redis · Qdrant + rank-bm25 (RRF) ·
sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2) ·
LM Studio (OpenAI-совместимый API) · PyMuPDF · PostgreSQL/SQLite ·
Docker Compose + Nginx · Streamlit · pytest (171 тест) · sacrebleu/ROUGE/BERTScore

## Команда проекта
- Балабанова Надежда — ML-инженерия и данные: инфраструктура репозитория, агент Extractor, корпус и глоссарий, гибридный RAG, чанкинг и иерархическая суммаризация, API `/summarize/*`, HITL и активное обучение (этапы 1, 6–10).
- Саттарова Аделина — Backend-инженерия и DevOps: каркас FastAPI, БД и Docker, JWT-аутентификация и роли, мультиагентный конвейер (Summarizer/Validator/Critic/Editor/Supervisor), Celery-фоновые задачи, Streamlit UI, метрики качества и Nginx (этапы 2–5, 11–12).

## Статус выполнения проекта
- [x] Этап 1: Выбор темы, проектирование архитектуры и настройка окружения
- [x] Этап 2: Каркас приложения и асинхронное логирование
- [x] Этап 3: Проектирование базы данных и минимальный Docker (PostgreSQL)
- [x] Этап 4: Система аутентификации и JWT-безопасность
- [x] Этап 5: Реализация базовой логики агентов
- [x] Этап 6: Скрипты сбора и импорта данных (собственный корпус PDF, инкрементально; глоссарий 660+ записей)
- [x] Этап 7: Реализация гибридного RAG (BM25 + Qdrant + RRF-фьюжн)
- [x] Этап 8: Тюнинг локального LLM-движка и стратегий суммаризации (chunking, иерархическая суммаризация)
- [x] Этап 9: Развертывание API-эндпоинтов (суммаризация текста/PDF/arXiv, кэш, HITL-очередь)
- [x] Этап 10: Реализация контура HITL и интерфейса верификации (очередь модерации, активное обучение, админ-роуты)
- [x] Этап 11: Настройка фоновых очередей задач (Celery + Redis, переиндексация RAG, очистка кэша)
- [x] Этап 12: Создание пользовательского интерфейса (Streamlit), метрики качества и деплой в Docker (Nginx)
