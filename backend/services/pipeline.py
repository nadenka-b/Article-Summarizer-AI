# backend/services/pipeline.py
"""
Центральный модуль суммаризации (этап 9).

perform_summarization используется роутами, а в дальнейшем —
Celery-задачами (этап 11) и eval-скриптами (этап 12), поэтому модуль
не импортирует FastAPI-зависимости.

Режимы (EVAL_MODE):
- extractor — только извлечение текста из PDF, без резюме;
- baseline — только Summarizer, без RAG и без агентов (ablation);
- rag      — Summarizer + few-shot примеры из гибридного поиска;
- agents   — полный цикл Supervisor (Summarizer → Validator → Critic
             → Editor), few-shot из RAG подключается автоматически;
- ensemble — отложен до этапа 12.
"""

import asyncio
import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import httpx
from sqlalchemy.ext.asyncio import AsyncSession

from backend.agents.extractor import ExtractorAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.supervisor import Supervisor
from backend.core.config import get_settings
from backend.core.exceptions import SummarizerException
from backend.data.cache import SummaryCacheManager
from backend.data.chunker import chunk_text
from backend.data.knowledge_base import KBManager, sha256_text
from backend.db.models import Article, ArticleChunk, ReasoningLog, Summary, User
from backend.hitl.active_learning import ActiveLearningManager
from backend.retrieval.hybrid_retriever import format_few_shot_examples, hybrid_search
from backend.retrieval.vector_index import get_vector_index

logger = logging.getLogger(__name__)
settings = get_settings()

# Не более 4 параллельных суммаризаций (защита LM Studio от перегрузки).
_summarization_semaphore = asyncio.Semaphore(4)

ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}.pdf"
ARXIV_ABS_URL = "https://arxiv.org/abs/{arxiv_id}"
# arXiv отклоняет запросы с дефолтным User-Agent httpx — идентифицируемся.
ARXIV_HEADERS = {
    "User-Agent": "Article-Summarizer-AI/0.1 (research project; local use)"
}


@dataclass
class SummarizationResult:
    """Результат работы пайплайна (передаётся в Pydantic-схему роута)."""

    article_id: Optional[str] = None
    summary_id: Optional[str] = None
    summary: str = ""
    mode: str = "agents"
    summary_mode: str = "abstract"
    strategy: str = "none"
    critic_score: Optional[float] = None
    validation_passed: Optional[bool] = None
    supervisor_iterations: int = 0
    final_reason: str = ""
    cached: bool = False
    hitl_created: bool = False
    reasoning_log: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None


# ============================================================================
# Подготовка входа
# ============================================================================
async def fetch_arxiv_pdf(arxiv_id: str) -> bytes:
    """Скачивает PDF статьи с arXiv по идентификатору."""
    url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
    async with httpx.AsyncClient(
        timeout=60.0, follow_redirects=True, headers=ARXIV_HEADERS
    ) as client:
        response = await client.get(url)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type and not response.content[:5] == b"%PDF-":
            raise SummarizerException(
                f"arXiv {arxiv_id}: по адресу {url} не вернулся PDF"
            )
        return response.content


async def _prepare_state(
    article_text: Optional[str],
    pdf_bytes: Optional[bytes],
    arxiv_id: Optional[str],
) -> Dict[str, Any]:
    """Приводит любой вход (текст / PDF / arXiv ID) к state агентов."""
    state: Dict[str, Any] = {}

    if arxiv_id:
        pdf_bytes = await fetch_arxiv_pdf(arxiv_id)

    if pdf_bytes:
        extractor = ExtractorAgent()
        state = await extractor.run({"pdf_bytes": pdf_bytes})
        if state.get("last_error"):
            raise SummarizerException(
                f"Не удалось извлечь текст: {state['last_error']}"
            )
    elif article_text:
        state["article_text"] = article_text
    else:
        raise SummarizerException(
            "Нечего суммаризировать: передайте text, файл или arxiv_id"
        )

    return state


# ============================================================================
# Few-shot из гибридного RAG
# ============================================================================
async def build_few_shot_examples(
    db: AsyncSession,
    query_text: str,
    field: Optional[str] = None,
) -> str:
    """
    Ищет похожие пары «текст → аннотация» гибридным поиском
    и форматирует их в блок few-shot примеров.

    Любая недоступность RAG (пустая БЗ, Qdrant выключен) не фатальна:
    возвращается пустая строка, суммаризация продолжается без примеров.
    """
    try:
        kb_manager = KBManager()
        await kb_manager.initialize(db)
        results = hybrid_search(
            kb_manager,
            get_vector_index(),
            query_text[:1000],
            field=field,
            top_k=3,
        )
        return format_few_shot_examples(results, max_examples=3)
    except Exception as e:
        logger.warning("Few-shot retrieval failed (продолжаем без RAG): %s", e)
        return ""


# ============================================================================
# Основная функция
# ============================================================================
async def perform_summarization(
    *,
    user: User,
    db: AsyncSession,
    mode: Optional[str] = None,
    summary_mode: Optional[str] = None,
    article_text: Optional[str] = None,
    pdf_bytes: Optional[bytes] = None,
    arxiv_id: Optional[str] = None,
    title: Optional[str] = None,
    field: Optional[str] = None,
    use_cache: bool = True,
) -> SummarizationResult:
    """Полный цикл: вход → (RAG) → агенты → БД → (HITL) → (кэш)."""
    mode = mode or settings.EVAL_MODE
    summary_mode = summary_mode or settings.SUMMARY_MODE

    if mode == "ensemble":
        raise SummarizerException(
            "Режим ensemble будет реализован на этапе 12"
        )

    result = SummarizationResult(mode=mode, summary_mode=summary_mode)

    async with _summarization_semaphore:
        state = await _prepare_state(article_text, pdf_bytes, arxiv_id)

        text = state.get("article_text") or ""
        if not text.strip():
            raise SummarizerException("Не удалось получить текст статьи")

        result.strategy = state.get("summarizer_strategy", "extracted")

        # --- Режим extractor: только извлечение ---
        if mode == "extractor":
            article = await _persist_article(
                db, user=user, title=title, field=field,
                text=text, sections=state.get("sections"),
                source="arxiv" if arxiv_id else "upload",
                source_id=arxiv_id,
            )
            result.article_id = article.id
            result.final_reason = "extractor_only"
            result.reasoning_log = state.get("reasoning_log", [])
            await _persist_reasoning_logs(db, None, result.reasoning_log)
            await db.flush()
            return result

        source_hash = sha256_text(text)

        # --- Кэш: повторные запросы не дёргают LLM ---
        cache = SummaryCacheManager()
        if use_cache:
            cached_output = await cache.get_cached(
                db, source_hash, summary_mode,
            )
            if cached_output:
                logger.info("Cache hit for %s...", source_hash[:8])
                result.summary = cached_output
                result.cached = True
                result.final_reason = "cache"
                return result

        # --- Few-shot примеры из RAG (режимы rag и agents) ---
        if mode in ("rag", "agents"):
            few_shot = await build_few_shot_examples(
                db, query_text=f"{title or ''}\n{text}", field=field
            )
            if few_shot:
                state["few_shot_examples"] = few_shot

        state["summary_mode"] = summary_mode

        # --- Прогон агентов ---
        if mode == "baseline":
            summarizer = SummarizerAgent()
            state = await summarizer.run(state)
        elif mode == "rag":
            summarizer = SummarizerAgent()
            state = await summarizer.run(state)
        else:  # agents — полный цикл
            supervisor = Supervisor()
            state = await supervisor.run(state)

        summary = state.get("summary") or ""
        if not summary:
            raise SummarizerException(
                "Не удалось сгенерировать резюме: "
                f"{state.get('summarizer_error') or 'LLM недоступна'}"
            )

        result.summary = summary
        result.strategy = state.get("summarizer_strategy", "none")
        result.critic_score = state.get("critic_score")
        result.validation_passed = state.get("validation_passed")
        result.supervisor_iterations = state.get("supervisor_iterations", 0)
        result.final_reason = state.get("supervisor_final_reason", mode)
        result.reasoning_log = state.get("reasoning_log", [])

        # --- Персистентность ---
        article = await _persist_article(
            db, user=user, title=title, field=field,
            text=text, sections=state.get("sections"),
            source="arxiv" if arxiv_id else "upload",
            source_id=arxiv_id,
        )
        result.article_id = article.id

        summary_row = Summary(
            article_id=article.id,
            summary_text=summary,
            summary_mode=summary_mode,
            critic_score=result.critic_score,
            validation_passed=result.validation_passed,
            critic_reasoning=state.get("critic_reasoning"),
            model_used=settings.LMSTUDIO_MODEL,
            engine_used="lmstudio",
            supervisor_iterations=result.supervisor_iterations,
        )
        db.add(summary_row)
        await db.flush()
        result.summary_id = summary_row.id

        await _persist_reasoning_logs(
            db, summary_row.id, result.reasoning_log
        )

        # --- HITL: плохое качество → очередь модерации ---
        if settings.ACTIVE_LEARNING_ENABLED and _needs_human_review(state):
            await ActiveLearningManager.add_uncertain_example(
                db,
                summary_id=summary_row.id,
                model_output=summary,
                critic_score=result.critic_score,
                validation_passed=result.validation_passed,
            )
            result.hitl_created = True

        # --- Кэш ---
        if use_cache:
            await cache.put(
                db, source_hash, summary_mode, summary,
                engine_used="lmstudio", model_used=settings.LMSTUDIO_MODEL,
            )

        await db.flush()
        return result


def _needs_human_review(state: Dict[str, Any]) -> bool:
    """Низкое качество → очередь HITL (по теме проекта)."""
    if not state.get("validation_passed", False):
        return True
    if state.get("critic_failed", False):
        return True
    score = state.get("critic_score")
    return score is not None and float(score) < settings.HITL_CRITIC_THRESHOLD


async def _persist_article(
    db: AsyncSession,
    *,
    user: User,
    title: Optional[str],
    field: Optional[str],
    text: str,
    sections: Optional[Dict[str, str]],
    source: str,
    source_id: Optional[str],
) -> Article:
    """Создаёт статью пользователя + её чанки."""
    article = Article(
        owner_id=user.id,
        title=(title or f"Статья от {user.username}")[:500],
        source=source,
        source_id=source_id,
        field=field,
        language="ru",
        full_text=text,
        sections=json.dumps(sections, ensure_ascii=False) if sections else None,
        char_count=len(text),
        status="summarized",
    )
    db.add(article)
    await db.flush()

    for chunk in chunk_text(text):
        db.add(ArticleChunk(
            article_id=article.id,
            chunk_index=chunk["index"],
            chunk_text=chunk["text"],
            char_count=chunk["char_count"],
            token_count=len(chunk["text"]) // 4,
            text_hash=chunk["text_hash"],
        ))

    return article


async def _persist_reasoning_logs(
    db: AsyncSession,
    summary_id: Optional[str],
    entries: List[Dict[str, Any]],
) -> None:
    """Сохраняет историю работы агентов в reasoning_logs."""
    if not entries:
        return

    request_id = str(uuid.uuid4())
    for entry in entries:
        db.add(ReasoningLog(
            request_id=request_id,
            agent_name=entry.get("agent", "unknown"),
            input_summary=None,
            output_summary=entry.get("error"),
            status=entry.get("status", "success"),
        ))


async def safe_perform_summarization(**kwargs) -> SummarizationResult:
    """Обёртка: исключение превращается в результат с error, не в 500."""
    try:
        return await perform_summarization(**kwargs)
    except SummarizerException as e:
        logger.warning("Summarization failed: %s", e)
        return SummarizationResult(error=str(e))
    except Exception as e:
        logger.exception("Unexpected summarization failure")
        return SummarizationResult(error=f"{type(e).__name__}: {e}")
