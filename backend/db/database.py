# backend/db/database.py
"""
Единая точка экспорта для БД.

Удобно: `from backend.db.database import Base, engine, get_async_session, User, ...`
"""

from backend.db.base import Base
from backend.db.session import engine, AsyncSessionLocal, get_async_session
from backend.db.models import (
    User,
    Article,
    Summary,
    KnowledgeBase,
    GlossaryDB,
    UncertainExample,
    SummaryCache,
    ReasoningLog,
    LLMCall,
)


__all__ = [
    "Base",
    "engine",
    "AsyncSessionLocal",
    "get_async_session",
    "User",
    "Article",
    "Summary",
    "KnowledgeBase",
    "GlossaryDB",
    "UncertainExample",
    "SummaryCache",
    "ReasoningLog",
    "LLMCall",
]