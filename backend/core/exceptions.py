# backend/core/exceptions.py
"""
Собственные исключения проекта Article-Summarizer-AI.

Иерархия:
    SummarizerException (базовое)
    ├── ExtractionError      — ошибка извлечения текста из PDF/API
    ├── SummarizationError   — ошибка при генерации резюме
    ├── RetrievalError       — ошибка при поиске в RAG (BM25/Qdrant)
    ├── ValidationError      — ошибка формальной проверки резюме
    ├── LLMError             — ошибка при обращении к LLM
    └── ConfigurationError   — ошибка конфигурации
"""


class SummarizerException(Exception):
    """Базовое исключение проекта. Все остальные наследуются от него."""
    pass


class ExtractionError(SummarizerException):
    """Ошибка извлечения текста из PDF или API (arXiv, PubMed и т.п.)."""
    pass


class SummarizationError(SummarizerException):
    """Ошибка генерации резюме (LLM вернула пустой результат, timeout и т.п.)."""
    pass


class RetrievalError(SummarizerException):
    """Ошибка поиска в базе знаний (BM25/Qdrant недоступны, индекс не построен)."""
    pass


class ValidationError(SummarizerException):
    """Ошибка формальной проверки резюме (мин. длина, отсутствие термина и т.п.)."""
    pass


class LLMError(SummarizerException):
    """Ошибка при обращении к LLM (Ollama, HuggingFace, внешние API)."""
    pass


class ConfigurationError(SummarizerException):
    """Ошибка конфигурации (не задан HF_TOKEN, некорректный DATABASE_URL и т.п.)."""
    pass