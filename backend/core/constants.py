# backend/core/constants.py
"""Константы проекта Article-Summarizer-AI."""

from typing import Dict, List

# Научные области (аналог "языков")
RESEARCH_FIELDS: Dict[str, str] = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "physics": "Physics",
    "bio": "Biology",
    "med": "Medicine",
    "chem": "Chemistry",
    "econ": "Economics",
    "psych": "Psychology",
    "eng": "Engineering",
    "stat": "Statistics",
    "ai": "Artificial Intelligence",
    "nlp": "Natural Language Processing",
    "ml": "Machine Learning",
    "cv": "Computer Vision",
}

# Секции научной статьи
ARTICLE_SECTIONS: List[str] = [
    "abstract",
    "introduction",
    "related_work",
    "methods",
    "experiments",
    "results",
    "discussion",
    "conclusion",
    "references",
]

# Связанные научные области (для cross-domain RAG)
RELATED_FIELDS: Dict[str, List[str]] = {
    "cs": ["math", "stat"],
    "ai": ["cs", "ml", "nlp"],
    "nlp": ["ai", "cs"],
    "ml": ["ai", "cs", "stat"],
    "bio": ["chem", "med"],
    "med": ["bio", "chem"],
    "physics": ["math"],
    "econ": ["stat", "math"],
}

# Метрики оценки качества
EVAL_METRICS: List[str] = [
    "rouge_1",
    "rouge_2",
    "rouge_l",
    "bleu",
    "chrf",
    "meteor",
    "bertscore",
    "coverage_score",
]