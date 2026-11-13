# tests/unit/test_agents.py
"""Юнит-тесты для всех агентов.

Реальный LLM не вызывается: LLMClient подменяется заглушкой
FakeLLMClient (методичка этапа 5 — на этом этапе LLM работает за заглушкой).
"""

import json

import pytest

from backend.agents.critic import CriticAgent
from backend.agents.editor import EditorAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.supervisor import Supervisor
from backend.agents.validator import ValidatorAgent


# ============================================================================
# Заглушка LLM
# ============================================================================
class FakeLLMClient:
    """Заглушка LLMClient: без сети, ответ задаётся в тесте.

    default — текст ответа для любого промпта;
    handler — callable(prompt) -> str, если нужны разные ответы разным агентам.
    """

    def __init__(self, default: str = ""):
        self.default = default
        self.handler = None
        self.calls = 0
        self.prompts = []

    async def generate(
        self, prompt, model=None, temperature=0.3, task="generate"
    ):
        self.calls += 1
        self.prompts.append(prompt)
        text = self.handler(prompt) if self.handler is not None else self.default
        return text, "mock", "mock-model"


@pytest.fixture
def fake_llm(monkeypatch):
    """Подменяет singleton-клиент LLM на заглушку."""
    client = FakeLLMClient()
    monkeypatch.setattr(
        "backend.utils.llm_client.get_llm_client", lambda: client
    )
    return client


def critic_json(score: float) -> str:
    """Ответ Critic-агента: одинаковый балл по всем критериям."""
    return json.dumps(
        {
            "completeness": score,
            "accuracy": score,
            "coherence": score,
            "style": score,
            "reasoning": "ok",
            "suggestions": ["добавить детали методов"],
            "missing_terms": ["метод"],
        }
    )


# Текст-заглушка резюме: 64 слова (лимит Validator — 50), содержит термин "test",
# не совпадает дословно со статьёй "Test ... Test".
SUMMARY_TEXT = (
    "This test summary describes a novel scientific method "
    "for article summarization and reports improved quality metrics. "
) * 4


# ============================================================================
# Summarizer
# ============================================================================
class TestSummarizer:
    async def test_abstract_mode(self, fake_llm):
        fake_llm.default = SUMMARY_TEXT
        s = SummarizerAgent()
        state = {
            "article_text": "A" * 1000,
            "summary_mode": "abstract",
        }
        result = await s.run(state)
        assert result["summary"]
        assert result["summary_mode"] == "abstract"
        assert "reasoning_log" in result

    async def test_section_wise_mode(self, fake_llm):
        fake_llm.default = "Резюме секции статьи. " * 10
        s = SummarizerAgent()
        state = {
            "sections": {
                "abstract": "Abstract text " * 20,
                "methods": "Methods text " * 20,
            },
            "summary_mode": "section_wise",
        }
        result = await s.run(state)
        assert "section_summaries" in result
        assert "abstract" in result["section_summaries"]

    async def test_empty_input(self):
        s = SummarizerAgent()
        result = await s.run({"summary_mode": "abstract"})
        assert result["summary"] == ""
        assert "summarizer_error" in result


# ============================================================================
# Validator
# ============================================================================
class TestValidator:
    async def test_short_summary_fails(self):
        v = ValidatorAgent()
        state = {"summary": "short text"}
        result = await v.run(state)
        assert result["validation_passed"] is False
        assert any("короткое" in e for e in result["validator_errors"])

    async def test_good_summary_passes(self):
        v = ValidatorAgent()
        summary = " ".join(["word"] * 100)  # 100 слов
        state = {
            "summary": summary,
            "key_terms": ["word"],
        }
        result = await v.run(state)
        assert result["validation_passed"] is True

    async def test_plagiarism_detected(self):
        v = ValidatorAgent()
        long_sentence = (
            "This is a very long sentence that has more than fifteen words "
            "and could be identified as plagiarism if it appears in the source."
        )
        # Делаем summary из 5 таких предложений + filler
        summary = " ".join([long_sentence] * 5) + " " + " ".join(["filler"] * 50)
        state = {
            "summary": summary,
            "article_text": "Some preamble. " + long_sentence + " Some following text.",
        }
        result = await v.run(state)
        assert result["validation_passed"] is False
        assert any("копий" in e for e in result["validator_errors"])

    async def test_empty_summary_fails(self):
        v = ValidatorAgent()
        result = await v.run({"summary": ""})
        assert result["validation_passed"] is False


# ============================================================================
# Critic
# ============================================================================
class TestCritic:
    async def test_fallback_when_response_not_json(self, fake_llm):
        """LLM ответила не-JSON → оценки нет: score=None, critic_failed=True."""
        fake_llm.default = "не JSON вообще"
        c = CriticAgent()
        state = {"summary": "Some summary text"}
        result = await c.run(state)
        assert result["critic_score"] is None
        assert result["critic_failed"] is True
        assert result["critic_passed"] is False
        assert "Fallback" in result["critic_reasoning"]

    async def test_llm_error_fallback(self, fake_llm):
        """LLM недоступна (исчерпаны retry) → fallback, а не фиктивный балл."""
        async def broken_generate(*args, **kwargs):
            raise RuntimeError("LM Studio недоступен")

        fake_llm.generate = broken_generate
        c = CriticAgent()
        result = await c.run({"summary": "Some summary text"})
        assert result["critic_score"] is None
        assert result["critic_failed"] is True

    async def test_score_from_valid_json(self, fake_llm):
        fake_llm.default = json.dumps(
            {
                "completeness": 9, "accuracy": 8, "coherence": 9, "style": 8,
                "reasoning": "точно передаёт суть",
                "suggestions": [],
                "missing_terms": [],
            }
        )
        c = CriticAgent()
        result = await c.run({"summary": "Some summary text"})
        assert result["critic_score"] == 8.5
        assert result["critic_failed"] is False
        assert result["critic_passed"] is True  # 8.5 >= 7

    async def test_empty_summary_fallback(self):
        c = CriticAgent()
        result = await c.run({"summary": ""})
        assert result["critic_score"] is None
        assert result["critic_failed"] is True
        assert "Fallback" in result["critic_reasoning"]

    def test_parse_plain_json(self):
        c = CriticAgent()
        raw = '{"completeness": 8, "accuracy": 9, "coherence": 7, "style": 8}'
        parsed = c._parse_json(raw)
        assert parsed is not None
        assert parsed["completeness"] == 8

    def test_parse_markdown_json(self):
        c = CriticAgent()
        raw = 'Here you go:\n```json\n{"completeness": 8, "accuracy": 9}\n```'
        parsed = c._parse_json(raw)
        assert parsed is not None
        assert parsed["accuracy"]

    def test_parse_json_with_text_around(self):
        c = CriticAgent()
        raw = 'Вот результат: {"completeness": 7, "accuracy": 8, "coherence": 7, "style": 8} Готово.'
        parsed = c._parse_json(raw)
        assert parsed is not None
        assert parsed["completeness"] == 7

    def test_parse_invalid_returns_none(self):
        c = CriticAgent()
        assert c._parse_json("not a json at all") is None
        assert c._parse_json("") is None

    def test_safe_float_clamps(self):
        c = CriticAgent()
        assert c._f(15) == 10.0
        assert c._f(-5) == 0.0
        assert c._f("not a number") == 5.0


# ============================================================================
# Editor
# ============================================================================
class TestEditor:
    async def test_no_issues_skips_llm(self, fake_llm):
        e = EditorAgent()
        state = {"summary": "Some summary " * 20}
        result = await e.run(state)
        assert result["editor_changed"] is False
        assert "Нет замечаний" in result["editor_reasoning"]
        assert fake_llm.calls == 0

    async def test_critic_failed_without_issues_skips_llm(self, fake_llm):
        """Критик не оценил и конкретных замечаний нет → LLM не дёргаем."""
        e = EditorAgent()
        state = {"summary": "Some summary " * 20, "critic_failed": True}
        result = await e.run(state)
        assert result["editor_changed"] is False
        assert fake_llm.calls == 0

    async def test_with_issues_calls_llm(self, fake_llm):
        fake_llm.default = (
            "Полностью переписанное резюме статьи с методами, "
            "результатами и выводами исследования. " * 5
        )
        e = EditorAgent()
        state = {
            "summary": "Short",
            "validator_errors": ["Слишком короткое"],
        }
        result = await e.run(state)
        assert result["editor_changed"] is True
        assert result["summary"] != "Short"
        assert fake_llm.calls == 1

    async def test_empty_summary_no_change(self):
        e = EditorAgent()
        result = await e.run({"summary": ""})
        assert result["editor_changed"] is False


# ============================================================================
# Supervisor
# ============================================================================
class TestSupervisor:
    async def test_full_pipeline(self, fake_llm):
        def handler(prompt: str) -> str:
            if "Оцени резюме" in prompt:  # промпт Critic
                return critic_json(9.0)
            return SUMMARY_TEXT  # Summarizer

        fake_llm.handler = handler
        sup = Supervisor(max_iterations=3)
        state = {
            "article_text": "Test " * 200,
            "summary_mode": "abstract",
            "key_terms": ["test"],
        }
        result = await sup.run(state)
        assert result["supervisor_iterations"] == 1
        assert result["supervisor_final_reason"] == "passed"
        assert result["supervisor_restored_best"] is False
        assert "reasoning_log" in result
        assert len(result["reasoning_log"]) >= 3  # Summarizer + Validator + Critic

    async def test_no_summary_exits_early(self):
        sup = Supervisor()
        result = await sup.run({"summary_mode": "abstract"})
        assert result["supervisor_final_reason"] == "no_summary"
        assert result["supervisor_iterations"] == 0
        assert result["supervisor_restored_best"] is False

    async def test_restores_best_iteration(self, fake_llm):
        """Если после Editor стало хуже — Supervisor откатывает к лучшей версии."""
        critic_scores = [6.5, 5.0]  # итерация 1, затем после Editor — хуже
        good_text = "Хорошее резюме научной статьи с методами и результатами. " * 12
        worse_text = "Плохое резюме без деталей и без результатов работы. " * 12

        def handler(prompt: str) -> str:
            if "Оцени резюме" in prompt:  # промпт Critic
                return critic_json(critic_scores.pop(0))
            if "НОВОЕ РЕЗЮМЕ" in prompt:  # промпт Editor
                return worse_text
            return good_text  # Summarizer

        fake_llm.handler = handler
        sup = Supervisor(max_iterations=2)
        state = {
            "article_text": "Научная статья. " * 100,
            "summary_mode": "abstract",
            "key_terms": [],
        }
        result = await sup.run(state)
        assert result["supervisor_final_reason"] == "max_iterations"
        assert result["supervisor_restored_best"] is True
        assert result["critic_score"] == 6.5
        assert result["summary"] == good_text.strip()

    def test_is_good_enough_logic(self):
        sup = Supervisor(critic_threshold=7.0)
        assert sup._is_good_enough({"validation_passed": True, "critic_score": 8.0})
        assert not sup._is_good_enough({"validation_passed": False, "critic_score": 9.0})
        assert not sup._is_good_enough({"validation_passed": True, "critic_score": 5.0})
        # Критик не оценил (score=None или critic_failed) — проходить нельзя
        assert not sup._is_good_enough({"validation_passed": True, "critic_score": None})
        assert not sup._is_good_enough(
            {"validation_passed": True, "critic_score": 9.0, "critic_failed": True}
        )

    def test_quality_requires_validation_and_score(self):
        assert Supervisor._quality({"validation_passed": True, "critic_score": 8.0}) == 8.0
        assert Supervisor._quality({"validation_passed": False, "critic_score": 8.0}) is None
        assert Supervisor._quality({"validation_passed": True, "critic_score": None}) is None
        assert Supervisor._quality(
            {"validation_passed": True, "critic_score": 8.0, "critic_failed": True}
        ) is None

    def test_snapshot_restore_keeps_reasoning_log(self):
        state = {
            "summary": "new",
            "section_summaries": {"abstract": "b"},
            "critic_score": 5.0,
            "reasoning_log": [{"agent": "x"}],
        }
        snap = Supervisor._snapshot(state)
        state["summary"] = "worse"
        state["reasoning_log"].append({"agent": "y"})

        Supervisor._restore(state, snap)

        assert state["summary"] == "new"
        assert state["critic_score"] == 5.0
        # reasoning_log — история, он не откатывается
        assert len(state["reasoning_log"]) == 2
