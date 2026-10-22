# tests/unit/test_agents.py
"""Юнит-тесты для всех агентов."""

import pytest

from backend.agents.critic import CriticAgent
from backend.agents.editor import EditorAgent
from backend.agents.summarizer import SummarizerAgent
from backend.agents.supervisor import Supervisor
from backend.agents.validator import ValidatorAgent


# ============================================================================
# Summarizer
# ============================================================================
class TestSummarizer:
    async def test_abstract_mode(self):
        s = SummarizerAgent()
        state = {
            "article_text": "A" * 1000,
            "summary_mode": "abstract",
        }
        result = await s.run(state)
        assert result["summary"]
        assert result["summary_mode"] == "abstract"
        assert "reasoning_log" in result

    async def test_section_wise_mode(self):
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
    async def test_default_score(self):
        c = CriticAgent()
        state = {"summary": "Some summary text"}
        result = await c.run(state)
        assert result["critic_score"] == 5.0
        assert result["critic_passed"] is False  # 5 < 7

    async def test_empty_summary_fallback(self):
        c = CriticAgent()
        result = await c.run({"summary": ""})
        assert result["critic_score"] == 5.0
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
    async def test_no_issues_skips_llm(self):
        e = EditorAgent()
        state = {"summary": "Some summary " * 20}
        result = await e.run(state)
        assert result["editor_changed"] is False
        assert "Нет замечаний" in result["editor_reasoning"]

    async def test_with_issues_calls_llm(self):
        e = EditorAgent()
        state = {
            "summary": "Short",
            "validator_errors": ["Слишком короткое"],
        }
        result = await e.run(state)
        assert result["editor_changed"] is True
        assert result["summary"] != "Short"

    async def test_empty_summary_no_change(self):
        e = EditorAgent()
        result = await e.run({"summary": ""})
        assert result["editor_changed"] is False


# ============================================================================
# Supervisor
# ============================================================================
class TestSupervisor:
    async def test_full_pipeline(self):
        sup = Supervisor(max_iterations=3)
        state = {
            "article_text": "Test " * 200,
            "summary_mode": "abstract",
            "key_terms": ["test"],
        }
        result = await sup.run(state)
        assert result["supervisor_iterations"] >= 1
        assert result["supervisor_final_reason"] in {
            "passed", "max_iterations", "editor_no_change"
        }
        assert "reasoning_log" in result
        assert len(result["reasoning_log"]) >= 3  # Summarizer + Validator + Critic

    async def test_no_summary_exits_early(self):
        sup = Supervisor()
        result = await sup.run({"summary_mode": "abstract"})
        assert result["supervisor_final_reason"] == "no_summary"
        assert result["supervisor_iterations"] == 0

    async def test_is_good_enough_logic(self):
        sup = Supervisor(critic_threshold=7.0)
        assert sup._is_good_enough({"validation_passed": True, "critic_score": 8.0})
        assert not sup._is_good_enough({"validation_passed": False, "critic_score": 9.0})
        assert not sup._is_good_enough({"validation_passed": True, "critic_score": 5.0})
        assert sup._is_good_enough({"validation_passed": True, "critic_score": None})