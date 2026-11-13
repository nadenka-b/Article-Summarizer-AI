# tests/unit/test_rrf.py
"""Юнит-тесты Reciprocal Rank Fusion."""

from backend.retrieval.hybrid_retriever import reciprocal_rank_fusion


def _entry(entry_id: str):
    class FakeEntry:
        def __init__(self, id):
            self.id = id

    return FakeEntry(entry_id)


class TestRRF:
    def test_formula_sums_reciprocal_ranks(self):
        # a: ранг 0 в первом списке, ранг 1 во втором
        # score(a) = 1/(60+0+1) + 1/(60+1+1)
        a, b, c = _entry("a"), _entry("b"), _entry("c")

        first = [(a, 9.0), (b, 8.0)]
        second = [(c, 0.9), (a, 0.8)]

        results = reciprocal_rank_fusion([first, second], k=60, top_k=3)

        scores = {e.id: s for e, s in results}
        assert abs(scores["a"] - (1 / 61 + 1 / 62)) < 1e-9
        assert abs(scores["b"] - 1 / 62) < 1e-9
        assert abs(scores["c"] - 1 / 61) < 1e-9

    def test_entry_present_in_both_lists_ranks_first(self):
        a = _entry("a")
        only = _entry("only")

        first = [(a, 1.0), (only, 0.9)]
        second = [(a, 0.5)]

        results = reciprocal_rank_fusion([first, second], top_k=2)
        assert results[0][0].id == "a"

    def test_dedup_by_key(self):
        # Одна запись в двух списках → один результат, не два
        a1, a2 = _entry("a"), _entry("a")
        results = reciprocal_rank_fusion([[(a1, 1.0)], [(a2, 1.0)]], top_k=5)
        assert len(results) == 1

    def test_entry_from_first_source(self):
        # При дедупликации объект берётся из первого источника
        a1, a2 = _entry("a"), _entry("a")
        results = reciprocal_rank_fusion([[(a1, 1.0)], [(a2, 1.0)]])
        assert results[0][0] is a1

    def test_top_k_limit(self):
        entries = [(_entry(f"e{i}"), 1.0) for i in range(10)]
        results = reciprocal_rank_fusion([entries], top_k=3)
        assert len(results) == 3

    def test_empty_rankings(self):
        assert reciprocal_rank_fusion([], top_k=5) == []
        assert reciprocal_rank_fusion([[], []], top_k=5) == []
