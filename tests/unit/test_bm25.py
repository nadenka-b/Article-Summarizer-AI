# tests/unit/test_bm25.py
"""Юнит-тесты BM25-индекса и токенизации."""

from backend.retrieval.bm25_index import BM25Index, tokenize


class TestTokenize:
    def test_russian_tokens(self):
        assert tokenize("Машинное обучение на больших данных") == [
            "машинное", "обучение", "больших", "данных"
        ]

    def test_english_tokens(self):
        assert tokenize("Neural networks for text") == [
            "neural", "networks", "text"
        ]

    def test_mixed_case_and_digits(self):
        tokens = tokenize("BERT-модель 2023 года")
        assert "bert" in tokens
        assert "модель" in tokens
        assert "2023" in tokens

    def test_stopwords_removed(self):
        tokens = tokenize("и в не на the of the")
        assert tokens == []

    def test_short_tokens_removed(self):
        assert tokenize("а я о к") == []


class TestBM25Index:
    def test_search_finds_relevant_document(self):
        docs = [
            "Машинное обучение применяется для классификации текстов",
            "Квантовая механика описывает поведение микрочастиц",
            "Нейронные сети обучаются на размеченных данных",
        ]
        index = BM25Index()
        index.initialize(docs)

        results = index.search("обучение нейронных сетей", top_k=2)
        assert len(results) >= 1
        texts = [doc for doc, _ in results]
        # Релевантные документы должны быть выше нерелевантного
        assert texts[0] in (docs[0], docs[2])
        assert docs[1] not in texts[:2]

    def test_only_positive_scores(self):
        docs = ["альфа бета гамма", "дельта эпсилон"]
        index = BM25Index()
        index.initialize(docs)
        results = index.search("альфа", top_k=5)
        assert all(score > 0 for _, score in results)

    def test_no_match_returns_empty(self):
        index = BM25Index()
        index.initialize(["альфа бета"])
        assert index.search("zzz xxx yyy") == []

    def test_empty_index_returns_empty(self):
        index = BM25Index()
        index.initialize([])
        assert index.search("что угодно") == []

    def test_len(self):
        index = BM25Index()
        index.initialize(["a b", "c d"])
        assert len(index) == 2
