# tests/unit/test_chunker.py
"""Юнит-тесты чанкера (этап 8)."""

from backend.data.chunker import chunk_text, needs_chunking, split_into_sentences


def _sentence(i: int) -> str:
    return f"Предложение номер {i} содержит информацию о методах и результатах исследования."


class TestSplitSentences:
    def test_basic(self):
        sentences = split_into_sentences("Первое. Второе! Третье?")
        assert sentences == ["Первое.", "Второе!", "Третье?"]

    def test_empty(self):
        assert split_into_sentences("   ") == []


class TestChunkText:
    def test_short_text_single_chunk(self):
        chunks = chunk_text("Короткий текст.")
        assert len(chunks) == 1
        assert chunks[0]["index"] == 0
        assert chunks[0]["text"] == "Короткий текст."
        assert len(chunks[0]["text_hash"]) == 64

    def test_empty_text(self):
        assert chunk_text("") == []

    def test_long_text_multiple_chunks(self):
        text = " ".join(_sentence(i) for i in range(400))  # ~30k символов
        chunks = chunk_text(text, chunk_size=1000, overlap=100)

        assert len(chunks) > 1
        # Все индексы последовательные с нуля
        assert [c["index"] for c in chunks] == list(range(len(chunks)))

    def test_chunks_respect_size_limit(self):
        text = " ".join(_sentence(i) for i in range(300))
        chunks = chunk_text(text, chunk_size=800, overlap=100)

        # Предложение не может превышать чанк; с перекрытием —
        # чуть больше размера чанка, но не сильно
        for chunk in chunks:
            assert chunk["char_count"] <= 800 + 100 + 120

    def test_full_coverage(self):
        """Каждое предложение исходника попадает хотя бы в один чанк."""
        sentences = [_sentence(i) for i in range(150)]
        text = " ".join(sentences)
        chunks = chunk_text(text, chunk_size=1000, overlap=100)

        joined = "\n".join(c["text"] for c in chunks)
        for sentence in sentences:
            assert sentence in joined

    def test_overlap_between_chunks(self):
        """Последние предложения чанка повторяются в начале следующего."""
        text = " ".join(_sentence(i) for i in range(200))
        chunks = chunk_text(text, chunk_size=800, overlap=150)

        assert len(chunks) >= 2
        first_tail = split_into_sentences(chunks[0]["text"])[-1]
        assert first_tail in chunks[1]["text"]

    def test_giant_sentence_hard_split(self):
        """Текст без знаков конца предложения режется жёстко, без превышения."""
        text = "а" * 5000  # одно гигантское "предложение"
        chunks = chunk_text(text, chunk_size=1000, overlap=100)

        assert len(chunks) >= 5
        for chunk in chunks:
            assert chunk["char_count"] <= 1000 + 100 + 50
        # Покрытие полное
        assert "".join(c["text"] for c in chunks).replace(" ", "") == text

    def test_deterministic_hash(self):
        text = _sentence(1) + " " + _sentence(2)
        first = chunk_text(text)
        second = chunk_text(text)
        assert first[0]["text_hash"] == second[0]["text_hash"]


class TestNeedsChunking:
    def test_short(self):
        assert needs_chunking("короткий текст") is False

    def test_long(self):
        assert needs_chunking("а" * 6001) is True
