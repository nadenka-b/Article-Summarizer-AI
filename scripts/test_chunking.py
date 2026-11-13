import asyncio
import re
import hashlib

from sentence_transformers import SentenceTransformer
from sqlalchemy import select, delete

from backend.db.models import Article, ArticleChunk
from backend.db.session import AsyncSessionLocal


# ============================================================
# Настройки
# ============================================================

MODEL_NAME = "BAAI/bge-m3"

CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

PREVIEW_LENGTH = 500


# ============================================================
# Загрузка модели и токенизатора
# ============================================================

print(f"Загружаем модель {MODEL_NAME}...")

model = SentenceTransformer(MODEL_NAME)

tokenizer = model[0].tokenizer

print("Модель загружена.")
print("Максимальная длина:", model.max_seq_length)
print("Размер эмбеддинга:", model.get_embedding_dimension())


# ============================================================
# Подсчёт токенов
# ============================================================

def count_tokens(text: str) -> int:
    """Считает количество токенов без специальных токенов."""
    return len(
        tokenizer.encode(
            text,
            add_special_tokens=False
        )
    )


# ============================================================
# Разделение текста на предложения
# ============================================================

def split_into_sentences(text: str) -> list[str]:
    """
    Сначала выделяет абзацы, затем предложения.
    Сохраняет исходный порядок текста.
    """

    paragraphs = re.split(r"\n\s*\n", text)

    sentences = []

    for paragraph in paragraphs:
        paragraph = paragraph.strip()

        if not paragraph:
            continue

        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-ZА-ЯЁ0-9])",
            paragraph
        )

        sentences.extend(
            part.strip()
            for part in parts
            if part.strip()
        )

    return sentences


# ============================================================
# Создание чанков
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP
) -> list[str]:

    sentences = split_into_sentences(text)

    chunks = []

    current_sentences = []
    current_tokens = 0

    for sentence in sentences:

        sentence_tokens = count_tokens(sentence)

        if sentence_tokens == 0:
            continue

        # Если одно предложение длиннее чанка,
        # разделяем его непосредственно по токенам.
        if sentence_tokens > chunk_size:

            if current_sentences:
                chunks.append(" ".join(current_sentences))

            current_sentences = []
            current_tokens = 0

            token_ids = tokenizer.encode(
                sentence,
                add_special_tokens=False
            )

            step = chunk_size - overlap

            for start in range(0, len(token_ids), step):

                part_ids = token_ids[start:start + chunk_size]

                part = tokenizer.decode(
                    part_ids,
                    skip_special_tokens=True
                ).strip()

                if part:
                    chunks.append(part)

                if start + chunk_size >= len(token_ids):
                    break

            continue

        # Если предложение не помещается,
        # завершаем текущий чанк.
        if current_tokens + sentence_tokens > chunk_size:

            chunks.append(" ".join(current_sentences))

            # Переносим последние предложения
            # в следующий чанк для перекрытия.
            overlap_sentences = []
            overlap_tokens = 0

            for old_sentence in reversed(current_sentences):

                old_tokens = count_tokens(old_sentence)

                if overlap_tokens + old_tokens > overlap:
                    break

                overlap_sentences.insert(0, old_sentence)
                overlap_tokens += old_tokens

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

            # Если перекрытие мешает добавить предложение,
            # убираем старые предложения.
            while (
                current_sentences
                and current_tokens + sentence_tokens > chunk_size
            ):
                removed = current_sentences.pop(0)
                current_tokens -= count_tokens(removed)

        current_sentences.append(sentence)
        current_tokens += sentence_tokens

    # Добавляем последний чанк.
    if current_sentences:
        chunks.append(" ".join(current_sentences))

    return chunks


# ============================================================
# Получение всех статей из SQLite
# ============================================================

async def get_test_articles():
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            select(Article).where(
                Article.source == "corpus",
                Article.full_text.is_not(None),
                Article.full_text != ""
            )
        )
        articles = list(result.scalars().all())

    articles.sort(
        key=lambda article: len(article.full_text or "")
    )

    if not articles:
        raise ValueError("В базе данных не найдено статей для обработки.")

    print(f"\nНайдено статей для обработки: {len(articles)}")
    return articles

# ============================================================
# Сохранение чанков в SQLite
# ============================================================

async def save_chunks(article_chunks):
    """
    Сохраняет чанки в article_chunks.

    При повторном запуске старые чанки выбранных статей
    удаляются и заменяются новыми.

    Удаление и вставка выполняются в одной транзакции.
    """

    async with AsyncSessionLocal() as session:

        async with session.begin():

            article_ids = [
                article_id
                for article_id, chunks in article_chunks
            ]

            # Удаляем предыдущие чанки тестируемых статей.
            await session.execute(
                delete(ArticleChunk).where(
                    ArticleChunk.article_id.in_(article_ids)
                )
            )

            new_chunks = []

            for article_id, chunks in article_chunks:

                for chunk_index, chunk_text in enumerate(chunks):

                    new_chunks.append(
                        ArticleChunk(
                            article_id=article_id,
                            chunk_index=chunk_index,
                            chunk_text=chunk_text,
                            char_count=len(chunk_text),
                            token_count=count_tokens(chunk_text),
                            text_hash=hashlib.sha256(
                                chunk_text.encode("utf-8")
                            ).hexdigest()
                        )
                    )

            session.add_all(new_chunks)

    return len(new_chunks)

# ============================================================
# Тестирование
# ============================================================

async def main():

    articles = await get_test_articles()

    print("\n" + "=" * 70)
    print("РЕЗУЛЬТАТЫ ТЕСТОВОГО ЧАНКИНГА")
    print("=" * 70)

    total_chunks = 0

    # Здесь временно храним чанки до записи в БД.
    article_chunks = []

    for article in articles:

        text = article.full_text or ""

        chunks = create_chunks(text)

        token_counts = [
            count_tokens(chunk)
            for chunk in chunks
        ]

        total_chunks += len(chunks)

        print("\n" + "-" * 70)
        print("Статья:", article.title)
        print("ID:", article.id)
        print("Источник:", article.source_id)
        print("Символов:", len(text))
        print("Токенов:", count_tokens(text))
        print("Количество чанков:", len(chunks))

        if token_counts:
            print("Минимум токенов в чанке:", min(token_counts))
            print("Максимум токенов в чанке:", max(token_counts))
            print(
                "Среднее количество токенов:",
                round(sum(token_counts) / len(token_counts), 1)
            )

        print("\nПЕРВЫЙ ЧАНК:")
        if chunks:
            print(chunks[0][:PREVIEW_LENGTH])

        print("\nПОСЛЕДНИЙ ЧАНК:")
        if chunks:
            print(chunks[-1][:PREVIEW_LENGTH])

        oversized = [
            i
            for i, count in enumerate(token_counts, start=1)
            if count > CHUNK_SIZE
        ]

        print("\nПроверка ограничения:")

        if oversized:
            print("Обнаружены чанки больше заданного размера:", oversized)
        else:
            print("Все чанки укладываются в заданный размер.")

        # Добавляем результаты для последующего сохранения.
        article_chunks.append((article.id, chunks))

    # Сохраняем результаты в БД.
    print("\n" + "=" * 70)
    print("СОХРАНЕНИЕ ЧАНКОВ В БАЗУ ДАННЫХ")
    print("=" * 70)

    saved_count = await save_chunks(article_chunks)

    print("Сохранено чанков:", saved_count)

    print("\n" + "=" * 70)
    print("ИТОГО")
    print("Проверено статей:", len(articles))
    print("Всего получено чанков:", total_chunks)
    print("Всего сохранено чанков:", saved_count)
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())

