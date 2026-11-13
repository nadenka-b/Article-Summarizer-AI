"""
Оценка качества суммаризации на корпусе (этап 12).

Сравнивает сгенерированные резюме с эталонными аннотациями статей
корпуса по метрикам: BLEU, ChrF (sacrebleu), ROUGE-1/2/L (своя
реализация), Coverage (доля ключевых терминов статьи в резюме) и
опционально BERTScore.

Работает напрямую с агентами (без записи в БД): данные берутся из
data/processed/*.json, где есть и полный текст, и эталонная аннотация.

Запуск (нужен запущенный LM Studio):
    python scripts/eval/run_eval.py --mode baseline --limit 5
    python scripts/eval/run_eval.py --mode agents --limit 5 --use-bertscore
"""

import argparse
import asyncio
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.retrieval.bm25_index import STOPWORDS  # переиспользуем стоп-слова

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROCESSED_DIR.parent / "eval"

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]{2,}")


# ============================================================================
# Метрики (без тяжёлых зависимостей; BERTScore — опционально)
# ============================================================================
def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in STOPWORDS]


def rouge_n(hypothesis: str, reference: str, n: int) -> float:
    """ROUGE-N (F1) по n-граммам."""
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)

    hyp_ngrams = Counter(tuple(hyp[i:i + n]) for i in range(len(hyp) - n + 1))
    ref_ngrams = Counter(tuple(ref[i:i + n]) for i in range(len(ref) - n + 1))

    overlap = sum((hyp_ngrams & ref_ngrams).values())
    hyp_total = sum(hyp_ngrams.values())
    ref_total = sum(ref_ngrams.values())

    if hyp_total == 0 or ref_total == 0 or overlap == 0:
        return 0.0

    precision = overlap / hyp_total
    recall = overlap / ref_total
    return 2 * precision * recall / (precision + recall)


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L (F1) по самой длинной общей подпоследовательности."""
    hyp = _tokens(hypothesis)
    ref = _tokens(reference)

    if not hyp or not ref:
        return 0.0

    # Динамика по LCS
    previous = [0] * (len(ref) + 1)
    for i in range(1, len(hyp) + 1):
        current = [0] * (len(ref) + 1)
        for j in range(1, len(ref) + 1):
            if hyp[i - 1] == ref[j - 1]:
                current[j] = previous[j - 1] + 1
            else:
                current[j] = max(previous[j], current[j - 1])
        previous = current

    lcs = previous[len(ref)]
    precision = lcs / len(hyp)
    recall = lcs / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def coverage_score(hypothesis: str, source: str, top_terms: int = 20) -> float:
    """
    Coverage Score (по теме): доля ключевых терминов статьи,
    попавших в резюме.
    """
    source_tokens = _tokens(source)
    term_counts = Counter(source_tokens)
    key_terms = [term for term, _ in term_counts.most_common(top_terms)]

    if not key_terms:
        return 0.0

    summary_tokens = set(_tokens(hypothesis))
    covered = sum(1 for term in key_terms if term in summary_tokens)
    return covered / len(key_terms)


def corpus_bleu_chrf(hypotheses: list[str], references: list[str]) -> dict:
    """BLEU и ChrF через sacrebleu (корпусные метрики)."""
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
    chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
    return {"bleu": round(bleu, 2), "chrf": round(chrf, 2)}


def bertscore_metrics(hypotheses: list[str], references: list[str]) -> dict:
    """BERTScore (опционально, медленно на CPU)."""
    from bert_score import score

    P, R, F1 = score(hypotheses, references, lang="ru", verbose=False)
    return {
        "bertscore_p": round(float(statistics.mean(P)), 4),
        "bertscore_r": round(float(statistics.mean(R)), 4),
        "bertscore_f1": round(float(statistics.mean(F1)), 4),
    }


# ============================================================================
# Прогон агентов без записи в БД
# ============================================================================
def load_eval_pairs(limit: int | None) -> list[dict]:
    """Пары «текст → эталонная аннотация» из обработанного корпуса."""
    pairs = []
    for path in sorted(PROCESSED_DIR.glob("[0-9]" * 4 + ".json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        abstract = data.get("abstract")
        text = data.get("full_text")

        if (
            isinstance(abstract, str) and abstract.strip()
            and isinstance(text, str) and text.strip()
            and data.get("abstract_extraction_status") == "success"
        ):
            pairs.append({
                "number": data["number"],
                "title": data.get("title", ""),
                "text": text,
                "reference": abstract,
            })

        if limit and len(pairs) >= limit:
            break

    return pairs


async def generate_summaries(pairs: list[dict], mode: str) -> list[str]:
    """Генерирует резюме агентами (без персистентности)."""
    from backend.agents.summarizer import SummarizerAgent
    from backend.agents.supervisor import Supervisor

    summaries = []
    for i, pair in enumerate(pairs, start=1):
        print(f"[{i}/{len(pairs)}] {pair['title'][:60]}...")
        state = {
            "article_text": pair["text"],
            "summary_mode": "abstract",
        }

        if mode in ("baseline", "rag"):
            # rag-режим на этой оценке эквивалентен baseline:
            # few-shot из БЗ подключается в API-пайплайне.
            state = await SummarizerAgent().run(state)
        elif mode == "agents":
            state = await Supervisor(max_iterations=2).run(state)
        else:
            raise ValueError(f"Неизвестный режим: {mode}")

        summaries.append(state.get("summary") or "")
    return summaries


# ============================================================================
# Основной прогон
# ============================================================================
async def run(args: argparse.Namespace) -> None:
    pairs = load_eval_pairs(args.limit)
    if not pairs:
        print("Нет пар для оценки — запустите prepare_articles.py.")
        return

    print(f"Пар для оценки: {len(pairs)} | режим: {args.mode}")

    hypotheses = await generate_summaries(pairs, args.mode)
    references = [pair["reference"] for pair in pairs]

    # Пустые резюме = LLM не отвечал; нулевые метрики будут неверией.
    empty = [i + 1 for i, h in enumerate(hypotheses) if not h.strip()]
    if empty:
        print(
            f"\nВНИМАНИЕ: {len(empty)} из {len(pairs)} резюме пустые — "
            "похоже, LM Studio не отвечал."
        )
        if len(empty) == len(pairs):
            print(
                "Все резюме пустые: запускайте LM Studio "
                "(сервер на localhost:1234) и повторите."
            )
            return

    print("\nСчитаю метрики...")
    rouge1 = [rouge_n(h, r, 1) for h, r in zip(hypotheses, references)]
    rouge2 = [rouge_n(h, r, 2) for h, r in zip(hypotheses, references)]
    rougeL = [rouge_l(h, r) for h, r in zip(hypotheses, references)]
    coverage = [
        coverage_score(h, pair["text"]) for h, pair in zip(hypotheses, pairs)
    ]

    metrics = {
        "n": len(pairs),
        "mode": args.mode,
        "rouge_1": round(statistics.mean(rouge1), 4),
        "rouge_2": round(statistics.mean(rouge2), 4),
        "rouge_l": round(statistics.mean(rougeL), 4),
        "coverage": round(statistics.mean(coverage), 4),
    }
    metrics.update(corpus_bleu_chrf(hypotheses, references))

    if args.use_bertscore:
        metrics.update(bertscore_metrics(hypotheses, references))

    # Отчёт
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUTPUT_DIR / f"eval_{args.mode}_{len(pairs)}.json"
    report_path.write_text(
        json.dumps(
            {
                "metrics": metrics,
                "per_article": [
                    {
                        "number": pair["number"],
                        "title": pair["title"],
                        "rouge_1": round(r1, 4),
                        "rouge_l": round(rl, 4),
                        "coverage": round(cov, 4),
                        "hypothesis": h[:500],
                        "reference": pair["reference"][:500],
                    }
                    for pair, h, r1, rl, cov in zip(
                        pairs, hypotheses, rouge1, rougeL, coverage
                    )
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n===== РЕЗУЛЬТАТЫ =====")
    for key, value in metrics.items():
        print(f"  {key}: {value}")
    print(f"\nОтчёт: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка качества суммаризации")
    parser.add_argument(
        "--mode", choices=["baseline", "rag", "agents"], default="baseline"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Оценить только N статей"
    )
    parser.add_argument(
        "--use-bertscore", action="store_true", help="Добавить BERTScore (медленно)"
    )
    args = parser.parse_args()

    asyncio.run(run(args))


if __name__ == "__main__":
    main()
