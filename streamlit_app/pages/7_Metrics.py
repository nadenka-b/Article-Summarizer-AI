# streamlit_app/pages/7_Metrics.py
"""Метрики качества суммаризации (только admin).

Показывает отчёты scripts/eval/run_eval.py (BLEU, ChrF, ROUGE-1/2/L,
Coverage, опц. BERTScore) с разбивкой по режимам baseline / rag / agents —
сравнение показывает вклад RAG-примеров и агентного контура качества.
"""

import streamlit as st
import pandas as pd

from utils.api_client import APIClient
from utils.auth_guard import require_role

st.set_page_config(page_title="Метрики качества", page_icon="🎯")
client = APIClient()

if not require_role(("admin",)):
    st.stop()

token = st.session_state["token"]
st.title("🎯 Метрики качества")
st.caption(
    "Абляционное сравнение режимов конвейера на эталонных аннотациях "
    "корпуса: baseline (только LLM) → rag (+ few-shot из БЗ) → "
    "agents (полный цикл Validator/Critic/Editor)."
)

response = client.get("/admin/metrics", token=token)
if response.status_code != 200:
    st.error(response.text[:500])
    st.stop()

payload = response.json()
reports = payload.get("reports", [])

with st.expander("📖 Что означают метрики"):
    st.markdown(
        "- **ROUGE-1 / ROUGE-2 / ROUGE-L** — совпадение уни-/биграмм и "
        "наибольшая общая подпоследовательность с эталонной аннотацией "
        "(0–1; стандарт суммаризации).\n"
        "- **BLEU** — корпусное n-граммное совпадение, sacrebleu (0–100).\n"
        "- **ChrF** — символьная F-мера, устойчива к длинной технической "
        "лексике (0–100).\n"
        "- **Coverage** — доля ключевых терминов статьи, попавших в резюме "
        "(0–1; дополнительная метрика из ТЗ).\n"
        "- **BERTScore** (если включали `--use-bertscore`) — семантическая "
        "близость через эмбеддинги."
    )

if not reports:
    st.info(payload.get("hint") or "Отчётов пока нет.")
    st.markdown(
        "Сгенерировать: `python scripts/eval/run_eval.py --mode baseline "
        "--limit 5` (затем `--mode rag`, `--mode agents`) и обновить страницу."
    )
    st.stop()

# Свежий отчёт каждого режима (по дате файла)
latest: dict = {}
for rep in reports:
    if rep["mode"] not in latest or rep["mtime"] > latest[rep["mode"]]["mtime"]:
        latest[rep["mode"]] = rep

COLUMNS = [
    ("n", "пар"),
    ("bleu", "BLEU"),
    ("chrf", "ChrF"),
    ("rouge_1", "ROUGE-1"),
    ("rouge_2", "ROUGE-2"),
    ("rouge_l", "ROUGE-L"),
    ("coverage", "Coverage"),
    ("bertscore_f1", "BERTScore F1"),
]
df = pd.DataFrame(
    [
        {
            "Режим": mode,
            **{label: rep["metrics"].get(key) for key, label in COLUMNS},
        }
        for mode, rep in sorted(latest.items())
    ]
)
st.subheader("Сравнение режимов (свежий прогон каждого)")
st.dataframe(df.set_index("Режим"), use_container_width=True)
st.caption(
    "Внимание на шкалы: BLEU и ChrF — 0–100 (sacrebleu), "
    "ROUGE и Coverage — 0–1."
)

st.subheader("График по метрике")
metric_label = st.selectbox(
    "Метрика",
    [label for _, label in COLUMNS if label != "пар"],
)
chart_df = df.set_index("Режим")[[metric_label]]
st.bar_chart(chart_df)
st.caption("Сравнивайте метрики одного типа между режимами — рост от "
           "baseline → rag → agents показывает вклад каждого компонента.")

# Детали по режимам
st.subheader("Детали прогонов")
for mode, rep in sorted(latest.items()):
    with st.expander(
        f"{mode} — {rep['file']} ({len(rep.get('per_article', []))} статей, "
        f"обновлён {rep['mtime'].replace('T', ' ')})"
    ):
        per_article = rep.get("per_article", [])
        if per_article:
            st.dataframe(
                pd.DataFrame(per_article)[
                    ["number", "title", "rouge_1", "rouge_l", "coverage"]
                ].rename(columns={
                    "number": "№",
                    "title": "Статья",
                    "rouge_1": "ROUGE-1",
                    "rouge_l": "ROUGE-L",
                    "coverage": "Coverage",
                }),
                use_container_width=True,
            )
        else:
            st.write("Постатейной детализации в отчёте нет.")

# История всех прогонов
with st.expander("🗂 История прогонов"):
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Файл": rep["file"],
                    "Режим": rep["mode"],
                    "пар": rep["metrics"].get("n"),
                    "BLEU": rep["metrics"].get("bleu"),
                    "ROUGE-1": rep["metrics"].get("rouge_1"),
                    "Обновлён": rep["mtime"].replace("T", " "),
                }
                for rep in sorted(reports, key=lambda r: r["mtime"], reverse=True)
            ]
        ),
        use_container_width=True,
    )
