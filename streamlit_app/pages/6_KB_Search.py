# streamlit_app/pages/6_KB_Search.py
"""Поиск по базе знаний — демо гибридного RAG (BM25 + Qdrant + RRF)."""

import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import require_login

st.set_page_config(page_title="Поиск по БЗ", page_icon="🔍")
client = APIClient()

if not require_login():
    st.stop()

token = st.session_state["token"]
st.title("🔍 Поиск по базе знаний")
st.caption(
    "Гибридный поиск: BM25 (лексика) + Qdrant (семантика) + RRF-фьюжн. "
    "Именно так система подбирает few-shot примеры для генерации."
)

query = st.text_input(
    "Запрос", placeholder="например: нейронные сети для обработки речи"
)
top_k = st.slider("Результатов", 1, 10, 5)

if st.button("Искать", type="primary") and len(query.strip()) >= 3:
    response = client.get(
        "/kb/search", token=token,
        params={"q": query, "top_k": top_k},
    )
    if response.status_code != 200:
        st.error(response.text[:500])
        st.stop()

    results = response.json()["results"]
    if not results:
        st.info("Ничего не найдено.")
        st.stop()

    for result in results:
        with st.expander(
            f"📎 [{result['score']:.4f}] {result['field']}/{result['language']}"
            + (" (hitl)" if result.get("source") == "hitl" else "")
        ):
            st.markdown("**Эталонная аннотация:**")
            st.write(result["target_summary"])
            st.markdown("**Начало исходного текста:**")
            st.caption(result["source_excerpt"] + "…")
