# streamlit_app/pages/2_My_Summaries.py
"""Мои статьи и резюме."""

import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import require_login

st.set_page_config(page_title="Мои резюме", page_icon="📁")
client = APIClient()

if not require_login():
    st.stop()

token = st.session_state["token"]
st.title("📁 Мои статьи и резюме")

response = client.get("/articles/my", token=token)
if response.status_code != 200:
    st.error(response.text[:500])
    st.stop()

articles = response.json()
if not articles:
    st.info("Пока пусто. Суммаризируйте первую статью!")
    st.stop()

st.caption(f"Всего статей: {len(articles)}")

for article in articles:
    summaries = article.get("summaries", [])
    head = (
        f"📄 {article['title']} — {len(summaries)} резюме"
        if summaries
        else f"📄 {article['title']} (без резюме)"
    )
    with st.expander(head):
        cols = st.columns(4)
        cols[0].write(f"Источник: `{article['source']}`")
        cols[1].write(f"Символов: {article.get('char_count') or '—'}")
        cols[2].write(f"Статус: {article['status']}")
        cols[3].write(f"Создано: {article['created_at'][:16].replace('T', ' ')}")

        for summary in summaries:
            score = summary.get("critic_score")
            score_str = f"{score:.2f}" if score is not None else "—"
            st.markdown(
                f"**Резюме** ({summary['summary_mode']}, critic: "
                f"{score_str}, итераций: "
                f"{summary.get('supervisor_iterations', 0)}, "
                f"`{summary['summary_id'][:8]}`)"
            )
            st.write(summary["summary_text"])
            st.divider()
