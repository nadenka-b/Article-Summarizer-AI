# streamlit_app/main.py
"""Главная страница (после входа): быстрая суммаризация текста.

Вызывается роутером app.py через st.navigation (страница по умолчанию).
"""

import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import logout_button

client = APIClient()

token = st.session_state["token"]
role = st.session_state.get("role", "user")

st.sidebar.success(f"Вы вошли\nРоль: **{role}**")
logout_button()

st.title("📄 Быстрая суммаризация")
st.caption(
    "Вставьте текст научной статьи — агенты сгенерируют резюме. "
    "Полные возможности (PDF, arXiv, фоновый режим) — на странице "
    "«Суммаризация» в меню слева."
)

text = st.text_area(
    "Текст статьи (минимум 200 символов)",
    height=250,
    placeholder="Вставьте сюда текст научной статьи...",
)
col1, col2 = st.columns(2)
with col1:
    mode = st.selectbox(
        "Режим",
        ["agents", "baseline", "rag"],
        help=(
            "agents — полный цикл (Summarizer → Validator → Critic → "
            "Editor); baseline — только LLM; rag — LLM + few-shot из БЗ"
        ),
    )
with col2:
    summary_mode = st.selectbox("Вид резюме", ["abstract", "section_wise"])

if st.button("Суммаризировать", type="primary", use_container_width=True):
    if len(text.strip()) < 200:
        st.warning("Текст должен быть не короче 200 символов.")
    else:
        with st.spinner(
            "Агенты работают... (локальный LLM, может занять минуты)"
        ):
            response = client.post(
                "/summarize/text",
                token=token,
                json={
                    "text": text,
                    "mode": mode,
                    "summary_mode": summary_mode,
                    "use_cache": False,
                },
            )

        if response.status_code != 200:
            st.error(f"Ошибка: {response.text[:500]}")
            st.stop()

        data = response.json()
        st.subheader("Резюме")
        st.write(data["summary"])

        cols = st.columns(5)
        cols[0].metric("Critic", data.get("critic_score") or "—")
        cols[1].metric(
            "Валидация",
            "✅" if data.get("validation_passed") else "❌",
        )
        cols[2].metric("Итерации", data.get("supervisor_iterations", 0))
        cols[3].metric("Стратегия", data.get("strategy", "—"))
        cols[4].metric(
            "HITL",
            "в очереди" if data.get("hitl_created") else "не нужен",
        )

        if data.get("hitl_created"):
            st.info(
                "Резюме отправлено в очередь модерации — эксперт "
                "доработает его на странице «HITL-очередь»."
            )
