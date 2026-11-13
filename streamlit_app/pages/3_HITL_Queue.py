# streamlit_app/pages/3_HITL_Queue.py
"""Очередь HITL: просмотр, правка, подтверждение/отклонение (editor+)."""

import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import require_role

st.set_page_config(page_title="HITL-очередь", page_icon="🛠")
client = APIClient()

if not require_role(("editor", "admin")):
    st.stop()

token = st.session_state["token"]
st.title("🛠 Очередь экспертной проверки")

stats = client.get("/hitl/stats", token=token)
if stats.status_code == 200:
    s = stats.json()
    cols = st.columns(3)
    cols[0].metric("В очереди", s["pending"])
    cols[1].metric("Подтверждено", s["confirmed"])
    cols[2].metric("Отклонено", s["rejected"])

queue = client.get("/hitl/queue", token=token)
if queue.status_code != 200:
    st.error(queue.text[:500])
    st.stop()

items = queue.json()
if not items:
    st.success("Очередь пуста — все резюме хорошего качества 👌")
    st.stop()

for item in items:
    score = item.get("critic_score")
    header = (
        f"⏳ {item['id'][:8]} — critic: "
        f"{f'{score:.2f}' if score is not None else 'ошибка оценки'}"
    )
    with st.expander(header):
        detail = client.get(f"/hitl/queue/{item['id']}", token=token).json()

        st.markdown("**Сгенерированное резюме (требует правки):**")
        st.write(detail["model_output"])

        with st.expander("Показать текст статьи"):
            st.write(detail.get("article_text") or "(нет текста)")

        corrected = st.text_area(
            "Исправленное резюме (мин. 50 символов)",
            value=detail["model_output"],
            height=200,
            key=f"edit_{item['id']}",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Подтвердить", key=f"ok_{item['id']}",
                         type="primary", use_container_width=True):
                response = client.post(
                    f"/hitl/queue/{item['id']}/confirm",
                    token=token,
                    json={"corrected_summary": corrected},
                )
                if response.status_code == 200:
                    st.success(
                        response.json()["message"]
                        + " — переиндексация RAG запущена в фоне."
                    )
                    st.rerun()
                else:
                    st.error(response.text[:500])

        with col2:
            reason = st.text_input(
                "Причина отклонения", key=f"reason_{item['id']}"
            )
            if st.button("❌ Отклонить", key=f"no_{item['id']}",
                         use_container_width=True):
                response = client.post(
                    f"/hitl/queue/{item['id']}/reject",
                    token=token,
                    json={"reason": reason or None},
                )
                if response.status_code == 200:
                    st.info(response.json()["message"])
                    st.rerun()
                else:
                    st.error(response.text[:500])
