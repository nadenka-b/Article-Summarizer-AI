# streamlit_app/pages/5_Stats.py
"""Статус сервисов и статистика системы."""

import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import require_login

st.set_page_config(page_title="Статистика", page_icon="📊")
client = APIClient()

if not require_login():
    st.stop()

token = st.session_state["token"]
st.title("📊 Статус и статистика")

# ----------------------------------------------------------------------------
# Health сервисов
# ----------------------------------------------------------------------------
st.subheader("Сервисы")
health = client.get("/health", token=None)

if health.status_code == 200:
    data = health.json()
    st.success("Все сервисы здоровы")
else:
    data = health.json().get("detail", {})
    st.warning("Часть сервисов деградировала — суммаризация может не работать")

cols = st.columns(3)
db_state = data.get("database", "unknown")
llm_state = data.get("llm_engine", "unknown")
cols[0].metric(
    "База данных", "🟢" if db_state == "healthy" else "🔴",
    help=db_state,
)
cols[1].metric(
    "LLM-движок (LM Studio)", "🟢" if llm_state == "healthy" else "🔴",
    help=llm_state,
)
cols[2].metric("API", "🟢" if health.status_code else "🔴")

# ----------------------------------------------------------------------------
# Очередь HITL (если есть права)
# ----------------------------------------------------------------------------
stats = client.get("/hitl/stats", token=token)
if stats.status_code == 200:
    st.subheader("Очередь HITL")
    s = stats.json()
    cols = st.columns(3)
    cols[0].metric("В ожидании", s["pending"])
    cols[1].metric("Подтверждено", s["confirmed"])
    cols[2].metric("Отклонено", s["rejected"])

# ----------------------------------------------------------------------------
# Научные области и метрики
# ----------------------------------------------------------------------------
fields = client.get("/fields")
if fields.status_code == 200:
    data = fields.json()
    st.subheader("Научные области корпуса")
    st.write(", ".join(sorted(data["fields"].keys())))
    st.subheader("Метрики оценки качества")
    st.write(", ".join(data["eval_metrics"]))
