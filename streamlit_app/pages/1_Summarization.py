# streamlit_app/pages/1_Summarization.py
"""Полная суммаризация: текст / PDF / arXiv, синхронно или в фоне."""

import re
import time

import httpx
import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import require_login

st.set_page_config(page_title="Суммаризация", page_icon="⚙️")
client = APIClient()


# ============================================================================
# Хелперы (определены ДО кнопок: иначе NameError при клике)
# ============================================================================
def render_result(data: dict):
    # Режим extractor: резюме нет, показываем извлечённый текст
    if data.get("extracted_text") is not None:
        st.subheader("Извлечённый текст")
        st.caption(
            f"Статья сохранена (id: {data.get('article_id', '—')}). "
            "Режим extractor не создаёт резюме."
        )
        with st.expander("Показать текст", expanded=True):
            st.text(data["extracted_text"])
        return

    st.subheader("Резюме")
    st.write(data["summary"])

    cols = st.columns(5)
    cols[0].metric("Critic", data.get("critic_score") or "—")
    cols[1].metric(
        "Валидация", "✅" if data.get("validation_passed") else "❌"
    )
    cols[2].metric("Итерации", data.get("supervisor_iterations", 0))
    cols[3].metric("Стратегия", data.get("strategy", "—"))
    cols[4].metric(
        "HITL", "в очереди" if data.get("hitl_created") else "не нужен"
    )

    with st.expander("История работы агентов"):
        for entry in data.get("reasoning_log", []):
            status_icon = "✅" if entry.get("status") == "success" else "❌"
            st.markdown(
                f"{status_icon} **{entry.get('agent')}** — "
                f"{entry.get('duration_sec', 0):.2f}с"
            )


def _run(request: dict, background: bool = False, path: str = None):
    """Вызывает API суммаризации; background=True — через Celery-очередь."""
    files = request.get("files")
    json_body = request.get("json")
    text_body = request.get("text")
    path = path or request.get("path", "/summarize/text")
    extra = request.get("params", {})
    merged = {**params, **extra}

    try:
        if background and text_body:
            response = client.post(
                "/summarize/async", token=token,
                json={"text": text_body, **merged},
            )
            if response.status_code != 202:
                st.error(response.text[:500])
                return
            _poll_task(response.json()["task_id"])
            return

        if files:
            response = client.post(
                path, token=token, files=files,
                params={k: v for k, v in merged.items() if v},
            )
        else:
            body = json_body or {"text": text_body, **merged}
            response = client.post(path, token=token, json=body)
    except httpx.HTTPError as e:
        st.error(f"Сетевая ошибка: {e}")
        return

    if response.status_code != 200:
        st.error(f"Ошибка {response.status_code}: {response.text[:500]}")
        return

    render_result(response.json())


def _poll_task(task_id: str):
    st.info(f"Задача в очереди: `{task_id}`")
    placeholder = st.empty()

    for _ in range(600):  # до ~10 минут
        try:
            response = client.get(f"/tasks/{task_id}", token=token)
        except httpx.HTTPError as e:
            placeholder.error(f"Сетевая ошибка: {e}")
            return

        if response.status_code != 200:
            placeholder.error(response.text[:500])
            return

        data = response.json()
        if data["status"] == "SUCCESS":
            placeholder.empty()
            result = data.get("result", {})
            if result.get("error"):
                st.error(result["error"])
            else:
                render_result(result)
            return
        if data["status"] == "FAILURE":
            placeholder.error(f"Задача упала: {data.get('error')}")
            return

        placeholder.write(f"Статус: **{data['status']}** ...")
        time.sleep(1)

    st.warning("Превышено время ожидания задачи.")


# ============================================================================
# Страница
# ============================================================================
if not require_login():
    st.stop()

token = st.session_state["token"]
st.title("⚙️ Суммаризация статьи")

source_tab, pdf_tab, arxiv_tab = st.tabs(
    ["Текст", "PDF-файл", "arXiv ID"]
)

params = {}
with st.sidebar:
    params["mode"] = st.selectbox(
        "Режим конвейера",
        ["agents", "baseline", "rag", "extractor"],
        help="agents — полный цикл агентов; extractor — только извлечь текст",
    )
    params["summary_mode"] = st.selectbox(
        "Вид резюме", ["abstract", "section_wise"]
    )
    params["title"] = st.text_input("Название (опционально)") or None
    background = st.checkbox(
        "Фоновый режим (Celery)",
        help="Задача уйдёт в очередь — можно закрыть страницу и вернуться "
        "по task_id. Требует запущенных Redis и worker.",
    )

# ----------------------------------------------------------------------------
# Текст
# ----------------------------------------------------------------------------
with source_tab:
    text = st.text_area("Текст статьи", height=250)
    if st.button("Суммаризировать текст", type="primary"):
        if len(text.strip()) < 200:
            st.warning("Минимум 200 символов.")
        else:
            with st.spinner("Агенты работают... (может занять минуты)"):
                _run({"text": text}, background=background)

# ----------------------------------------------------------------------------
# PDF
# ----------------------------------------------------------------------------
with pdf_tab:
    uploaded = st.file_uploader("PDF-файл статьи", type=["pdf"])
    if st.button("Суммаризировать PDF", type="primary"):
        if uploaded is None:
            st.warning("Выберите файл.")
        else:
            with st.spinner("Извлекаем текст и суммаризируем..."):
                _run(
                    {
                        "files": {
                            "file": (
                                uploaded.name,
                                uploaded.getvalue(),
                                "application/pdf",
                            )
                        },
                        "path": "/summarize/pdf",
                    }
                )

# ----------------------------------------------------------------------------
# arXiv
# ----------------------------------------------------------------------------
with arxiv_tab:
    arxiv_id = st.text_input(
        "arXiv ID", placeholder="например, 2401.12345"
    )
    if st.button("Скачать и суммаризировать", type="primary"):
        if not re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", arxiv_id.strip()):
            st.warning("Некорректный arXiv ID.")
        else:
            with st.spinner("Скачиваем статью с arXiv..."):
                _run(
                    {
                        "json": {"arxiv_id": arxiv_id.strip()},
                        "path": "/summarize/arxiv",
                    }
                )
