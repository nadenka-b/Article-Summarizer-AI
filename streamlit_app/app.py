# streamlit_app/app.py
"""
Article-Summarizer-AI — роутер UI (этап 12).

До входа — форма входа/регистрации. После входа — навигация со страницами,
вид которых зависит от роли:
- user:   Главная, Суммаризация, Мои резюме, Статистика, Поиск по БЗ;
- editor: + HITL-очередь;
- admin:  + Администрирование.

Важно (Streamlit >= 1.36): st.navigation() только регистрирует страницы,
скрипт текущей страницы нужно запустить самому через .run() у возвращённой
страницы — иначе после входа будет пустой экран.

Запуск:
    pip install -r streamlit_app/requirements.txt
    streamlit run streamlit_app/app.py --server.port 8501
"""

import httpx
import streamlit as st

from utils.api_client import APIClient

st.set_page_config(
    page_title="Article-Summarizer-AI",
    page_icon="📄",
    layout="wide",
)

client = APIClient()


def _api_error(e: Exception):
    """Дружелюбное сообщение вместо трейсбека при недоступном API."""
    if isinstance(e, httpx.ConnectError):
        return (
            "❌ API недоступен. Запустите backend: poetry run python run.py "
            "(должен отвечать на http://localhost:8000)."
        )
    return f"Ошибка соединения с API: {e}"


def render_auth():
    """Форма входа/регистрации."""
    st.title("📄 Article-Summarizer-AI")
    st.markdown(
        "Мультиагентная платформа суммаризации научных статей с "
        "гибридным RAG и контуром экспертной проверки (HITL)."
    )

    tab_login, tab_register = st.tabs(["Вход", "Регистрация"])

    with tab_login:
        with st.form("login_form"):
            email = st.text_input("Email")
            password = st.text_input("Пароль", type="password")
            submitted = st.form_submit_button("Войти", use_container_width=True)

        if submitted:
            try:
                response = client.login(email, password)
            except httpx.HTTPError as e:
                st.error(_api_error(e))
            else:
                if response.status_code == 200:
                    _save_session(response.json())
                    st.rerun()
                else:
                    st.error(response.json().get("detail", "Ошибка входа"))

    with tab_register:
        with st.form("register_form"):
            email = st.text_input("Email ", key="reg_email")
            username = st.text_input("Имя пользователя (мин. 3 символа)")
            password = st.text_input(
                "Пароль (мин. 8 символов, цифра и заглавная буква)",
                type="password",
            )
            submitted = st.form_submit_button(
                "Зарегистрироваться", use_container_width=True
            )

        if submitted:
            try:
                response = client.register(email, username, password)
            except httpx.HTTPError as e:
                st.error(_api_error(e))
            else:
                if response.status_code == 201:
                    _save_session(response.json())
                    st.rerun()
                else:
                    detail = response.json().get("detail", "Ошибка регистрации")
                    if isinstance(detail, list):
                        detail = "; ".join(
                            e.get("msg", str(e)) for e in detail
                        )
                    st.error(detail)


def _save_session(token_response: dict):
    """Сохраняет токен и подтягивает роль/имя через /auth/me."""
    token = token_response["access_token"]
    st.session_state["token"] = token
    # Токен в query params переживает F5/перезагрузку вкладки: session_state
    # живёт только в рамках одной websocket-сессии. UI локальный (localhost),
    # поэтому токен в URL допустим; при выходе params очищаются.
    st.query_params["token"] = token

    try:
        response = client.get("/auth/me", token=token)
    except httpx.HTTPError:
        st.session_state["role"] = "user"
        return

    if response.status_code == 200:
        me = response.json()
        st.session_state["role"] = me.get("role", "user")
        st.session_state["email"] = me.get("email", "")
        st.session_state["username"] = me.get("username", "")
    else:
        st.session_state["role"] = "user"


def _drop_session():
    """Полностью сбрасывает сессию (session_state + query params)."""
    for key in ("token", "email", "role", "username"):
        st.session_state.pop(key, None)
    st.query_params.clear()


def _restore_session():
    """Восстанавливает сессию после перезагрузки страницы (F5).

    session_state умирает вместе с websocket-сессией, поэтому после
    перезагрузки вкладки пользователь всегда оказывался на форме входа.
    Токен берём из query params и проверяем через /auth/me.
    """
    if st.session_state.get("token"):
        return

    token = st.query_params.get("token")
    if not token:
        return

    st.session_state["token"] = token
    try:
        response = client.get("/auth/me", token=token)
    except httpx.HTTPError:
        return  # API недоступен — пусть страницы сами покажут ошибку

    if response.status_code == 200:
        me = response.json()
        st.session_state["role"] = me.get("role", "user")
        st.session_state["email"] = me.get("email", "")
        st.session_state["username"] = me.get("username", "")
    else:
        # Токен истёк или отозван — сбрасываем, показываем форму входа.
        _drop_session()


def build_navigation():
    """Навигация с учётом роли (st.navigation отключает авто-список pages/).

    Возвращает текущую страницу — её скрипт нужно запустить через .run().
    """
    role = st.session_state.get("role", "user")

    pages = [
        st.Page("main.py", title="Главная", icon="📄", default=True),
        st.Page("pages/1_Summarization.py", title="Суммаризация", icon="⚙️"),
        st.Page("pages/2_My_Summaries.py", title="Мои резюме", icon="📁"),
        st.Page("pages/6_KB_Search.py", title="Поиск по БЗ", icon="🔍"),
    ]

    if role in ("editor", "admin"):
        pages.append(
            st.Page("pages/3_HITL_Queue.py", title="HITL-очередь", icon="🛠")
        )

    pages.append(st.Page("pages/5_Stats.py", title="Статистика", icon="📊"))

    if role == "admin":
        pages.append(
            st.Page("pages/4_Admin.py", title="Администрирование", icon="👑")
        )

    return st.navigation(pages)


# ============================================================================
# Точка входа
# ============================================================================
_restore_session()

if st.session_state.get("token"):
    # При клиентской навигации между страницами query params сбрасываются —
    # восстанавливаем токен в URL, чтобы F5 на любой странице сохранял сессию.
    st.query_params["token"] = st.session_state["token"]
    build_navigation().run()
else:
    # Форма входа тоже оформлена как страница: единый паттерн st.navigation
    # и никакого авто-списка pages/ в сайдбаре до входа.
    st.navigation(
        [st.Page(render_auth, title="Вход", icon="🔒", default=True)]
    ).run()
