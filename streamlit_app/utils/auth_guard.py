# streamlit_app/utils/auth_guard.py
"""Проверка авторизации и ролей для страниц Streamlit."""

import streamlit as st


def require_login() -> bool:
    """True, если пользователь авторизован; иначе рисует подсказку."""
    if st.session_state.get("token"):
        return True

    st.info("Эта страница требует входа. Обновите страницу и войдите.")
    return False


def require_role(roles: tuple) -> bool:
    """True, если роль пользователя входит в roles."""
    if not require_login():
        return False
    role = st.session_state.get("role", "user")
    if role not in roles:
        st.error(
            f"Недостаточно прав: требуется одна из ролей {roles}, "
            f"у вас — «{role}»."
        )
        return False
    return True


def logout_button():
    if st.session_state.get("token"):
        if st.sidebar.button("Выйти"):
            for key in ("token", "email", "role", "username"):
                st.session_state.pop(key, None)
            # Токен мог быть сохранён в URL (переживает F5) — чистим и его.
            st.query_params.clear()
            st.rerun()
