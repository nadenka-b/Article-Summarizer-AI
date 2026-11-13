# streamlit_app/pages/4_Admin.py
"""Администрирование: пользователи и роли (только admin)."""

import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import require_role

st.set_page_config(page_title="Администрирование", page_icon="👑")
client = APIClient()

if not require_role(("admin",)):
    st.stop()

token = st.session_state["token"]
st.title("👑 Администрирование")

response = client.get("/admin/users", token=token)
if response.status_code != 200:
    st.error(response.text[:500])
    st.stop()

users = response.json()
st.caption(f"Всего пользователей: {len(users)}")

for user in users:
    with st.expander(f"👤 {user['username']} — {user['role']}"):
        cols = st.columns([2, 1])
        with cols[0]:
            st.write(f"Email: {user['email']}")
            st.write(f"ID: `{user['id']}`")
            st.write(
                f"Активен: {'да' if user['is_active'] else 'нет'} | "
                f"Создан: {user['created_at'][:10]}"
            )
        with cols[1]:
            new_role = st.selectbox(
                "Роль",
                ["user", "editor", "admin"],
                index=["user", "editor", "admin"].index(user["role"]),
                key=f"role_{user['id']}",
            )
            if st.button("Применить", key=f"save_{user['id']}"):
                patch = client.patch(
                    f"/admin/users/{user['id']}/role",
                    token=token,
                    json={"role": new_role},
                )
                if patch.status_code == 200:
                    st.success(f"Роль изменена на {new_role}")
                    st.rerun()
                else:
                    st.error(patch.text[:500])
