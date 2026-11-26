# streamlit_app/pages/2_My_Summaries.py
"""Мои статьи и резюме (пагинация + реальное число статей из API)."""

import streamlit as st

from utils.api_client import APIClient
from utils.auth_guard import require_login

st.set_page_config(page_title="Мои резюме", page_icon="📁")
client = APIClient()

if not require_login():
    st.stop()

token = st.session_state["token"]
st.title("📁 Мои статьи и резюме")

# Размер страницы пагинации (совпадает с дефолтом limit в API).
PAGE_SIZE = 50

# Загруженные статьи живут в session_state: «Показать ещё» догружает
# следующую страницу, не перезакачивая уже показанные. Ключ — токен:
# после выхода/входа другим пользователем кэш сбрасывается.
if st.session_state.get("my_articles_token") != token:
    st.session_state["my_articles"] = []
    st.session_state["my_articles_total"] = None
    st.session_state["my_articles_token"] = token


def _fetch_page(offset: int) -> bool:
    """Загружает страницу статей в session_state. False — ошибка API."""
    response = client.get(
        "/articles/my",
        token=token,
        params={"limit": PAGE_SIZE, "offset": offset},
    )
    if response.status_code != 200:
        st.error(response.text[:500])
        return False

    data = response.json()
    st.session_state["my_articles_total"] = data["total"]
    st.session_state["my_articles"].extend(data["articles"])
    return True


# Первый заход: грузим первую страницу
if not st.session_state["my_articles"]:
    if not _fetch_page(0):
        st.stop()

articles = st.session_state["my_articles"]
total = st.session_state["my_articles_total"] or 0

if total == 0:
    st.info("Пока пусто. Суммаризируйте первую статью!")
    st.stop()

# Счётчик показывает total из COUNT-запроса API, а не len(выдачи):
# раньше при >50 статьях число «замирало» на размере страницы.
caption = f"Всего статей: {total}"
if len(articles) < total:
    caption += f" (показано {len(articles)})"
st.caption(caption)

# Ручное обновление: свежесуммаризированные статьи без перезагрузки страницы
if st.button("🔄 Обновить"):
    st.session_state["my_articles"] = []
    st.rerun()

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

# Статьи за пределами загруженного — догружаем по кнопке
if len(articles) < total:
    remaining = total - len(articles)
    if st.button(f"Показать ещё (осталось {remaining})", use_container_width=True):
        if _fetch_page(len(articles)):
            st.rerun()
