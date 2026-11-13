# streamlit_app/utils/api_client.py
"""HTTP-клиент к API Article-Summarizer-AI для Streamlit UI."""

import httpx


DEFAULT_API_URL = "http://localhost:8000"


class APIClient:
    """Тонкая обёртка над httpx с токеном авторизации."""

    def __init__(self, base_url: str = None):
        self.base_url = (base_url or DEFAULT_API_URL).rstrip("/")

    # ------------------------------------------------------------------
    # Авторизация (возвращают httpx.Response — единый контракт с get/post)
    # ------------------------------------------------------------------
    def register(self, email: str, username: str, password: str) -> httpx.Response:
        return self._request(
            "POST", "/auth/register",
            json={"email": email, "username": username, "password": password},
        )

    def login(self, email: str, password: str) -> httpx.Response:
        return self._request(
            "POST", "/auth/login",
            json={"email": email, "password": password},
        )

    # ------------------------------------------------------------------
    # Универсальные методы (токен передаётся явно — он живёт в session_state)
    # ------------------------------------------------------------------
    def get(self, path: str, token: str = None, params: dict = None) -> httpx.Response:
        return self._request("GET", path, token=token, params=params)

    def post(self, path: str, token: str = None, json: dict = None,
             files=None, data=None, params=None) -> httpx.Response:
        return self._request(
            "POST", path, token=token, json=json,
            files=files, data=data, params=params,
        )

    def patch(self, path: str, token: str = None, json: dict = None) -> httpx.Response:
        return self._request("PATCH", path, token=token, json=json)

    def _request(self, method: str, path: str, token: str = None, **kwargs) -> httpx.Response:
        headers = kwargs.pop("headers", {})
        if token:
            headers["Authorization"] = f"Bearer {token}"

        timeout = kwargs.pop("timeout", 900.0)  # суммаризация бывает долгой
        with httpx.Client(timeout=timeout) as client:
            return client.request(
                method, f"{self.base_url}{path}", headers=headers, **kwargs
            )
