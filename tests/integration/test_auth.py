# tests/integration/test_auth.py
"""Интеграционные тесты для /auth/* эндпоинтов."""

import pytest
from httpx import AsyncClient


@pytest.fixture
def user_payload() -> dict:
    return {
        "email": "test@example.com",
        "username": "testuser",
        "password": "Secret123",
        "full_name": "Test User",
    }


class TestRegister:
    async def test_register_success(self, client: AsyncClient, user_payload):
        resp = await client.post("/auth/register", json=user_payload)
        assert resp.status_code == 201
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    async def test_register_duplicate_email(self, client: AsyncClient, user_payload):
        await client.post("/auth/register", json=user_payload)
        resp = await client.post("/auth/register", json=user_payload)
        assert resp.status_code == 409
        assert "email" in resp.json()["detail"].lower()

    async def test_register_duplicate_username(self, client: AsyncClient, user_payload):
        await client.post("/auth/register", json=user_payload)
        other = {**user_payload, "email": "other@example.com"}
        resp = await client.post("/auth/register", json=other)
        assert resp.status_code == 409
        assert "username" in resp.json()["detail"].lower()

    async def test_register_weak_password(self, client: AsyncClient, user_payload):
        weak = {**user_payload, "password": "weak"}
        resp = await client.post("/auth/register", json=weak)
        assert resp.status_code == 422

    async def test_register_invalid_email(self, client: AsyncClient, user_payload):
        bad = {**user_payload, "email": "not-an-email"}
        resp = await client.post("/auth/register", json=bad)
        assert resp.status_code == 422

    async def test_register_invalid_username(self, client: AsyncClient, user_payload):
        bad = {**user_payload, "username": "user with spaces"}
        resp = await client.post("/auth/register", json=bad)
        assert resp.status_code == 422


class TestLogin:
    async def test_login_success(self, client: AsyncClient, user_payload):
        await client.post("/auth/register", json=user_payload)
        resp = await client.post(
            "/auth/login",
            json={"email": user_payload["email"], "password": user_payload["password"]},
        )
        assert resp.status_code == 200
        assert "access_token" in resp.json()

    async def test_login_wrong_password(self, client: AsyncClient, user_payload):
        await client.post("/auth/register", json=user_payload)
        resp = await client.post(
            "/auth/login",
            json={"email": user_payload["email"], "password": "WrongPass1"},
        )
        assert resp.status_code == 401

    async def test_login_nonexistent_user(self, client: AsyncClient):
        resp = await client.post(
            "/auth/login",
            json={"email": "nobody@example.com", "password": "Secret123"},
        )
        assert resp.status_code == 401


class TestMe:
    async def test_me_without_token(self, client: AsyncClient):
        resp = await client.get("/auth/me")
        assert resp.status_code == 401

    async def test_me_with_valid_token(self, client: AsyncClient, user_payload):
        reg = await client.post("/auth/register", json=user_payload)
        token = reg.json()["access_token"]
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["email"] == user_payload["email"]
        assert data["username"] == user_payload["username"]
        assert data["role"] == "user"
        assert "hashed_password" not in data  # не утекло

    async def test_me_with_invalid_token(self, client: AsyncClient):
        resp = await client.get(
            "/auth/me",
            headers={"Authorization": "Bearer not-a-jwt"},
        )
        assert resp.status_code == 401