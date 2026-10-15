# tests/unit/test_security.py
"""Юнит-тесты для backend.utils.security."""

from datetime import timedelta

import pytest
from jose import JWTError

from backend.utils.security import (
    create_access_token,
    decode_token,
    get_password_hash,
    verify_password,
)


class TestPasswordHashing:
    """Хеширование пароля и проверка."""

    def test_hash_is_not_plaintext(self):
        h = get_password_hash("Secret123")
        assert h != "Secret123"
        assert h.startswith("$2b$")  # bcrypt-префикс

    def test_verify_correct_password(self):
        h = get_password_hash("Secret123")
        assert verify_password("Secret123", h) is True

    def test_verify_wrong_password(self):
        h = get_password_hash("Secret123")
        assert verify_password("WrongPassword", h) is False

    def test_verify_broken_hash_returns_false(self):
        # Не падаем — возвращаем False
        assert verify_password("Secret123", "not-a-valid-hash") is False

    def test_same_password_different_hashes(self):
        """bcrypt добавляет соль — хеши разные."""
        h1 = get_password_hash("Secret123")
        h2 = get_password_hash("Secret123")
        assert h1 != h2
        # Но оба валидны
        assert verify_password("Secret123", h1)
        assert verify_password("Secret123", h2)

    def test_long_password_does_not_crash(self):
        """Пароль > 72 байт не должен падать."""
        long_pw = "A1" + "x" * 200
        h = get_password_hash(long_pw)
        assert verify_password(long_pw, h) is True


class TestJWT:
    """Создание и валидация токенов."""

    def test_encode_decode_roundtrip(self):
        token = create_access_token({"sub": "user-1", "role": "user"})
        payload = decode_token(token)
        assert payload["sub"] == "user-1"
        assert payload["role"] == "user"
        assert "exp" in payload
        assert "iat" in payload

    def test_expired_token_raises(self):
        token = create_access_token(
            {"sub": "user-1"},
            expires_delta=timedelta(seconds=-1),
        )
        with pytest.raises(JWTError):
            decode_token(token)

    def test_garbage_token_raises(self):
        with pytest.raises(JWTError):
            decode_token("not-a-jwt")