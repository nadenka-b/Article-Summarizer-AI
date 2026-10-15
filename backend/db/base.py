# backend/db/base.py
"""
Базовый класс для всех SQLAlchemy-моделей.

Использование:
    from backend.db.base import Base

    class User(Base):
        __tablename__ = "users"
        ...
"""

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    Базовый класс для всех моделей проекта.

    SQLAlchemy 2.0 использует DeclarativeBase вместо declarative_base().
    Все модели наследуются от этого класса — так SQLAlchemy знает,
    какие таблицы нужно создать в БД.
    """
    pass