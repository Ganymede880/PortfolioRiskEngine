"""
Database session management for the CMCSIF portfolio tracker.

This module is responsible for:
- creating the SQLAlchemy engine
- providing a session factory
- exposing a convenient session context manager
- initializing database tables once models are defined

For MVP, the default database can be SQLite, but the structure is ready
to support PostgreSQL later via DATABASE_URL in the .env file.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

from src.config.settings import settings


# ---------------------------------------------------------------------
# SQLAlchemy base class for ORM models
# ---------------------------------------------------------------------
Base = declarative_base()


# ---------------------------------------------------------------------
# Engine setup
# ---------------------------------------------------------------------
def _build_engine():
    """
    Create and return the SQLAlchemy engine.

    Special handling:
    - SQLite needs check_same_thread=False for Streamlit-style access
    """
    database_url = settings.database_url

    if database_url.startswith("sqlite"):
        return create_engine(
            database_url,
            connect_args={"check_same_thread": False},
            future=True,
        )

    return create_engine(
        database_url,
        future=True,
    )


engine = _build_engine()


# ---------------------------------------------------------------------
# Session factory
# ---------------------------------------------------------------------
SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


# ---------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------
def get_session() -> Session:
    """
    Return a new SQLAlchemy session.

    Caller is responsible for closing it.
    """
    return SessionLocal()


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    """
    Provide a transactional session scope.

    Usage:
        with session_scope() as session:
            ...
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------------------------------------------------------------
# Database initialization
# ---------------------------------------------------------------------
def init_db() -> None:
    """
    Create all database tables registered on the Base metadata.
    """
    import src.db.models  # noqa: F401
    Base.metadata.create_all(bind=engine)