from collections.abc import Iterator
from functools import lru_cache

from sqlalchemy import URL, create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings, get_settings


def build_postgres_url(settings: Settings | None = None) -> str:
    resolved_settings = settings or get_settings()
    if resolved_settings.postgres_url:
        return resolved_settings.postgres_url

    return str(
        URL.create(
            drivername="postgresql+psycopg",
            username=resolved_settings.postgres_user,
            password=resolved_settings.postgres_password,
            host=resolved_settings.postgres_host,
            port=resolved_settings.postgres_port,
            database=resolved_settings.postgres_database,
        )
    )


@lru_cache
def create_postgres_engine(postgres_url: str) -> Engine:
    return create_engine(postgres_url, pool_pre_ping=True)


@lru_cache
def create_sessionmaker(postgres_url: str) -> sessionmaker[Session]:
    return sessionmaker(
        bind=create_postgres_engine(postgres_url),
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
    )


def get_postgres_engine(settings: Settings | None = None) -> Engine:
    return create_postgres_engine(build_postgres_url(settings))


def get_postgres_sessionmaker(settings: Settings | None = None) -> sessionmaker[Session]:
    return create_sessionmaker(build_postgres_url(settings))


def get_postgres_session() -> Iterator[Session]:
    session_factory = get_postgres_sessionmaker()
    with session_factory() as session:
        yield session


def create_postgres_tables(settings: Settings | None = None) -> None:
    from app.persistence.models import Base

    Base.metadata.create_all(bind=get_postgres_engine(settings))
