from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import hmac
from threading import Lock
from typing import Any

from fastapi import Header, HTTPException, Request, status

from app.config import Settings, load_settings
from app.db import create_postgres_pool


POSTGRES_POOL_STATE_NAME = "postgres_pool"
POSTGRES_POOL_LOCK_STATE_NAME = "postgres_pool_lock"


def initialize_postgres_pool_state(app: Any) -> None:
    setattr(app.state, POSTGRES_POOL_STATE_NAME, None)
    setattr(app.state, POSTGRES_POOL_LOCK_STATE_NAME, Lock())


def get_settings(request: Request) -> Settings:
    settings = request.app.state.settings
    if settings is None:
        settings = load_settings()
        request.app.state.settings = settings
    return settings


def get_postgres_pool(request: Request) -> Any:
    state = request.app.state
    pool = getattr(state, POSTGRES_POOL_STATE_NAME, None)
    if pool is not None:
        return pool

    lock = getattr(state, POSTGRES_POOL_LOCK_STATE_NAME, None)
    if lock is None:
        lock = Lock()
        setattr(state, POSTGRES_POOL_LOCK_STATE_NAME, lock)

    with lock:
        pool = getattr(state, POSTGRES_POOL_STATE_NAME, None)
        if pool is None:
            pool = create_postgres_pool(get_settings(request))
            setattr(state, POSTGRES_POOL_STATE_NAME, pool)
        return pool


@contextmanager
def checkout_postgres_connection(request: Request) -> Iterator[Any]:
    pool = get_postgres_pool(request)
    with pool.connection() as connection:
        yield connection


def close_postgres_pool(app: Any) -> None:
    state = app.state
    pool = getattr(state, POSTGRES_POOL_STATE_NAME, None)
    if pool is None:
        return
    try:
        pool.close()
    finally:
        setattr(state, POSTGRES_POOL_STATE_NAME, None)


def require_internal_key(
    request: Request,
    x_loop_ad_internal_key: str | None = Header(
        default=None,
        alias="X-Loop-Ad-Internal-Key",
    ),
) -> None:
    settings = get_settings(request)
    if not x_loop_ad_internal_key:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if not hmac.compare_digest(x_loop_ad_internal_key, settings.internal_api_key):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
