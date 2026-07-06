from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends
from fastapi import FastAPI

from app.analysis.router import router as analysis_router
from app.config import Settings, load_settings
from app.decision.router import (
    ad_experiment_router,
    promotion_run_router,
    router as decision_router,
)
from app.dependencies import (
    close_postgres_pool,
    initialize_postgres_pool_state,
    require_internal_key,
)
from app.generation.router import router as generation_router
from app.internal.router import router as internal_batch_router


def create_app(*, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _get_or_load_settings(app)
        try:
            yield
        finally:
            close_postgres_pool(app)

    app = FastAPI(
        title="Loop-Ad Decision API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    initialize_postgres_pool_state(app)

    @app.get("/health", tags=["system"])
    def health() -> dict[str, str]:
        resolved_settings = _get_or_load_settings(app)
        return {
            "status": "ok",
            "service": resolved_settings.service_id,
            "env": resolved_settings.env,
        }

    @app.get(
        "/internal/health",
        tags=["internal"],
        dependencies=[Depends(require_internal_key)],
    )
    def internal_health() -> dict[str, str]:
        resolved_settings = _get_or_load_settings(app)
        return {
            "status": "ok",
            "service": resolved_settings.service_id,
        }

    app.include_router(analysis_router)
    app.include_router(generation_router)
    app.include_router(decision_router)
    app.include_router(promotion_run_router)
    app.include_router(ad_experiment_router)
    app.include_router(internal_batch_router)
    return app


def _get_or_load_settings(app: Any) -> Settings:
    settings = app.state.settings
    if settings is None:
        settings = load_settings()
        app.state.settings = settings
    return settings


app = create_app()
