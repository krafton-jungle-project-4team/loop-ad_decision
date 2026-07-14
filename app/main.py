from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends
from fastapi import FastAPI

from app.analysis.router import router as analysis_router
from app.config import Settings, load_settings
from app.db import (
    create_generation_coordinator_connection,
    create_postgres_connection,
)
from app.decision.router import (
    ad_experiment_router,
    promotion_run_router,
    router as decision_router,
)
from app.dependencies import require_internal_key
from app.generation.router import router as generation_router
from app.generation.coordinator import GenerationCoordinator
from app.generation.worker import GenerationJobProcessor
from app.internal.router import router as internal_batch_router
from app.logging import configure_logging, request_logging_middleware


def create_app(*, settings: Settings | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        resolved_settings = _get_or_load_settings(app)
        configure_logging(resolved_settings)
        coordinator: GenerationCoordinator | None = None
        if resolved_settings.env != "test":
            coordinator = GenerationCoordinator(
                settings=resolved_settings,
                connection_factory=create_generation_coordinator_connection,
                processor_factory=lambda: GenerationJobProcessor(
                    settings=resolved_settings,
                    connection_factory=create_postgres_connection,
                ),
            )
            app.state.generation_coordinator = coordinator
            coordinator.start()
        try:
            yield
        finally:
            if coordinator is not None:
                coordinator.shutdown()
            app.state.generation_coordinator = None

    app = FastAPI(
        title="Loop-Ad Decision API",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.generation_coordinator = None
    if settings is not None:
        configure_logging(settings)

    app.middleware("http")(request_logging_middleware)

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
