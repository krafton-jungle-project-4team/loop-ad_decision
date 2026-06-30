from __future__ import annotations

import hmac
from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import date
from typing import Literal, Protocol

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request, status
from pydantic import BaseModel

from app.config import Settings, load_settings
from app.jobs.decision_job import (
    DecisionRunHandle,
    DecisionRunRequest,
    ProjectNotFoundError,
    build_daily_decision_job_service,
)


class JobService(Protocol):
    def start_run(self, request: DecisionRunRequest) -> DecisionRunHandle:
        ...

    def execute_run(self, run_id: int) -> object:
        ...


class RunDailyDecisionRequest(BaseModel):
    project_key: str
    analysis_date: date
    mode: Literal["normal", "demo", "backfill"] = "normal"
    force: bool = False


class RunDailyDecisionResponse(BaseModel):
    run_id: int
    project_key: str
    analysis_date: date
    status: str
    message: str


def create_app(
    *,
    settings: Settings | None = None,
    job_service_factory: Callable[[], JobService] | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        _get_settings(app)
        yield

    app = FastAPI(title="Loop-Ad Decision API", lifespan=lifespan)
    app.state.settings = settings
    app.state.job_service_factory = job_service_factory

    @app.get("/health")
    def health() -> dict[str, str]:
        resolved_settings = _get_settings(app)
        return {
            "status": "ok",
            "service": resolved_settings.service_id,
            "env": resolved_settings.env,
        }

    @app.post(
        "/internal/jobs/daily-decision/run",
        response_model=RunDailyDecisionResponse,
        status_code=status.HTTP_202_ACCEPTED,
    )
    def run_daily_decision(
        payload: RunDailyDecisionRequest,
        background_tasks: BackgroundTasks,
        request: Request,
        x_loop_ad_internal_key: str | None = Header(
            default=None,
            alias="X-Loop-Ad-Internal-Key",
        ),
        x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
        x_loop_ad_requester: str | None = Header(
            default=None,
            alias="X-Loop-Ad-Requester",
        ),
    ) -> RunDailyDecisionResponse:
        resolved_settings = _get_settings(app)
        if not _is_authorized(
            settings=resolved_settings,
            internal_key=x_loop_ad_internal_key,
            legacy_admin_token=x_admin_token,
        ):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

        job_service = _get_job_service(app, resolved_settings)
        client_host = request.client.host if request.client is not None else None
        try:
            run = job_service.start_run(
                DecisionRunRequest(
                    project_key=payload.project_key,
                    analysis_date=payload.analysis_date,
                    mode=payload.mode,
                    force=payload.force,
                    run_type="manual_api",
                    trigger_source="api",
                    requested_by=x_loop_ad_requester or client_host,
                )
            )
        except ProjectNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

        background_tasks.add_task(job_service.execute_run, run.run_id)
        return RunDailyDecisionResponse(
            run_id=run.run_id,
            project_key=run.project_key,
            analysis_date=run.analysis_date,
            status=run.status,
            message="daily decision job started",
        )

    return app


def _get_settings(app: FastAPI) -> Settings:
    settings = app.state.settings
    if settings is None:
        settings = load_settings()
        app.state.settings = settings
    return settings


def _get_job_service(app: FastAPI, settings: Settings) -> JobService:
    factory = app.state.job_service_factory
    if factory is not None:
        return factory()
    return build_daily_decision_job_service(settings)


def _is_authorized(
    *,
    settings: Settings,
    internal_key: str | None,
    legacy_admin_token: str | None,
) -> bool:
    if internal_key and hmac.compare_digest(internal_key, settings.internal_api_key):
        return True
    if (
        settings.legacy_admin_token
        and legacy_admin_token
        and hmac.compare_digest(legacy_admin_token, settings.legacy_admin_token)
    ):
        return True
    return False


app = create_app()
