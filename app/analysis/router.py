from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.analysis.schemas import (
    AnalysisJobCreateResponse,
    AnalysisJobStatusResponse,
    FunnelRecommendationAnalysisRequest,
    FunnelRecommendationAnalysisResponse,
)
from app.analysis.service import run_funnel_recommendation_analysis
from app.db.clickhouse import ClickHouseClientFactory, get_clickhouse_client_factory
from app.db.postgres import get_postgres_session
from app.persistence.job_statuses import ANALYSIS_JOB_STATUS_QUEUED
from app.metrics.repository import FunnelMetricsRepository
from app.persistence.repository import PostgresRepository
from app.root_causes.repository import RootCauseRepository

router = APIRouter(prefix="/analysis", tags=["analysis"])


def get_analysis_repository(
    postgres_session: Annotated[Session, Depends(get_postgres_session)],
) -> PostgresRepository:
    return PostgresRepository(postgres_session)


@router.post("/funnel/recommend", response_model=FunnelRecommendationAnalysisResponse)
def recommend_for_funnel_analysis(
    request: FunnelRecommendationAnalysisRequest,
    client_factory: Annotated[ClickHouseClientFactory, Depends(get_clickhouse_client_factory)],
    repository: Annotated[PostgresRepository, Depends(get_analysis_repository)],
) -> FunnelRecommendationAnalysisResponse:
    with client_factory() as client:
        return run_funnel_recommendation_analysis(
            request=request,
            metrics_repository=FunnelMetricsRepository(client),
            root_cause_repository=RootCauseRepository(client),
            persistence_repository=repository,
        )


@router.post("/funnel/recommend/jobs", response_model=AnalysisJobCreateResponse)
def create_funnel_recommendation_analysis_job(
    request: FunnelRecommendationAnalysisRequest,
    repository: Annotated[PostgresRepository, Depends(get_analysis_repository)],
) -> AnalysisJobCreateResponse:
    try:
        job = repository.create_analysis_job(
            project_id=request.project_id,
            request_json=request.model_dump(mode="json"),
            status=ANALYSIS_JOB_STATUS_QUEUED,
        )
        repository.commit()
    except Exception:
        repository.rollback()
        raise

    return AnalysisJobCreateResponse(
        job_id=job.id,
        status=job.status,
        recommendation_result_id=job.recommendation_result_id,
        polling_url=f"/analysis/jobs/{job.id}",
    )


@router.get("/jobs/{job_id}", response_model=AnalysisJobStatusResponse)
def get_analysis_job_status(
    job_id: int,
    repository: Annotated[PostgresRepository, Depends(get_analysis_repository)],
) -> AnalysisJobStatusResponse:
    job = repository.get_analysis_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis job not found.",
        )

    return AnalysisJobStatusResponse(
        job_id=job.id,
        status=job.status,
        recommendation_result_id=job.recommendation_result_id,
        error_message=job.error_message,
        created_at=job.created_at,
        updated_at=job.updated_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
    )
