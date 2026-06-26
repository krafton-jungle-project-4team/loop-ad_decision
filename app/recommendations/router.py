from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.postgres import get_postgres_session
from app.persistence.repository import PostgresRepository
from app.recommendations.schemas import (
    ActiveSegmentAdMappingResponse,
    RecommendationApprovalResponse,
    RecommendationApproveRequest,
    RecommendationRejectionResponse,
    RecommendationRejectRequest,
    RecommendationResultResponse,
)
from app.recommendations.service import (
    approve_recommendation_result,
    get_recommendation_result_response,
    list_active_segment_ad_mappings,
    list_recommendation_results,
    reject_recommendation_result,
)

recommendations_router = APIRouter(prefix="/recommendations", tags=["recommendations"])
ad_mappings_router = APIRouter(prefix="/ad-mappings", tags=["ad-mappings"])


def get_recommendation_repository(
    session: Annotated[Session, Depends(get_postgres_session)],
) -> PostgresRepository:
    return PostgresRepository(session)


@recommendations_router.get("", response_model=list[RecommendationResultResponse])
def get_recommendations(
    repository: Annotated[PostgresRepository, Depends(get_recommendation_repository)],
    project_id: str | None = None,
    status: str | None = None,
    limit: int = Query(default=100, ge=1, le=500),
) -> list[RecommendationResultResponse]:
    return list_recommendation_results(
        repository=repository,
        project_id=project_id,
        status_filter=status,
        limit=limit,
    )


@recommendations_router.get("/{recommendation_result_id}", response_model=RecommendationResultResponse)
def get_recommendation(
    recommendation_result_id: int,
    repository: Annotated[PostgresRepository, Depends(get_recommendation_repository)],
) -> RecommendationResultResponse:
    return get_recommendation_result_response(
        repository=repository,
        recommendation_result_id=recommendation_result_id,
    )


@recommendations_router.post(
    "/{recommendation_result_id}/approve",
    response_model=RecommendationApprovalResponse,
)
def approve_recommendation(
    recommendation_result_id: int,
    request: RecommendationApproveRequest,
    repository: Annotated[PostgresRepository, Depends(get_recommendation_repository)],
) -> RecommendationApprovalResponse:
    return approve_recommendation_result(
        repository=repository,
        recommendation_result_id=recommendation_result_id,
        request=request,
    )


@recommendations_router.post(
    "/{recommendation_result_id}/reject",
    response_model=RecommendationRejectionResponse,
)
def reject_recommendation(
    recommendation_result_id: int,
    request: RecommendationRejectRequest,
    repository: Annotated[PostgresRepository, Depends(get_recommendation_repository)],
) -> RecommendationRejectionResponse:
    return reject_recommendation_result(
        repository=repository,
        recommendation_result_id=recommendation_result_id,
        request=request,
    )


@ad_mappings_router.get("/active", response_model=list[ActiveSegmentAdMappingResponse])
def get_active_ad_mappings(
    project_id: str,
    repository: Annotated[PostgresRepository, Depends(get_recommendation_repository)],
) -> list[ActiveSegmentAdMappingResponse]:
    return list_active_segment_ad_mappings(
        repository=repository,
        project_id=project_id,
    )
