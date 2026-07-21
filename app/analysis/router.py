from collections.abc import Iterator
from typing import NoReturn

from fastapi import APIRouter, Depends, HTTPException, Request, status
from psycopg import IntegrityError, errors

from app.audience_contract import SegmentAudienceContractError
from app.audience_allocation import (
    AudienceAllocationService,
    PostgresAudienceAllocationRepository,
)
from app.audience_exclusions import (
    PromotionAudienceExclusionRepository,
    SegmentAudienceExclusionError,
)
from app.analysis.audience_selection import build_audience_selection_policy
from app.analysis.audience_search_repository import (
    PgClickHouseAudienceVectorSearchRepository,
)
from app.analysis.audience_snapshot_repository import (
    AudienceSnapshotBindingError,
    AudienceSnapshotRepository,
)
from app.analysis.audience_v2 import AudienceV2Coordinator
from app.analysis.repositories import (
    HotelProfileRepository,
    PromotionAnalysisRepository,
    PromotionTargetSegmentWrite,
    PromotionRepository,
    PsycopgPostgresExecutor,
    SegmentDefinitionRepository,
    SegmentVectorRepository,
    UserBehaviorVectorRepository,
)
from app.analysis.report_generator import build_segment_suggestion_report_generator
from app.analysis.raw_event_segments import build_promotion_intent_extractor
from app.analysis.segment_performance import build_segment_performance_predictor
from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    ContentBriefResponse,
    SegmentAnalysisRequest,
    TargetSegmentResponse,
)
from app.analysis.segment_suggester import VectorClusterSegmentSuggester
from app.analysis.service import (
    PromotionAnalysisResult,
    PromotionAnalysisService,
    PromotionNotFoundError,
    SegmentSelectionError,
)
from app.analysis.vector_service import (
    DEFAULT_VECTOR_VERSION,
    SegmentVectorConflictError,
    SegmentVectorDataUnavailableError,
    SegmentVectorService,
)
from app.content_brief import normalize_content_brief
from app.db import create_clickhouse_client, create_postgres_connection
from app.dependencies import get_settings

router = APIRouter(prefix="/decision/v1/promotions", tags=["analysis"])


UNIQUE_CONSTRAINTS = {
    "promotion_analyses_pkey",
    "uq_promotion_segment_suggestions_analysis_segment",
}


def get_analysis_service(request: Request) -> Iterator[PromotionAnalysisService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    clickhouse_client = None
    try:
        clickhouse_client = create_clickhouse_client(settings)
        postgres_executor = PsycopgPostgresExecutor(connection)
        exclusion_repository = PromotionAudienceExclusionRepository(
            postgres=postgres_executor,
            clickhouse=clickhouse_client,
        )
        allocation_service = AudienceAllocationService(
            PostgresAudienceAllocationRepository(
                postgres=postgres_executor,
                exclusion_reader=exclusion_repository,
            )
        )
        user_behavior_vector_repository = UserBehaviorVectorRepository(clickhouse_client)
        segment_vector_repository = SegmentVectorRepository(postgres_executor)
        segment_vector_service = SegmentVectorService(
            segment_vector_repository=segment_vector_repository,
            user_behavior_vector_repository=user_behavior_vector_repository,
        )
        audience_search_repository = PgClickHouseAudienceVectorSearchRepository(
            postgres=postgres_executor,
            clickhouse=clickhouse_client,
            exclusion_repository=exclusion_repository,
        )
        audience_v2_coordinator = AudienceV2Coordinator(
            search_repository=audience_search_repository,
            snapshot_repository=AudienceSnapshotRepository(postgres_executor),
            segment_vector_service=segment_vector_service,
        )
        segment_report_generator = build_segment_suggestion_report_generator(settings)
        yield PromotionAnalysisService(
            promotion_repository=PromotionRepository(postgres_executor),
            segment_definition_repository=SegmentDefinitionRepository(
                postgres_executor,
            ),
            hotel_profile_repository=HotelProfileRepository(clickhouse_client),
            promotion_analysis_repository=PromotionAnalysisRepository(
                postgres_executor,
            ),
            segment_vector_service=segment_vector_service,
            segment_suggester=VectorClusterSegmentSuggester(
                user_behavior_vector_repository=user_behavior_vector_repository,
                raw_event_signal_repository=user_behavior_vector_repository,
                audience_context_provider=audience_search_repository,
                promotion_intent_extractor=build_promotion_intent_extractor(settings),
                performance_predictor=build_segment_performance_predictor(
                    settings.segment_performance_model_path
                ),
                audience_selection_policy=build_audience_selection_policy(),
                vector_version=DEFAULT_VECTOR_VERSION,
            ),
            segment_report_generator=segment_report_generator,
            audience_v2_coordinator=audience_v2_coordinator,
            audience_allocation_service=allocation_service,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        _close_clickhouse_client(clickhouse_client)


@router.post(
    "/{promotion_id}/segment-suggestions/recommend",
    response_model=AnalysisResponse,
    response_model_exclude_none=True,
)
def recommend_promotion_segments(
    promotion_id: str,
    request: AnalysisRequest,
    analysis_service: PromotionAnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    _validate_promotion_id(promotion_id, request.promotion_id)

    try:
        return _analysis_response_from_result(
            analysis_service.recommend_segments(request)
        )
    except Exception as exc:
        _raise_analysis_http_error(exc)


@router.post(
    "/{promotion_id}/analyses",
    response_model=AnalysisResponse,
    response_model_exclude_none=True,
)
def analyze_promotion_segments(
    promotion_id: str,
    request: SegmentAnalysisRequest,
    analysis_service: PromotionAnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    _validate_promotion_id(promotion_id, request.promotion_id)

    try:
        return _analysis_response_from_result(
            analysis_service.analyze_segments(request)
        )
    except Exception as exc:
        _raise_analysis_http_error(exc)


@router.post(
    "/{promotion_id}/analysis",
    response_model=AnalysisResponse,
    response_model_exclude_none=True,
    deprecated=True,
    include_in_schema=False,
)
def analyze_promotion_legacy(
    promotion_id: str,
    request: AnalysisRequest,
    analysis_service: PromotionAnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    """Compatibility alias for clients migrating to the recommendation endpoint."""
    return recommend_promotion_segments(
        promotion_id=promotion_id,
        request=request,
        analysis_service=analysis_service,
    )


def _validate_promotion_id(path_promotion_id: str, request_promotion_id: str) -> None:
    if path_promotion_id != request_promotion_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path promotion_id must match request promotion_id",
        )


def _raise_analysis_http_error(exc: Exception) -> NoReturn:
    if isinstance(exc, SegmentAudienceExclusionError):
        status_code = (
            422
            if exc.code == "segment_audience_exclusion_contract_missing"
            else 409
        )
        raise HTTPException(status_code=status_code, detail=exc.to_detail()) from exc
    if isinstance(exc, SegmentAudienceContractError):
        raise HTTPException(status_code=422, detail=exc.to_detail()) from exc
    if isinstance(exc, PromotionNotFoundError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    if isinstance(exc, SegmentSelectionError):
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    if isinstance(exc, AudienceSnapshotBindingError):
        raise HTTPException(status_code=409, detail=exc.to_detail()) from exc
    if isinstance(exc, SegmentVectorConflictError):
        raise HTTPException(status_code=409, detail=exc.to_detail()) from exc
    if isinstance(exc, SegmentVectorDataUnavailableError):
        raise HTTPException(
            status_code=422,
            detail="segment vector data unavailable",
        ) from exc
    if isinstance(exc, IntegrityError) and _is_unique_violation(exc):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                "promotion analysis already exists or contains duplicate "
                "segment suggestions"
            ),
        ) from exc
    raise exc


def _analysis_response_from_result(
    result: PromotionAnalysisResult,
) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_id=result.analysis.analysis_id,
        promotion_id=result.analysis.promotion_id,
        status=AnalysisStatus(result.analysis.status),
        target_segments=[
            _target_segment_response(target_segment)
            for target_segment in result.target_segments
        ],
    )


def _target_segment_response(
    target_segment: PromotionTargetSegmentWrite,
) -> TargetSegmentResponse:
    if target_segment.segment_vector_id is None:
        raise RuntimeError("analysis target segment must have segment_vector_id")
    content_brief = normalize_content_brief(target_segment.content_brief_json)
    return TargetSegmentResponse(
        segment_id=target_segment.segment_id,
        segment_name=target_segment.segment_name,
        segment_vector_id=target_segment.segment_vector_id,
        estimated_size=target_segment.estimated_size,
        audience_snapshot_id=target_segment.audience_snapshot_id,
        eligible_user_count=target_segment.data_evidence_json.get(
            "total_eligible_user_count"
        ),
        behavior_match_count=target_segment.data_evidence_json.get(
            "matching_user_count"
        ),
        final_audience_count=(
            target_segment.estimated_size
            if target_segment.audience_snapshot_id is not None
            else None
        ),
        meets_min_sample_size=target_segment.data_evidence_json.get(
            "meets_min_sample_size"
        ),
        targetable=target_segment.data_evidence_json.get("targetable"),
        audience_status=target_segment.data_evidence_json.get("audience_status"),
        selection_method=target_segment.data_evidence_json.get(
            "selection_method"
        ),
        recall_lower_bound=target_segment.data_evidence_json.get(
            "recall_lower_bound"
        ),
        content_brief=ContentBriefResponse(
            message_direction=content_brief.message_direction,
            keywords=content_brief.keywords,
        ),
    )


def _is_unique_violation(exc: IntegrityError) -> bool:
    if isinstance(exc, errors.UniqueViolation):
        return True
    constraint_name = getattr(getattr(exc, "diag", None), "constraint_name", None)
    return constraint_name in UNIQUE_CONSTRAINTS


def _close_clickhouse_client(clickhouse_client: object | None) -> None:
    if clickhouse_client is None:
        return
    close = getattr(clickhouse_client, "close", None)
    if callable(close):
        close()
