from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Request, status
from psycopg import IntegrityError, errors

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
from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    ContentBriefResponse,
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
    SegmentVectorDataUnavailableError,
    SegmentVectorService,
)
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
        user_behavior_vector_repository = UserBehaviorVectorRepository(clickhouse_client)
        segment_vector_repository = SegmentVectorRepository(postgres_executor)
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
            segment_vector_service=SegmentVectorService(
                segment_vector_repository=segment_vector_repository,
                user_behavior_vector_repository=user_behavior_vector_repository,
            ),
            segment_suggester=VectorClusterSegmentSuggester(
                user_behavior_vector_repository=user_behavior_vector_repository,
            ),
            segment_report_generator=segment_report_generator,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        _close_clickhouse_client(clickhouse_client)


@router.post("/{promotion_id}/analysis", response_model=AnalysisResponse)
def analyze_promotion(
    promotion_id: str,
    request: AnalysisRequest,
    analysis_service: PromotionAnalysisService = Depends(get_analysis_service),
) -> AnalysisResponse:
    if promotion_id != request.promotion_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path promotion_id must match request promotion_id",
        )

    try:
        return _analysis_response_from_result(analysis_service.analyze(request))
    except PromotionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SegmentSelectionError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except SegmentVectorDataUnavailableError as exc:
        raise HTTPException(
            status_code=422,
            detail="segment vector data unavailable",
        ) from exc
    except IntegrityError as exc:
        if _is_unique_violation(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="promotion analysis already exists or contains duplicate segment suggestions",
            ) from exc
        raise


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
    raw_keywords = target_segment.content_brief_json.get("keywords", [])
    keywords = raw_keywords if isinstance(raw_keywords, list) else []
    return TargetSegmentResponse(
        segment_id=target_segment.segment_id,
        segment_name=target_segment.segment_name,
        segment_vector_id=target_segment.segment_vector_id,
        estimated_size=target_segment.estimated_size,
        content_brief=ContentBriefResponse(
            message_direction=str(
                target_segment.content_brief_json.get("message_direction", ""),
            ),
            keywords=[str(keyword) for keyword in keywords],
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
