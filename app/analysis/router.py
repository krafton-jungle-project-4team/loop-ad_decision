from collections.abc import Iterator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from app.analysis.repositories import (
    HotelProfileRepository,
    PromotionAnalysisRepository,
    PromotionRepository,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRepository,
    SegmentVectorRepository,
)
from app.analysis.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    AnalysisStatus,
    ContentBriefResponse,
    TargetSegmentResponse,
)
from app.analysis.service import (
    AnalysisRequestHandler,
    PromotionAnalysisResult,
    PromotionAnalysisService,
    PromotionNotFoundError,
    SegmentSelectionError,
)
from app.db import (
    PsycopgPostgresExecutor,
    create_clickhouse_client,
    create_postgres_connection,
)
from app.dependencies import get_settings

router = APIRouter(prefix="/decision/v1/promotions", tags=["analysis"])


def get_analysis_service(request: Request) -> Iterator[AnalysisRequestHandler]:
    settings = get_settings(request)
    postgres_connection: Any | None = None
    clickhouse_client: Any | None = None
    try:
        postgres_connection = create_postgres_connection(settings)
        clickhouse_client = create_clickhouse_client(settings)
        postgres = PsycopgPostgresExecutor(postgres_connection)
        yield PromotionAnalysisService(
            promotion_repository=PromotionRepository(postgres),
            segment_definition_repository=SegmentDefinitionRepository(postgres),
            hotel_profile_repository=HotelProfileRepository(clickhouse_client),
            promotion_analysis_repository=PromotionAnalysisRepository(postgres),
            segment_vector_repository=SegmentVectorRepository(postgres),
        )
        postgres_connection.commit()
    except Exception:
        if postgres_connection is not None:
            postgres_connection.rollback()
        raise
    finally:
        if postgres_connection is not None:
            postgres_connection.close()
        if clickhouse_client is not None and hasattr(clickhouse_client, "close"):
            clickhouse_client.close()


@router.post("/{promotion_id}/analysis", response_model=AnalysisResponse)
def analyze_promotion(
    promotion_id: str,
    request: AnalysisRequest,
    analysis_service: AnalysisRequestHandler = Depends(get_analysis_service),
) -> AnalysisResponse:
    if promotion_id != request.promotion_id:
        raise HTTPException(
            status_code=400,
            detail="path promotion_id must match request promotion_id",
        )

    try:
        result = analysis_service.analyze(request)
    except PromotionNotFoundError as exc:
        raise HTTPException(status_code=404, detail="promotion not found") from exc
    except SegmentSelectionError as exc:
        raise HTTPException(
            status_code=422,
            detail="no target segment candidates",
        ) from exc

    return _analysis_response_from_result(result)


def _analysis_response_from_result(result: PromotionAnalysisResult) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_id=result.analysis.analysis_id,
        promotion_id=result.analysis.promotion_id,
        status=AnalysisStatus(result.analysis.status),
        target_segments=[
            _target_segment_response_from_write(target_segment)
            for target_segment in result.target_segments
        ],
    )


def _target_segment_response_from_write(
    target_segment: PromotionTargetSegmentWrite,
) -> TargetSegmentResponse:
    if target_segment.segment_vector_id is None:
        raise RuntimeError("analysis target segment is missing segment_vector_id")
    return TargetSegmentResponse(
        segment_id=target_segment.segment_id,
        segment_name=target_segment.segment_name,
        segment_vector_id=target_segment.segment_vector_id,
        estimated_size=target_segment.estimated_size,
        content_brief=ContentBriefResponse(
            message_direction=str(
                target_segment.content_brief_json["message_direction"]
            ),
            keywords=list(target_segment.content_brief_json["keywords"]),
        ),
    )
