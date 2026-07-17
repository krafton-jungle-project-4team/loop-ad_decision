from __future__ import annotations

from collections.abc import Iterator
from json import JSONDecodeError

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from psycopg import IntegrityError, errors
from pydantic import ValidationError

from app.analysis.audience_search_repository import (
    PgClickHouseAudienceVectorSearchRepository,
)
from app.analysis.audience_snapshot_repository import (
    AudienceSnapshotRepository as AnalysisAudienceSnapshotRepository,
)
from app.analysis.audience_v2 import AudienceV2Coordinator
from app.audience_allocation import (
    AudienceAllocationService,
    PostgresAudienceAllocationRepository,
)
from app.audience_exclusions import PromotionAudienceExclusionRepository
from app.analysis.repositories import (
    HotelProfileRepository as AnalysisHotelProfileRepository,
    PromotionAnalysisRepository as AnalysisPromotionAnalysisRepository,
    PromotionRepository as AnalysisPromotionRepository,
    PsycopgPostgresExecutor as AnalysisPostgresExecutor,
    SegmentDefinitionRepository as AnalysisSegmentDefinitionRepository,
    SegmentVectorRepository as AnalysisSegmentVectorRepository,
    UserBehaviorVectorRepository as AnalysisUserBehaviorVectorRepository,
)
from app.analysis.report_generator import build_segment_suggestion_report_generator
from app.analysis.raw_event_segments import build_promotion_intent_extractor
from app.analysis.segment_suggester import VectorClusterSegmentSuggester
from app.analysis.service import PromotionAnalysisService
from app.analysis.vector_service import (
    DEFAULT_VECTOR_VERSION,
    SegmentVectorDataUnavailableError,
    SegmentVectorService,
)
from app.db import create_clickhouse_client, create_postgres_connection
from app.decision.assignment_service import (
    SegmentAssignmentAudienceContractError,
    SegmentAssignmentRunNotFoundError,
    SegmentAssignmentService,
    SegmentAssignmentValidationError,
)
from app.decision.audience_snapshots import AudienceSnapshotRepository
from app.decision.evaluation_service import (
    AdExperimentEvaluationNotFoundError,
    AdExperimentEvaluationService,
    AdExperimentEvaluationValidationError,
    PromotionRunEvaluationNotFoundError,
    PromotionRunEvaluationService,
    PromotionRunEvaluationValidationError,
)
from app.decision.matcher import SegmentCandidateReranker
from app.decision.next_loop_service import (
    NextLoopConflictError,
    NextLoopGenerationFailedError,
    NextLoopNotFoundError,
    NextLoopService,
    NextLoopValidationError,
    ServiceNextLoopAnalysisGateway,
    ServiceNextLoopGenerationGateway,
)
from app.decision.repositories import (
    AdExperimentRepository,
    ContentCandidateRepository,
    EvaluationMetricRepository,
    GenerationRunRepository,
    NextLoopPreparationRecord,
    NextLoopPreparationRepository,
    PromotionAnalysisRepository,
    PromotionEvaluationRepository,
    PromotionRepository,
    PromotionRunRepository,
    PromotionTargetSegmentRepository,
    PsycopgPostgresExecutor,
    SegmentVectorRepository,
    UserBehaviorVectorRepository,
    UserSegmentAssignmentRepository,
)
from app.decision.schemas import (
    AdExperimentEvaluateRequest,
    AdExperimentEvaluateResponse,
    NextLoopRequest,
    NextLoopResponse,
    PromotionRunEvaluateRequest,
    PromotionRunEvaluateResponse,
    RunCreateRequest,
    RunCreateResponse,
    SegmentAssignmentBuildRequest,
    SegmentAssignmentBuildResponse,
)
from app.decision.service import (
    PromotionNotFoundError,
    PromotionRunService,
    RunConflictError,
    RunValidationError,
)
from app.dependencies import get_settings
from app.generation.adapters import (
    build_external_content_generator,
    build_s3_creative_artifact_publisher,
)
from app.generation.brand_context import (
    BrandContextRepository,
    BrandContextRetrievalService,
    OpenAIEmbeddingClient,
)
from app.generation.brand_context_s3 import S3BrandContextLoader
from app.generation.repositories import (
    ContentCandidateRepository as GenerationContentCandidateRepository,
    GenerationInputRepository,
    GenerationRunRepository as GenerationGenerationRunRepository,
)
from app.generation.service import GenerationService


UNIQUE_CONSTRAINTS = {
    "promotion_analyses_pkey",
    "generation_runs_pkey",
    "content_candidates_pkey",
    "uq_content_candidates_one_approved_per_segment",
    "uq_promotion_runs_segment_scope",
    "uq_ad_experiments_segment_per_run",
}

APPROVED_CONTENT_UNIQUE_CONSTRAINT = "uq_content_candidates_one_approved_per_segment"


class SerializedNextLoopPreparationRepository(NextLoopPreparationRepository):
    """Serialize manual preparation creation for a single source run."""

    def get_active_by_source_run(
        self,
        source_promotion_run_id: str,
    ) -> NextLoopPreparationRecord | None:
        self._db.execute(
            """
            SELECT pg_advisory_xact_lock(
                hashtext('next-loop-preparation'),
                hashtext(%s)
            )
            """,
            (source_promotion_run_id,),
        )
        return super().get_active_by_source_run(source_promotion_run_id)


router = APIRouter(
    prefix="/decision/v1/promotions",
    tags=["decision-runs"],
)

promotion_run_router = APIRouter(
    prefix="/decision/v1/promotion-runs",
    tags=["decision-segment-assignments"],
)

ad_experiment_router = APIRouter(
    prefix="/decision/v1/ad-experiments",
    tags=["decision-ad-experiment-evaluations"],
)


def get_promotion_run_service(request: Request) -> Iterator[PromotionRunService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    executor = PsycopgPostgresExecutor(connection)
    audience_snapshot_repository = AudienceSnapshotRepository(executor)
    try:
        yield PromotionRunService(
            promotion_repository=PromotionRepository(executor),
            promotion_analysis_repository=PromotionAnalysisRepository(executor),
            promotion_target_segment_repository=PromotionTargetSegmentRepository(
                executor,
            ),
            generation_run_repository=GenerationRunRepository(executor),
            content_candidate_repository=ContentCandidateRepository(executor),
            promotion_run_repository=PromotionRunRepository(executor),
            ad_experiment_repository=AdExperimentRepository(executor),
            promotion_evaluation_repository=PromotionEvaluationRepository(executor),
            next_loop_preparation_repository=NextLoopPreparationRepository(executor),
            run_audience_binding_repository=audience_snapshot_repository,
            manual_activation_enabled=_manual_next_loop_enabled(request),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def get_segment_assignment_service(
    request: Request,
) -> Iterator[SegmentAssignmentService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    clickhouse_client = create_clickhouse_client(settings)
    executor = PsycopgPostgresExecutor(connection)
    try:
        yield SegmentAssignmentService(
            promotion_run_repository=PromotionRunRepository(executor),
            ad_experiment_repository=AdExperimentRepository(executor),
            segment_vector_repository=SegmentVectorRepository(executor),
            user_behavior_vector_repository=UserBehaviorVectorRepository(
                clickhouse_client,
            ),
            user_segment_assignment_repository=UserSegmentAssignmentRepository(
                executor,
            ),
            reranker=SegmentCandidateReranker(),
            audience_snapshot_repository=AudienceSnapshotRepository(executor),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        _close_clickhouse_client(clickhouse_client)


def get_ad_experiment_evaluation_service(
    request: Request,
) -> Iterator[AdExperimentEvaluationService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    clickhouse_client = create_clickhouse_client(settings)
    executor = PsycopgPostgresExecutor(connection)
    try:
        yield AdExperimentEvaluationService(
            ad_experiment_repository=AdExperimentRepository(executor),
            promotion_run_repository=PromotionRunRepository(executor),
            promotion_evaluation_repository=PromotionEvaluationRepository(executor),
            evaluation_metric_repository=EvaluationMetricRepository(
                clickhouse_client,
            ),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        _close_clickhouse_client(clickhouse_client)


def get_promotion_run_evaluation_service(
    request: Request,
) -> Iterator[PromotionRunEvaluationService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    clickhouse_client = create_clickhouse_client(settings)
    executor = PsycopgPostgresExecutor(connection)
    ad_experiment_repository = AdExperimentRepository(executor)
    promotion_run_repository = PromotionRunRepository(executor)
    promotion_evaluation_repository = PromotionEvaluationRepository(executor)
    try:
        ad_experiment_evaluation_service = AdExperimentEvaluationService(
            ad_experiment_repository=ad_experiment_repository,
            promotion_run_repository=promotion_run_repository,
            promotion_evaluation_repository=promotion_evaluation_repository,
            evaluation_metric_repository=EvaluationMetricRepository(
                clickhouse_client,
            ),
        )
        yield PromotionRunEvaluationService(
            promotion_run_repository=promotion_run_repository,
            ad_experiment_repository=ad_experiment_repository,
            promotion_evaluation_repository=promotion_evaluation_repository,
            ad_experiment_evaluation_service=ad_experiment_evaluation_service,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        _close_clickhouse_client(clickhouse_client)


def get_next_loop_service(request: Request) -> Iterator[NextLoopService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    clickhouse_client = None
    executor = PsycopgPostgresExecutor(connection)
    promotion_repository = PromotionRepository(executor)
    promotion_run_repository = PromotionRunRepository(executor)
    ad_experiment_repository = AdExperimentRepository(executor)
    promotion_evaluation_repository = PromotionEvaluationRepository(executor)
    next_loop_preparation_repository = SerializedNextLoopPreparationRepository(
        executor
    )
    try:
        clickhouse_client = create_clickhouse_client(settings)
        analysis_executor = AnalysisPostgresExecutor(connection)
        analysis_user_behavior_vector_repository = AnalysisUserBehaviorVectorRepository(
            clickhouse_client
        )
        analysis_segment_vector_repository = AnalysisSegmentVectorRepository(
            analysis_executor
        )
        analysis_segment_vector_service = SegmentVectorService(
            segment_vector_repository=analysis_segment_vector_repository,
            user_behavior_vector_repository=analysis_user_behavior_vector_repository,
        )
        audience_v2_coordinator = AudienceV2Coordinator(
            search_repository=PgClickHouseAudienceVectorSearchRepository(
                postgres=analysis_executor,
                clickhouse=clickhouse_client,
                exclusion_repository=PromotionAudienceExclusionRepository(
                    postgres=analysis_executor,
                    clickhouse=clickhouse_client,
                ),
            ),
            snapshot_repository=AnalysisAudienceSnapshotRepository(
                analysis_executor
            ),
            segment_vector_service=analysis_segment_vector_service,
        )
        analysis_service = PromotionAnalysisService(
            promotion_repository=AnalysisPromotionRepository(analysis_executor),
            segment_definition_repository=AnalysisSegmentDefinitionRepository(
                analysis_executor,
            ),
            hotel_profile_repository=AnalysisHotelProfileRepository(clickhouse_client),
            promotion_analysis_repository=AnalysisPromotionAnalysisRepository(
                analysis_executor,
            ),
            segment_vector_service=analysis_segment_vector_service,
            segment_suggester=VectorClusterSegmentSuggester(
                user_behavior_vector_repository=analysis_user_behavior_vector_repository,
                raw_event_signal_repository=analysis_user_behavior_vector_repository,
                promotion_intent_extractor=build_promotion_intent_extractor(settings),
                vector_version=DEFAULT_VECTOR_VERSION,
            ),
            segment_report_generator=build_segment_suggestion_report_generator(settings),
            audience_v2_coordinator=audience_v2_coordinator,
            audience_allocation_service=AudienceAllocationService(
                PostgresAudienceAllocationRepository(
                    postgres=analysis_executor,
                    exclusion_reader=PromotionAudienceExclusionRepository(
                        postgres=analysis_executor,
                        clickhouse=clickhouse_client,
                    ),
                )
            ),
        )
        content_generator = None
        artifact_publisher = None
        brand_context_snapshot_reader = None
        brand_context_provider = None
        if settings.env != "test":
            content_generator = build_external_content_generator(settings)
            artifact_publisher = build_s3_creative_artifact_publisher(settings)
            brand_context_source_loader = S3BrandContextLoader(
                bucket_name=settings.data_storage_bucket,
                base_prefix=settings.brand_context_base_prefix,
            )
            brand_context_repository = BrandContextRepository(connection)
            brand_context_snapshot_reader = brand_context_source_loader
            brand_context_provider = BrandContextRetrievalService(
                repository=brand_context_repository,
                embedding_client=OpenAIEmbeddingClient(
                    api_key=settings.openai_api_key,
                    timeout_seconds=settings.generation_provider_timeout_seconds,
                ),
                source_loader=brand_context_source_loader,
            )
        generation_run_repository = GenerationGenerationRunRepository(connection)
        generation_content_candidate_repository = (
            GenerationContentCandidateRepository(connection)
        )
        generation_service = GenerationService(
            generation_run_repository=generation_run_repository,
            content_candidate_repository=generation_content_candidate_repository,
            generation_input_reader=GenerationInputRepository(connection),
            brand_context_snapshot_reader=brand_context_snapshot_reader,
            brand_context_provider=brand_context_provider,
            content_generator=content_generator,
            generation_model_version=(
                settings.openai_content_model
                if settings.env != "test"
                else None
            ),
            artifact_publisher=artifact_publisher,
        )
        run_creator = PromotionRunService(
            promotion_repository=promotion_repository,
            promotion_analysis_repository=PromotionAnalysisRepository(executor),
            promotion_target_segment_repository=PromotionTargetSegmentRepository(
                executor,
            ),
            generation_run_repository=GenerationRunRepository(executor),
            content_candidate_repository=ContentCandidateRepository(executor),
            promotion_run_repository=promotion_run_repository,
            ad_experiment_repository=ad_experiment_repository,
            promotion_evaluation_repository=promotion_evaluation_repository,
            next_loop_preparation_repository=next_loop_preparation_repository,
            run_audience_binding_repository=AudienceSnapshotRepository(executor),
            manual_activation_enabled=_manual_next_loop_enabled(request),
        )
        yield NextLoopService(
            promotion_repository=promotion_repository,
            promotion_run_repository=promotion_run_repository,
            ad_experiment_repository=ad_experiment_repository,
            promotion_evaluation_repository=promotion_evaluation_repository,
            next_loop_preparation_repository=next_loop_preparation_repository,
            generation_run_repository=GenerationRunRepository(executor),
            content_candidate_repository=generation_content_candidate_repository,
            analysis_gateway=ServiceNextLoopAnalysisGateway(analysis_service),
            generation_gateway=ServiceNextLoopGenerationGateway(generation_service),
            run_creator=run_creator,
            manual_prepare_enabled=_manual_next_loop_enabled(request),
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
        _close_clickhouse_client(clickhouse_client)


@router.post(
    "/{promotion_id}/runs",
    response_model=RunCreateResponse,
    status_code=status.HTTP_200_OK,
)
async def create_promotion_run(
    promotion_id: str,
    request: Request,
    promotion_run_service: PromotionRunService = Depends(get_promotion_run_service),
) -> RunCreateResponse:
    run_request = await _parse_run_create_request(request)
    try:
        return promotion_run_service.create_run(
            promotion_id=promotion_id,
            request=run_request,
        )
    except PromotionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RunValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RunConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(exc.to_detail() if hasattr(exc, "to_detail") else str(exc)),
        ) from exc
    except IntegrityError as exc:
        if _is_unique_violation(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="promotion run or ad experiment already exists",
            ) from exc
        raise


@promotion_run_router.post(
    "/{promotion_run_id}/segment-assignments/build",
    response_model=SegmentAssignmentBuildResponse,
    status_code=status.HTTP_200_OK,
)
async def build_segment_assignments(
    promotion_run_id: str,
    request: Request,
    segment_assignment_service: SegmentAssignmentService = Depends(
        get_segment_assignment_service
    ),
) -> SegmentAssignmentBuildResponse:
    build_request = await _parse_segment_assignment_build_request(request)
    try:
        return segment_assignment_service.build_assignments(
            promotion_run_id=promotion_run_id,
            request=build_request,
        )
    except SegmentAssignmentRunNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except SegmentAssignmentAudienceContractError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=exc.to_detail(),
        ) from exc
    except SegmentAssignmentValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@promotion_run_router.post(
    "/{promotion_run_id}/evaluate",
    response_model=PromotionRunEvaluateResponse,
    status_code=status.HTTP_200_OK,
)
async def evaluate_promotion_run(
    promotion_run_id: str,
    request: Request,
    promotion_run_evaluation_service: PromotionRunEvaluationService = Depends(
        get_promotion_run_evaluation_service
    ),
) -> PromotionRunEvaluateResponse:
    evaluate_request = await _parse_promotion_run_evaluate_request(request)
    try:
        return promotion_run_evaluation_service.evaluate(
            promotion_run_id=promotion_run_id,
            request=evaluate_request,
        )
    except PromotionRunEvaluationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except (
        PromotionRunEvaluationValidationError,
        AdExperimentEvaluationValidationError,
    ) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


@promotion_run_router.post(
    "/{promotion_run_id}/next-loop",
    response_model=NextLoopResponse,
    status_code=status.HTTP_200_OK,
)
async def create_next_loop(
    promotion_run_id: str,
    request: Request,
    next_loop_service: NextLoopService = Depends(
        get_next_loop_service,
        scope="function",
    ),
) -> NextLoopResponse | JSONResponse:
    next_loop_request = await _parse_next_loop_request(request)
    try:
        return next_loop_service.create_next_loop(
            promotion_run_id=promotion_run_id,
            request=next_loop_request,
        )
    except NextLoopNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NextLoopGenerationFailedError as exc:
        # Returning a response lets the dependency commit the failed generation
        # diagnostics while preserving the existing 422 API contract.
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            content={"detail": str(exc)},
        )
    except NextLoopValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except SegmentVectorDataUnavailableError as exc:
        raise HTTPException(
            status_code=422,
            detail="segment vector data unavailable",
        ) from exc
    except NextLoopConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except (RunValidationError, PromotionNotFoundError) as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except RunConflictError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(exc.to_detail() if hasattr(exc, "to_detail") else str(exc)),
        ) from exc
    except IntegrityError as exc:
        if _is_unique_violation(exc):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=_next_loop_unique_violation_detail(exc),
            ) from exc
        raise


@ad_experiment_router.post(
    "/{ad_experiment_id}/evaluate",
    response_model=AdExperimentEvaluateResponse,
    status_code=status.HTTP_200_OK,
)
async def evaluate_ad_experiment(
    ad_experiment_id: str,
    request: Request,
    ad_experiment_evaluation_service: AdExperimentEvaluationService = Depends(
        get_ad_experiment_evaluation_service
    ),
) -> AdExperimentEvaluateResponse:
    evaluate_request = await _parse_ad_experiment_evaluate_request(request)
    try:
        return ad_experiment_evaluation_service.evaluate(
            ad_experiment_id=ad_experiment_id,
            request=evaluate_request,
        )
    except AdExperimentEvaluationNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except AdExperimentEvaluationValidationError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc


async def _parse_run_create_request(request: Request) -> RunCreateRequest:
    try:
        payload = await request.json()
        return RunCreateRequest.model_validate(payload)
    except JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be valid JSON",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=jsonable_encoder(exc.errors()),
        ) from exc


async def _parse_segment_assignment_build_request(
    request: Request,
) -> SegmentAssignmentBuildRequest:
    try:
        payload = await request.json()
        return SegmentAssignmentBuildRequest.model_validate(payload)
    except JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be valid JSON",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=jsonable_encoder(exc.errors()),
        ) from exc


async def _parse_ad_experiment_evaluate_request(
    request: Request,
) -> AdExperimentEvaluateRequest:
    try:
        payload = await request.json()
        return AdExperimentEvaluateRequest.model_validate(payload)
    except JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be valid JSON",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=jsonable_encoder(exc.errors()),
        ) from exc


async def _parse_promotion_run_evaluate_request(
    request: Request,
) -> PromotionRunEvaluateRequest:
    try:
        payload = await request.json()
        return PromotionRunEvaluateRequest.model_validate(payload)
    except JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be valid JSON",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=jsonable_encoder(exc.errors()),
        ) from exc


async def _parse_next_loop_request(request: Request) -> NextLoopRequest:
    try:
        payload = await request.json()
        return NextLoopRequest.model_validate(payload)
    except JSONDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="request body must be valid JSON",
        ) from exc
    except ValidationError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=jsonable_encoder(exc.errors()),
        ) from exc


def _manual_next_loop_enabled(request: Request) -> bool:
    return bool(
        getattr(
            getattr(getattr(request, "app", None), "state", None),
            "manual_next_loop_enabled",
            False,
        )
    )


def _is_unique_violation(exc: IntegrityError) -> bool:
    if isinstance(exc, errors.UniqueViolation):
        return True
    constraint_name = _unique_violation_constraint_name(exc)
    return constraint_name in UNIQUE_CONSTRAINTS


def _unique_violation_constraint_name(exc: IntegrityError) -> str | None:
    return getattr(getattr(exc, "diag", None), "constraint_name", None)


def _next_loop_unique_violation_detail(exc: IntegrityError) -> str:
    if _unique_violation_constraint_name(exc) == APPROVED_CONTENT_UNIQUE_CONSTRAINT:
        return "approved content already exists for segment"
    return "next-loop output already exists"


def _close_clickhouse_client(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
