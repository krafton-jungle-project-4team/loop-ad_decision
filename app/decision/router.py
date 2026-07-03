from __future__ import annotations

from collections.abc import Iterator
from json import JSONDecodeError

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from psycopg import IntegrityError, errors
from pydantic import ValidationError

from app.db import create_clickhouse_client, create_postgres_connection
from app.decision.assignment_service import (
    SegmentAssignmentRunNotFoundError,
    SegmentAssignmentService,
    SegmentAssignmentValidationError,
)
from app.decision.matcher import ExactCosineMatcher
from app.decision.repositories import (
    AdExperimentRepository,
    ContentCandidateRepository,
    GenerationRunRepository,
    PromotionAnalysisRepository,
    PromotionRepository,
    PromotionRunRepository,
    PromotionTargetSegmentRepository,
    PsycopgPostgresExecutor,
    SegmentVectorRepository,
    UserBehaviorVectorRepository,
    UserSegmentAssignmentRepository,
)
from app.decision.schemas import (
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


UNIQUE_CONSTRAINTS = {
    "uq_promotion_runs_loop",
    "uq_ad_experiments_segment_per_run",
}


router = APIRouter(
    prefix="/decision/v1/promotions",
    tags=["decision-runs"],
)

promotion_run_router = APIRouter(
    prefix="/decision/v1/promotion-runs",
    tags=["decision-segment-assignments"],
)


def get_promotion_run_service(request: Request) -> Iterator[PromotionRunService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    executor = PsycopgPostgresExecutor(connection)
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
            promotion_target_segment_repository=PromotionTargetSegmentRepository(
                executor,
            ),
            segment_vector_repository=SegmentVectorRepository(executor),
            user_behavior_vector_repository=UserBehaviorVectorRepository(
                clickhouse_client,
            ),
            user_segment_assignment_repository=UserSegmentAssignmentRepository(
                executor,
            ),
            matcher=ExactCosineMatcher(),
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
            detail=str(exc),
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
    except SegmentAssignmentValidationError as exc:
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


def _is_unique_violation(exc: IntegrityError) -> bool:
    if isinstance(exc, errors.UniqueViolation):
        return True
    constraint_name = getattr(getattr(exc, "diag", None), "constraint_name", None)
    return constraint_name in UNIQUE_CONSTRAINTS


def _close_clickhouse_client(client: object) -> None:
    close = getattr(client, "close", None)
    if callable(close):
        close()
