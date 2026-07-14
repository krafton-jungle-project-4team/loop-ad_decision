from collections.abc import Iterator
from json import JSONDecodeError

from fastapi import APIRouter, Depends, Header, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.db import create_postgres_connection
from app.dependencies import get_settings
from app.generation.brand_context_s3 import S3BrandContextLoader
from app.generation.errors import GenerationError
from app.generation.repositories import (
    GenerationInputRepository,
    GenerationRunRepository,
)
from app.generation.schemas import (
    GenerationAcceptedResponse,
    GenerationRequest,
)
from app.generation.submission import (
    GenerationInputUnavailable,
    GenerationIdempotencyConflict,
    GenerationSubmissionService,
    GenerationSubmissionUnavailable,
    normalize_idempotency_key,
)


router = APIRouter(
    prefix="/decision/v1/promotions",
    tags=["generation"],
)


def get_generation_service(request: Request) -> Iterator[GenerationSubmissionService]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    try:
        yield GenerationSubmissionService(
            connection=connection,
            generation_run_repository=GenerationRunRepository(connection),
            generation_input_reader=GenerationInputRepository(connection),
            brand_context_repository=S3BrandContextLoader(
                bucket_name=settings.data_storage_bucket,
                base_prefix=settings.brand_context_base_prefix,
            ),
            model_version=settings.openai_content_model,
            coordinator=getattr(
                request.app.state,
                "generation_coordinator",
                None,
            ),
        )
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@router.post(
    "/{promotion_id}/generation",
    response_model=GenerationAcceptedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_generation(
    promotion_id: str,
    request: Request,
    idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
    generation_service: GenerationSubmissionService = Depends(get_generation_service),
) -> GenerationAcceptedResponse:
    generation_request = await _parse_generation_request(request)

    if promotion_id != generation_request.promotion_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="promotion_id path parameter does not match request body",
        )

    try:
        normalized_idempotency_key = normalize_idempotency_key(
            idempotency_key or ""
        )
        return generation_service.submit(
            generation_request,
            idempotency_key=normalized_idempotency_key,
        )
    except GenerationInputUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except GenerationIdempotencyConflict as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc
    except GenerationSubmissionUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except GenerationError as exc:
        raise HTTPException(
            status_code=(
                status.HTTP_503_SERVICE_UNAVAILABLE
                if exc.retryable
                else status.HTTP_409_CONFLICT
            ),
            detail=exc.safe_message,
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc


async def _parse_generation_request(request: Request) -> GenerationRequest:
    try:
        payload = await request.json()
        return GenerationRequest.model_validate(payload)
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
