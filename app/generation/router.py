from collections.abc import Iterator
from json import JSONDecodeError

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.db import create_postgres_connection
from app.dependencies import get_settings
from app.generation.adapters import build_external_content_generator
from app.generation.repositories import (
    ContentCandidateRepository,
    GenerationInputRepository,
    GenerationRunRepository,
)
from app.generation.schemas import (
    GenerationRequest,
    GenerationResponse,
)
from app.generation.service import (
    GenerationInputUnavailable,
    GenerationRequestHandler,
    GenerationService,
)


router = APIRouter(
    prefix="/decision/v1/promotions",
    tags=["generation"],
)


def get_generation_service(request: Request) -> Iterator[GenerationRequestHandler]:
    settings = get_settings(request)
    connection = create_postgres_connection(settings)
    content_generator = None
    if settings.env != "test":
        content_generator = build_external_content_generator(settings)
    try:
        yield GenerationService(
            generation_run_repository=GenerationRunRepository(connection),
            content_candidate_repository=ContentCandidateRepository(connection),
            generation_input_reader=GenerationInputRepository(connection),
            content_generator=content_generator,
        )
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


@router.post(
    "/{promotion_id}/generation",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
)
async def create_generation(
    promotion_id: str,
    request: Request,
    generation_service: GenerationRequestHandler = Depends(get_generation_service),
) -> GenerationResponse:
    generation_request = await _parse_generation_request(request)

    if promotion_id != generation_request.promotion_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="promotion_id path parameter does not match request body",
        )

    try:
        return generation_service.generate(generation_request)
    except GenerationInputUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
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
