from json import JSONDecodeError

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.generation.schemas import (
    GenerationRequest,
    GenerationResponse,
)
from app.generation.service import GenerationRequestHandler, GenerationService


router = APIRouter(
    prefix="/decision/v1/promotions",
    tags=["generation"],
)


def get_generation_service() -> GenerationRequestHandler:
    return GenerationService()


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

    return generation_service.generate(generation_request)


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
