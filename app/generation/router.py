from json import JSONDecodeError
import re

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from pydantic import ValidationError

from app.generation.schemas import (
    ContentCandidateResponse,
    ContentCandidateStatus,
    ContentChannel,
    GenerationRequest,
    GenerationResponse,
    GenerationStatus,
)


router = APIRouter(
    prefix="/decision/v1/promotions",
    tags=["generation"],
)


@router.post(
    "/{promotion_id}/generation",
    response_model=GenerationResponse,
    status_code=status.HTTP_200_OK,
)
async def create_generation(
    promotion_id: str,
    request: Request,
) -> GenerationResponse:
    generation_request = await _parse_generation_request(request)

    if promotion_id != generation_request.promotion_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="promotion_id path parameter does not match request body",
        )

    return _build_mock_generation_response(generation_request)


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


def _build_mock_generation_response(request: GenerationRequest) -> GenerationResponse:
    channel = ContentChannel.ONSITE_BANNER
    channel_slug = "banner"
    segment_slug = "repeat_hotel"
    segment_id = "seg_repeat_hotel_no_booking"
    generation_id = _generation_id_from_promotion(request.promotion_id)

    candidates = [
        ContentCandidateResponse(
            content_id=f"content_{channel_slug}_{segment_slug}_{index:03d}",
            content_option_id=f"{channel_slug}_{segment_slug}_option_{index:03d}",
            segment_id=segment_id,
            channel=channel,
            title="Book this weekend's rooms before they are gone",
            body=(
                "Show repeat hotel viewers a refundable summer offer while "
                "rooms are still available."
            ),
            cta="View hotel deals",
            image_prompt=(
                "modern hotel room summer promotion banner, clean, bright, travel"
            ),
            landing_url="https://demo-stay.example.com/summer",
            status=ContentCandidateStatus.DRAFT,
        )
        for index in range(1, request.content_option_count + 1)
    ]

    return GenerationResponse(
        generation_id=generation_id,
        promotion_id=request.promotion_id,
        status=GenerationStatus.COMPLETED,
        content_candidates=candidates,
    )


def _generation_id_from_promotion(promotion_id: str) -> str:
    promotion_slug = promotion_id.removeprefix("promo_")
    safe_slug = re.sub(r"[^a-zA-Z0-9_]+", "_", promotion_slug).strip("_")
    return f"generation_{safe_slug or 'content'}"
