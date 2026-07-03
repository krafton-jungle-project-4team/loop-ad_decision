import pytest
from pydantic import ValidationError

from app.generation.schemas import (
    ContentCandidateResponse,
    ContentChannel,
    GenerationRequest,
)
from app.generation.repositories import ContentCandidateRecord


def test_generation_request_accepts_valid_payload() -> None:
    request = GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=3,
        operator_instruction=None,
    )

    assert request.project_id == "hotel-client-a"
    assert request.content_option_count == 3


def test_generation_request_rejects_non_positive_option_count() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            analysis_id="analysis_banner_001",
            content_option_count=0,
            operator_instruction=None,
        )


def test_generation_request_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        GenerationRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            analysis_id="analysis_banner_001",
            content_option_count=1,
            operator_instruction=None,
            creative_id="legacy",
        )


@pytest.mark.parametrize(
    "candidate",
    [
        {
            "content_id": "content_email_repeat_hotel_001",
            "content_option_id": "email_repeat_hotel_option_001",
            "segment_id": "seg_repeat_hotel_no_booking",
            "channel": "email",
            "subject": "Weekend rooms are still available",
            "preheader": "See refundable summer hotel offers.",
            "body": "Return to the hotels you viewed and compare today's offer.",
            "cta": "View hotel deals",
            "landing_url": "https://demo-stay.example.com/summer",
        },
        {
            "content_id": "content_sms_repeat_hotel_001",
            "content_option_id": "sms_repeat_hotel_option_001",
            "segment_id": "seg_repeat_hotel_no_booking",
            "channel": "sms",
            "message": "The hotel you viewed still has refundable summer rooms.",
            "landing_url": "https://demo-stay.example.com/summer",
        },
        {
            "content_id": "content_banner_repeat_hotel_001",
            "content_option_id": "banner_repeat_hotel_option_001",
            "segment_id": "seg_repeat_hotel_no_booking",
            "channel": "onsite_banner",
            "title": "Book this weekend's rooms",
            "body": "Compare refundable summer offers before rooms run out.",
            "cta": "View hotel deals",
            "image_prompt": "bright modern hotel room, summer travel banner",
            "landing_url": "https://demo-stay.example.com/summer",
        },
    ],
)
def test_content_candidate_response_accepts_channel_required_fields(candidate) -> None:
    dto = ContentCandidateResponse.model_validate(candidate)

    assert dto.status == "draft"


def test_content_candidate_response_rejects_unknown_channel() -> None:
    with pytest.raises(ValidationError):
        ContentCandidateResponse(
            content_id="content_push_repeat_hotel_001",
            content_option_id="push_repeat_hotel_option_001",
            segment_id="seg_repeat_hotel_no_booking",
            channel="push",
            message="Unsupported channel",
            landing_url="https://demo-stay.example.com/summer",
        )


def test_content_candidate_response_rejects_missing_channel_fields() -> None:
    with pytest.raises(ValidationError):
        ContentCandidateResponse(
            content_id="content_banner_repeat_hotel_001",
            content_option_id="banner_repeat_hotel_option_001",
            segment_id="seg_repeat_hotel_no_booking",
            channel=ContentChannel.ONSITE_BANNER,
            title="Book this weekend's rooms",
            body="Compare refundable summer offers before rooms run out.",
            cta="View hotel deals",
            landing_url="https://demo-stay.example.com/summer",
        )


def test_content_candidate_record_rejects_missing_channel_fields() -> None:
    with pytest.raises(ValueError):
        ContentCandidateRecord(
            content_id="content_sms_repeat_hotel_001",
            content_option_id="sms_repeat_hotel_option_001",
            generation_id="generation_banner_001",
            analysis_id="analysis_banner_001",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            segment_id="seg_repeat_hotel_no_booking",
            channel=ContentChannel.SMS,
        )
