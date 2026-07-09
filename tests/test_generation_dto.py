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
            "channel": "email",
            "creative_format": "email_html",
            "source": {
                "creative_format": "email_html",
                "subject": "Weekend rooms are still available",
                "preheader": "See refundable summer hotel offers.",
                "text_body": "Return to the hotels you viewed and compare today's offer.",
            },
            "artifact": {
                "creative_format": "email_html",
                "artifact_status": "published",
                "storage_key": "genai/content_email_repeat_hotel_001.html",
                "public_url": "https://gen-ai.asset.dev.loop-ad.org/genai/content_email_repeat_hotel_001.html",
                "sha256": "abc123",
                "bytes": 128,
                "content_type": "text/html; charset=utf-8",
            },
        },
        {
            "channel": "sms",
            "creative_format": "sms_text",
            "source": {
                "creative_format": "sms_text",
                "message": "The hotel you viewed still has refundable summer rooms. {{redirect_url}}",
            },
            "artifact": {
                "creative_format": "sms_text",
                "artifact_status": "not_required",
            },
        },
        {
            "channel": "onsite_banner",
            "creative_format": "banner_html",
            "source": {
                "creative_format": "banner_html",
                "width": 320,
                "height": 100,
                "click_protocol": "post_message",
                "allowed_message_type": "loopad:click",
            },
            "artifact": {
                "creative_format": "banner_html",
                "artifact_status": "pending",
            },
        },
    ],
)
def test_content_candidate_response_accepts_channel_required_fields(candidate) -> None:
    candidate["attribution"] = attribution_for_candidate(candidate["channel"])

    dto = ContentCandidateResponse.model_validate(candidate)

    assert dto.channel == candidate["channel"]
    assert dto.artifact.creative_format == candidate["creative_format"]


def test_content_candidate_response_rejects_unknown_channel() -> None:
    with pytest.raises(ValidationError):
        ContentCandidateResponse(
            channel="push",
            creative_format="sms_text",
            attribution=attribution_for_candidate("sms"),
            source={
                "creative_format": "sms_text",
                "message": "Unsupported channel {{redirect_url}}",
            },
            artifact={
                "creative_format": "sms_text",
                "artifact_status": "not_required",
            },
        )


def test_content_candidate_response_rejects_missing_channel_fields() -> None:
    with pytest.raises(ValidationError):
        ContentCandidateResponse(
            channel="onsite_banner",
            creative_format="banner_html",
            attribution=attribution_for_candidate("onsite_banner"),
            source={
                "creative_format": "banner_html",
                "height": 100,
                "click_protocol": "post_message",
                "allowed_message_type": "loopad:click",
            },
            artifact={
                "creative_format": "banner_html",
                "artifact_status": "pending",
            },
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


def attribution_for_candidate(channel: str) -> dict[str, str]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "promotion_run_id": "run_promo_banner_001",
        "ad_experiment_id": "exp_promo_banner_001",
        "segment_id": "seg_repeat_hotel_no_booking",
        "content_id": "content_repeat_hotel_001",
        "content_option_id": "option_repeat_hotel_001",
        "creative_id": "content_repeat_hotel_001",
        "promotion_channel": channel,
        "target_url": "https://demo-stay.example.com/summer",
    }
