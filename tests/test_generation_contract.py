from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.generation.generator import GeneratedContent
from app.generation.prompt_builder import (
    GenerationPromptInput,
    PromotionPromptInput,
    PromptBuildResult,
    TargetSegmentPromptInput,
)
from app.generation.repositories import (
    ContentCandidateRecord,
    GenerationRunRecord,
)
from app.generation.router import get_generation_service
from app.generation.schemas import ContentChannel, GenerationRequest
from app.generation.service import GenerationService
from app.main import create_app


GENERATION_RESPONSE_KEYS = {
    "generation_id",
    "promotion_id",
    "status",
    "content_candidates",
}

CONTENT_CANDIDATE_RESPONSE_KEYS = {
    "channel",
    "creative_format",
    "attribution",
    "source",
    "artifact",
}

GENERATION_RUN_DB_PARAM_KEYS = {
    "generation_id",
    "analysis_id",
    "project_id",
    "campaign_id",
    "promotion_id",
    "content_option_count",
    "operator_instruction",
    "input_json",
    "output_json",
    "generation_report_json",
    "status",
}

CONTENT_CANDIDATE_DB_PARAM_KEYS = {
    "content_id",
    "content_option_id",
    "generation_id",
    "analysis_id",
    "project_id",
    "campaign_id",
    "promotion_id",
    "segment_id",
    "channel",
    "subject",
    "preheader",
    "title",
    "body",
    "cta",
    "message",
    "image_prompt",
    "image_url",
    "landing_url",
    "generation_prompt",
    "reason_summary",
    "data_evidence_json",
    "message_strategy",
    "metadata_json",
    "status",
}

CANDIDATE_METADATA_KEYS = {
    "report_version",
    "content_id",
    "content_option_id",
    "segment_id",
    "segment_name",
    "channel",
    "status",
    "reason_summary",
    "data_evidence",
    "message_strategy",
    "operator_instruction",
    "source_segment_definition_id",
    "source_query_preview_id",
    "generated_sql_summary",
    "prompt_builder_version",
    "content_generator_version",
    "subject",
    "preheader",
    "title",
    "body",
    "cta",
    "message",
    "image_prompt",
    "image_url",
    "landing_url",
    "creative",
}

RUN_OUTPUT_KEYS = {
    "report_version",
    "content_candidate_ids",
    "generation_summary",
    "segment_summaries",
    "content_report_summaries",
}

FORBIDDEN_PUBLIC_TERMS = (
    "variant_id",
    "generated_contents",
    "shopping mall",
    "product",
    "cart",
    "purchase",
)

IMAGE_URL = (
    "https://gen-ai.asset.dev.loop-ad.org/generated/"
    "content_banner_repeat_hotel_001.png"
)


def test_generation_api_response_contract_for_dashboard() -> None:
    client = _generation_client(GenerationService())

    response = client.post(
        "/decision/v1/promotions/promo_banner_001/generation",
        json=_generation_request_payload(content_option_count=2),
    )

    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == GENERATION_RESPONSE_KEYS
    assert payload["generation_id"] == "generation_banner_001"
    assert payload["promotion_id"] == "promo_banner_001"
    assert payload["status"] == "completed"
    assert len(payload["content_candidates"]) == 2

    candidate = payload["content_candidates"][0]
    assert set(candidate) == CONTENT_CANDIDATE_RESPONSE_KEYS
    assert candidate["channel"] == "onsite_banner"
    assert candidate["creative_format"] == "banner_html"
    assert candidate["attribution"]["content_id"] == "content_banner_repeat_hotel_001"
    assert candidate["attribution"]["content_option_id"] == "banner_repeat_hotel_option_001"
    assert candidate["attribution"]["segment_id"] == "seg_repeat_hotel_no_booking"
    assert candidate["attribution"]["creative_id"] == "content_banner_repeat_hotel_001"
    assert candidate["attribution"]["target_url"] == "https://demo-stay.example.com/summer"
    assert candidate["source"] == {
        "creative_format": "banner_html",
        "width": 320,
        "height": 100,
        "click_protocol": "post_message",
        "allowed_message_type": "loopad:click",
    }
    assert candidate["artifact"]["creative_format"] == "banner_html"
    assert candidate["artifact"]["artifact_status"] in {"pending", "published", "failed"}

    _assert_no_forbidden_terms(payload)


def test_generation_storage_contract_includes_report_and_image_url() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        content_generator=ImageUrlContentGenerator(),
    )

    response = service.generate(_generation_request(content_option_count=1))

    assert response.content_candidates[0].artifact.public_url
    assert len(generation_run_repository.saved) == 1
    assert len(content_candidate_repository.saved) == 1

    generation_run = generation_run_repository.saved[0]
    run_params = generation_run.to_db_params()
    assert set(run_params) == GENERATION_RUN_DB_PARAM_KEYS
    assert run_params["status"] == "completed"
    assert run_params["input_json"].obj == {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "content_option_count": 1,
        "operator_instruction": "Keep the hotel message concise.",
        "target_segment_ids": ["seg_repeat_hotel_no_booking"],
        "channel": "onsite_banner",
    }
    assert set(run_params["output_json"].obj) == RUN_OUTPUT_KEYS
    assert run_params["output_json"].obj["generation_summary"] == {
        "status": "completed",
        "content_candidate_count": 1,
        "target_segment_count": 1,
    }
    assert run_params["generation_report_json"].obj == {
        "status": "completed",
        "content_candidate_count": 1,
        "target_segment_count": 1,
        "prompt_builder": "dec-c2.v2",
        "content_generator": "dec-c5.image-url-test.v1",
        "report_builder": "dec-c4.v1",
    }

    candidate = content_candidate_repository.saved[0]
    candidate_params = candidate.to_db_params()
    assert set(candidate_params) == CONTENT_CANDIDATE_DB_PARAM_KEYS
    assert candidate_params["channel"] == "onsite_banner"
    assert candidate_params["status"] == "draft"
    assert candidate_params["image_url"] == IMAGE_URL
    assert candidate_params["metadata_json"].obj["image_url"] == IMAGE_URL
    assert candidate_params["metadata_json"].obj["creative"]["artifact"]["public_url"]
    assert candidate_params["data_evidence_json"].obj == (
        candidate.metadata_json["data_evidence"]
    )
    assert set(candidate.metadata_json) == CANDIDATE_METADATA_KEYS
    assert candidate.metadata_json["report_version"] == "dec-c4.v1"
    assert candidate.metadata_json["reason_summary"]
    assert candidate.metadata_json["data_evidence"]["sample_size"] == 1342
    assert candidate.metadata_json["message_strategy"]
    assert candidate.metadata_json["operator_instruction"] == (
        "Keep the hotel message concise."
    )
    assert candidate.metadata_json["source_query_preview_id"] is None
    assert candidate.metadata_json["generated_sql_summary"] is None

    _assert_no_forbidden_terms(candidate.to_public_values())
    _assert_no_forbidden_terms(candidate.metadata_json)


@pytest.mark.parametrize(
    ("channel", "required_fields", "empty_fields"),
    (
        (
            ContentChannel.EMAIL,
            ("subject", "preheader", "body", "cta", "landing_url"),
            ("title", "message", "image_prompt"),
        ),
        (
            ContentChannel.SMS,
            ("message", "landing_url"),
            ("subject", "preheader", "title", "cta", "image_prompt"),
        ),
        (
            ContentChannel.ONSITE_BANNER,
            ("title", "body", "cta", "image_prompt", "landing_url"),
            ("subject", "preheader", "message"),
        ),
    ),
)
def test_generation_channel_contract_fields_are_stable(
    channel: ContentChannel,
    required_fields: tuple[str, ...],
    empty_fields: tuple[str, ...],
) -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_builder=StaticGenerationInputBuilder(channel),
    )

    response = service.generate(_generation_request(content_option_count=1))

    candidate = response.content_candidates[0].model_dump()
    assert candidate["channel"] == channel.value
    assert candidate["attribution"]["promotion_channel"] == channel.value
    assert candidate["attribution"]["target_url"] == "https://demo-stay.example.com/summer"
    assert candidate["source"]["creative_format"] == candidate["creative_format"]
    if channel == ContentChannel.EMAIL:
        assert candidate["source"]["subject"]
        assert candidate["source"]["preheader"]
        assert candidate["source"]["text_body"]
        assert candidate["artifact"]["artifact_status"] in {"pending", "published", "failed"}
    elif channel == ContentChannel.SMS:
        assert candidate["source"]["message"]
        assert candidate["artifact"]["artifact_status"] == "not_required"
    else:
        assert candidate["source"]["width"] == 320
        assert candidate["source"]["height"] == 100
        assert candidate["artifact"]["artifact_status"] in {"pending", "published", "failed"}

    saved_candidate = content_candidate_repository.saved[0]
    assert saved_candidate.channel == channel
    assert saved_candidate.status == "draft"


def test_service_repo_does_not_duplicate_data_source_schema() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    schema_files = [
        path.relative_to(repo_root)
        for path in repo_root.rglob("schema.sql")
        if ".git" not in path.parts and ".venv" not in path.parts
    ]

    assert schema_files == []


def _generation_client(service: GenerationService) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_generation_service] = lambda: service
    return TestClient(app)


def _generation_request_payload(*, content_option_count: int) -> dict[str, object]:
    return {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "content_option_count": content_option_count,
        "operator_instruction": "Keep the hotel message concise.",
    }


def _generation_request(*, content_option_count: int) -> GenerationRequest:
    return GenerationRequest.model_validate(
        _generation_request_payload(content_option_count=content_option_count)
    )


def _assert_no_forbidden_terms(value: Any) -> None:
    text = " ".join(_collect_strings(value)).lower()
    for term in FORBIDDEN_PUBLIC_TERMS:
        assert term not in text


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, dict):
        strings: list[str] = []
        for key, child in value.items():
            strings.append(str(key))
            strings.extend(_collect_strings(child))
        return strings
    if isinstance(value, list):
        strings = []
        for child in value:
            strings.extend(_collect_strings(child))
        return strings
    if isinstance(value, tuple):
        strings = []
        for child in value:
            strings.extend(_collect_strings(child))
        return strings
    if isinstance(value, str):
        return [value]
    return []


class FakeGenerationRunRepository:
    def __init__(self) -> None:
        self.saved: list[GenerationRunRecord] = []

    def create(self, record: GenerationRunRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"generation_id": record.generation_id}

    def list_ids_by_promotion(self, promotion_id: str) -> list[str]:
        return [
            generation_run.generation_id
            for generation_run in self.saved
            if generation_run.promotion_id == promotion_id
        ]


class FakeContentCandidateRepository:
    def __init__(self) -> None:
        self.saved: list[ContentCandidateRecord] = []

    def create(self, record: ContentCandidateRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"content_id": record.content_id}


class StaticGenerationInputBuilder:
    def __init__(self, channel: ContentChannel) -> None:
        self._channel = channel

    def build(
        self,
        *,
        request: GenerationRequest,
        promotion: PromotionPromptInput,
        target_segments: list[TargetSegmentPromptInput],
    ) -> list[GenerationPromptInput]:
        del target_segments
        return [
            GenerationPromptInput(
                request=request,
                promotion=PromotionPromptInput(
                    project_id=promotion.project_id,
                    campaign_id=promotion.campaign_id,
                    promotion_id=promotion.promotion_id,
                    channel=self._channel,
                    goal_metric=promotion.goal_metric,
                    goal_target_value=promotion.goal_target_value,
                    goal_basis=promotion.goal_basis,
                    message_brief=promotion.message_brief,
                    landing_url=promotion.landing_url,
                ),
                target_segment=_target_segment_input(),
            )
        ]


class ImageUrlContentGenerator:
    version = "dec-c5.image-url-test.v1"

    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> GeneratedContent:
        del prompt_input, prompt_result, option_index
        return GeneratedContent(
            title="Hotel rooms ready this weekend",
            body="Compare refundable hotel stays before rooms run out.",
            cta="View hotel deals",
            image_prompt="bright hotel suite banner with summer travel layout",
            image_url=IMAGE_URL,
            landing_url="https://demo-stay.example.com/summer",
        )


def _target_segment_input() -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id="analysis_banner_001",
        promotion_id="promo_banner_001",
        segment_id="seg_repeat_hotel_no_booking",
        segment_name="Repeat hotel viewers without booking",
        content_slug="repeat_hotel",
        content_brief_json={
            "message_direction": "Highlight refundable hotel stays.",
            "keywords": ["refundable stays", "hotel deals"],
            "top_common_features": [
                "same_hotel_repeat_view",
                "near_checkin",
            ],
            "booking_conversion_rate": "0.018",
            "comparison_group_conversion_rate": "0.034",
        },
        segment_vector_id="segvec_repeat_hotel_v1",
        estimated_size=1342,
        priority="high",
        natural_language_query="hotel visitors without booking",
        generated_sql=None,
        sample_ratio="0.018000",
        source="system_default",
        query_preview_id=None,
    )
