from dataclasses import replace

import pytest

from app.generation.generator import GeneratedContent
from app.generation.image_tasks import ImageGenerationJob
from app.generation.prompt_builder import (
    CANDIDATE_STRATEGY_BLOCK_HEADER,
    GenerationPromptInput,
    PromotionPromptInput,
    PromptBuildResult,
    TargetSegmentPromptInput,
)
from app.generation.repositories import (
    ContentCandidateRecord,
    GenerationRunRecord,
)
from app.generation.schemas import (
    ContentChannel,
    GenerationRequest,
)
from app.generation.service import (
    DEMO_DEFAULT_LANDING_URL,
    DEMO_PROJECT_ID,
    GenerationInputUnavailable,
    GenerationService,
    NextLoopFocusGenerationRequest,
)


class FakeGenerationRunRepository:
    def __init__(self, existing_generation_ids: list[str] | None = None) -> None:
        self.existing_generation_ids = existing_generation_ids or []
        self.saved: list[GenerationRunRecord] = []

    def create(self, record: GenerationRunRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"generation_id": record.generation_id}

    def list_ids_by_promotion(self, promotion_id: str) -> list[str]:
        return [
            *self.existing_generation_ids,
            *[
                generation_run.generation_id
                for generation_run in self.saved
                if generation_run.promotion_id == promotion_id
            ],
        ]


class FakeContentCandidateRepository:
    def __init__(self) -> None:
        self.saved: list[ContentCandidateRecord] = []

    def create(self, record: ContentCandidateRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"content_id": record.content_id}


class FakeImageGenerationScheduler:
    def __init__(self) -> None:
        self.jobs: list[ImageGenerationJob] = []

    def enqueue(self, job: ImageGenerationJob) -> None:
        self.jobs.append(job)


def generation_request(
    *,
    project_id: str = "hotel-client-a",
    content_option_count: int = 2,
    operator_instruction: str | None = "Make the banner direct and concise.",
) -> GenerationRequest:
    return GenerationRequest(
        project_id=project_id,
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=content_option_count,
        operator_instruction=operator_instruction,
    )


def test_generation_service_persists_run_and_content_candidates() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
    )

    response = service.generate(generation_request(content_option_count=2))

    assert response.generation_id == "generation_banner_001"
    assert response.promotion_id == "promo_banner_001"
    assert response.status == "completed"
    assert len(response.content_candidates) == 2

    assert len(generation_run_repository.saved) == 1
    generation_run = generation_run_repository.saved[0]
    assert generation_run.generation_id == response.generation_id
    assert generation_run.project_id == "hotel-client-a"
    assert generation_run.status == "completed"
    assert generation_run.input_json == {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "content_option_count": 2,
        "operator_instruction": "Make the banner direct and concise.",
        "target_segment_ids": ["seg_repeat_hotel_no_booking"],
        "channel": "onsite_banner",
    }
    assert generation_run.output_json is not None
    assert generation_run.output_json["report_version"] == "dec-c4.v3"
    assert generation_run.output_json["content_candidate_ids"] == [
        "content_banner_repeat_hotel_001",
        "content_banner_repeat_hotel_002",
    ]
    assert generation_run.output_json["generation_summary"] == {
        "status": "completed",
        "content_candidate_count": 2,
        "target_segment_count": 1,
    }
    assert generation_run.output_json["segment_summaries"][0][
        "segment_id"
    ] == "seg_repeat_hotel_no_booking"
    assert generation_run.output_json["segment_summaries"][0][
        "content_candidate_ids"
    ] == [
        "content_banner_repeat_hotel_001",
        "content_banner_repeat_hotel_002",
    ]
    assert generation_run.generation_report_json == {
        "status": "completed",
        "content_candidate_count": 2,
        "target_segment_count": 1,
        "prompt_builder": "dec-c2.v4",
        "content_generator": "dec-c3.deterministic.v4",
        "report_builder": "dec-c4.v3",
    }

    assert len(content_candidate_repository.saved) == 2
    first_candidate = content_candidate_repository.saved[0]
    assert first_candidate.content_id == "content_banner_repeat_hotel_001"
    assert first_candidate.content_option_id == "banner_repeat_hotel_option_001"
    assert first_candidate.generation_id == response.generation_id
    assert first_candidate.project_id == "hotel-client-a"
    assert first_candidate.channel == ContentChannel.ONSITE_BANNER
    assert first_candidate.generation_prompt
    assert "Required output fields" in first_candidate.generation_prompt
    assert "title, body, cta, image_prompt" in first_candidate.generation_prompt
    assert first_candidate.reason_summary
    assert first_candidate.message_strategy
    assert first_candidate.data_evidence_json["segment_id"] == (
        "seg_repeat_hotel_no_booking"
    )
    assert first_candidate.metadata_json["content_id"] == first_candidate.content_id
    assert first_candidate.metadata_json["channel"] == "onsite_banner"
    assert first_candidate.metadata_json["report_version"] == "dec-c4.v3"
    assert first_candidate.metadata_json["prompt_builder_version"] == "dec-c2.v4"
    assert (
        first_candidate.metadata_json["content_generator_version"]
        == "dec-c3.deterministic.v4"
    )
    assert first_candidate.metadata_json["reason_summary"] == (
        first_candidate.reason_summary
    )
    assert first_candidate.metadata_json["message_strategy"] == (
        first_candidate.message_strategy
    )
    assert first_candidate.metadata_json["data_evidence"] == (
        first_candidate.data_evidence_json
    )
    assert first_candidate.metadata_json["operator_instruction"] == (
        "Make the banner direct and concise."
    )
    assert first_candidate.metadata_json["data_evidence"]["sample_size"] == 1342
    assert first_candidate.metadata_json["data_evidence"]["sample_ratio"] == 0.018
    assert first_candidate.title == "이번 주말 호텔 확인"
    assert first_candidate.body == (
        "관심 호텔의 객실 정보와 예약 조건을 지금 비교해보세요."
    )
    assert first_candidate.cta == "호텔 정보 보기"
    fallback_copy = " ".join(
        value
        for value in (
            first_candidate.title,
            first_candidate.body,
            first_candidate.cta,
        )
        if value
    )
    for unsupported_claim in ("환불 가능", "객실 마감", "특가"):
        assert unsupported_claim not in fallback_copy
    assert first_candidate.image_url is None
    assert first_candidate.metadata_json["image_url"] is None
    assert first_candidate.metadata_json["source_query_preview_id"] is None
    assert first_candidate.metadata_json["generated_sql_summary"] is None


def test_generation_service_can_generate_response_without_repositories() -> None:
    service = GenerationService()

    response = service.generate(generation_request(content_option_count=1))

    assert response.generation_id == "generation_banner_001"
    assert len(response.content_candidates) == 1
    assert response.content_candidates[0].attribution.content_option_id == (
        "banner_repeat_hotel_option_001"
    )


def test_generation_service_enqueues_deferred_banner_image_generation() -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    image_generation_scheduler = FakeImageGenerationScheduler()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        image_generation_scheduler=image_generation_scheduler,
    )

    response = service.generate(generation_request(content_option_count=2))

    assert [candidate.artifact.artifact_status for candidate in response.content_candidates] == [
        "published",
        "published",
    ]
    assert image_generation_scheduler.jobs == [
        ImageGenerationJob(
            content_id="content_banner_repeat_hotel_001",
            image_prompt=content_candidate_repository.saved[0].image_prompt or "",
        ),
        ImageGenerationJob(
            content_id="content_banner_repeat_hotel_002",
            image_prompt=content_candidate_repository.saved[1].image_prompt or "",
        ),
    ]


def test_generation_service_uses_new_generation_id_for_regeneration() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
    )

    first_response = service.generate(generation_request(content_option_count=1))
    second_response = service.generate(generation_request(content_option_count=1))

    assert first_response.generation_id == "generation_banner_001"
    assert second_response.generation_id == "generation_banner_001_run_2"
    assert [
        generation_run.generation_id
        for generation_run in generation_run_repository.saved
    ] == [
        "generation_banner_001",
        "generation_banner_001_run_2",
    ]
    assert [
        candidate.generation_id for candidate in content_candidate_repository.saved
    ] == [
        "generation_banner_001",
        "generation_banner_001_run_2",
    ]
    assert [
        candidate.content_id for candidate in content_candidate_repository.saved
    ] == [
        "content_banner_repeat_hotel_001",
        "content_banner_repeat_hotel_run_2_001",
    ]
    assert second_response.content_candidates[0].attribution.content_option_id == (
        "banner_repeat_hotel_run_2_option_001"
    )


def test_generation_service_skips_existing_regeneration_ids() -> None:
    generation_run_repository = FakeGenerationRunRepository(
        existing_generation_ids=[
            "generation_banner_001",
            "generation_banner_001_run_2",
        ]
    )
    service = GenerationService(generation_run_repository=generation_run_repository)

    response = service.generate(generation_request(content_option_count=1))

    assert response.generation_id == "generation_banner_001_run_3"


def test_generation_service_requires_confirmed_target_segments_from_reader() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        generation_input_reader=StaticGenerationInputReader([]),
    )

    with pytest.raises(GenerationInputUnavailable, match="promotion_target_segments"):
        service.generate(generation_request(content_option_count=1))

    assert generation_run_repository.saved == []
    assert content_candidate_repository.saved == []


def test_generation_service_creates_candidates_for_each_segment_option() -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_builder=StaticGenerationInputBuilder(
            [
                target_segment_input(index=1, content_slug="repeat_hotel"),
                target_segment_input(index=2, content_slug="family_trip"),
                target_segment_input(index=3, content_slug="weekday_business"),
                target_segment_input(index=4, content_slug="spa_interest"),
            ]
        ),
    )

    response = service.generate(generation_request(content_option_count=3))

    assert len(response.content_candidates) == 12
    assert len(content_candidate_repository.saved) == 12
    content_ids = {
        candidate.content_id for candidate in content_candidate_repository.saved
    }
    option_ids = {
        candidate.content_option_id for candidate in content_candidate_repository.saved
    }
    assert len(content_ids) == 12
    assert len(option_ids) == 12
    assert "content_banner_repeat_hotel_003" in content_ids
    assert "content_banner_spa_interest_003" in content_ids


def test_generation_service_creates_candidates_for_focus_segment_only() -> None:
    service = GenerationService(
        generation_input_builder=StaticGenerationInputBuilder(
            [target_segment_input(index=9, content_slug="failed_focus")]
        ),
    )

    response = service.generate(generation_request(content_option_count=3))

    assert len(response.content_candidates) == 3
    assert {
        candidate.attribution.content_id for candidate in response.content_candidates
    } == {
        "content_banner_failed_focus_001",
        "content_banner_failed_focus_002",
        "content_banner_failed_focus_003",
    }


def test_generation_service_generates_next_loop_focus_candidate_as_approved() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        generation_input_reader=StaticGenerationInputReader(
            [
                target_segment_input(
                    analysis_id="analysis_banner_001_loop_2",
                    segment_id="seg_near_checkin",
                    content_slug="near_checkin",
                ),
                target_segment_input(
                    analysis_id="analysis_banner_001_loop_2",
                    segment_id="seg_mobile_user",
                    content_slug="mobile_user",
                ),
            ]
        ),
    )

    result = service.generate_focus(
        NextLoopFocusGenerationRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            analysis_id="analysis_banner_001_loop_2",
            focus_segment_ids=["seg_near_checkin"],
            loop_count=2,
            source_promotion_run_id="prun_banner_001_loop_1",
            source_generation_id="generation_banner_001",
            operator_instruction="Stress breakfast.",
        )
    )

    assert result.generation_id == "generation_banner_001_loop_2"
    assert result.generated_segment_ids == ["seg_near_checkin"]
    assert result.status == "completed"
    assert len(generation_run_repository.saved) == 1
    generation_run = generation_run_repository.saved[0]
    assert generation_run.input_json["target_segment_ids"] == ["seg_near_checkin"]
    assert generation_run.input_json["content_option_count"] == 1
    assert generation_run.input_json["next_loop"] == {
        "loop_count": 2,
        "source_promotion_run_id": "prun_banner_001_loop_1",
        "source_generation_id": "generation_banner_001",
        "focus_segment_ids": ["seg_near_checkin"],
    }
    assert len(content_candidate_repository.saved) == 1
    assert {
        candidate.segment_id for candidate in content_candidate_repository.saved
    } == {"seg_near_checkin"}
    assert {
        candidate.status for candidate in content_candidate_repository.saved
    } == {"approved"}
    candidate = content_candidate_repository.saved[0]
    assert candidate.content_id == "content_banner_near_checkin_loop_2_001"
    assert candidate.content_option_id == "banner_near_checkin_loop_2_option_001"


def test_generation_service_bounds_long_next_loop_content_identifiers() -> None:
    segment_id = (
        "seg_ai_raw_promo_864e3031_5e33_4715_ad00_17_1_"
        "target_destination_affinity_9e2a5d129c"
    )
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_reader=StaticGenerationInputReader(
            [
                replace(
                    target_segment_input(
                        analysis_id="analysis_banner_001_loop_2",
                        segment_id=segment_id,
                    ),
                    content_slug=None,
                )
            ],
            channel=ContentChannel.EMAIL,
        ),
    )

    result = service.generate_focus(
        NextLoopFocusGenerationRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            analysis_id="analysis_banner_001_loop_2",
            focus_segment_ids=[segment_id],
            loop_count=2,
            source_promotion_run_id="prun_banner_001_loop_1",
            source_generation_id="generation_banner_001",
        )
    )

    assert result.status == "completed"
    candidate = content_candidate_repository.saved[0]
    assert candidate.content_id == (
        "content_email_ai_raw_promo_864e3031_5e33_4715_ad00_17_1_"
        "target_destination_affi_6c87bac404eb7fd3_001"
    )
    assert candidate.content_option_id == (
        "email_ai_raw_promo_864e3031_5e33_4715_ad00_17_1_"
        "target_destination_affi_6c87bac404eb7fd3_option_001"
    )
    assert len(candidate.content_id) == 100
    assert len(candidate.content_option_id) == 99


def test_generation_service_focus_generation_bypasses_confirmed_status_filter() -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_reader=StaticGenerationInputReader(
            [],
            focus_target_segments=[
                target_segment_input(
                    analysis_id="analysis_banner_001_loop_2",
                    segment_id="seg_near_checkin",
                    content_slug="near_checkin",
                    status="planned",
                )
            ],
        ),
    )

    result = service.generate_focus(
        NextLoopFocusGenerationRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            analysis_id="analysis_banner_001_loop_2",
            focus_segment_ids=["seg_near_checkin"],
            loop_count=2,
            source_promotion_run_id="prun_banner_001_loop_1",
            source_generation_id="generation_banner_001",
            operator_instruction=None,
        )
    )

    assert result.generated_segment_ids == ["seg_near_checkin"]
    assert len(content_candidate_repository.saved) == 1
    candidate = content_candidate_repository.saved[0]
    assert candidate.segment_id == "seg_near_checkin"
    assert candidate.status == "approved"
    assert candidate.metadata_json["data_evidence"]["target_segment_status"] == (
        "planned"
    )


def test_generation_service_records_failed_run_when_generator_fails() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        content_generator=FailingContentGenerator(),
    )

    response = service.generate(generation_request(content_option_count=2))

    assert response.status == "failed"
    assert response.content_candidates == []
    assert len(generation_run_repository.saved) == 1
    generation_run = generation_run_repository.saved[0]
    assert generation_run.status == "failed"
    assert generation_run.output_json == {
        "report_version": "dec-c4.v3",
        "content_candidate_ids": [],
        "generation_summary": {
            "status": "failed",
            "content_candidate_count": 0,
            "target_segment_count": 1,
        },
        "segment_summaries": [],
        "content_report_summaries": [],
        "error_code": "content_generation_failed",
    }
    assert generation_run.generation_report_json == {
        "status": "failed",
        "content_candidate_count": 0,
        "target_segment_count": 1,
        "prompt_builder": "dec-c2.v4",
        "content_generator": "dec-c3.deterministic.v4",
        "report_builder": "dec-c4.v3",
        "error_code": "content_generation_failed",
    }
    assert content_candidate_repository.saved == []
    assert "secret" not in str(generation_run.output_json)
    assert "secret" not in str(generation_run.generation_report_json)


def test_generation_service_records_validation_error_detail() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        content_generator=MissingImagePromptContentGenerator(),
    )

    response = service.generate(generation_request(content_option_count=1))

    assert response.status == "failed"
    assert response.content_candidates == []
    generation_run = generation_run_repository.saved[0]
    assert generation_run.generation_report_json["error_code"] == (
        "content_generation_validation_failed"
    )
    assert generation_run.generation_report_json["error_detail"] == {
        "reason": "missing_required_fields",
        "channel": "onsite_banner",
        "missing_fields": ["image_prompt"],
    }
    assert content_candidate_repository.saved == []


def test_generation_service_uses_demo_default_landing_url_when_missing() -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_reader=StaticGenerationInputReader(
            [target_segment_input()],
            landing_url=None,
        ),
    )

    response = service.generate(
        generation_request(project_id=DEMO_PROJECT_ID, content_option_count=1)
    )

    assert response.status == "completed"
    assert response.content_candidates[0].attribution.target_url == DEMO_DEFAULT_LANDING_URL
    candidate = content_candidate_repository.saved[0]
    assert candidate.landing_url == DEMO_DEFAULT_LANDING_URL
    assert f"Fixed landing URL: {DEMO_DEFAULT_LANDING_URL}" in (
        candidate.generation_prompt
    )


def test_generation_service_fails_when_non_demo_landing_url_is_missing() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        generation_input_reader=StaticGenerationInputReader(
            [target_segment_input()],
            landing_url=None,
        ),
    )

    response = service.generate(generation_request(content_option_count=1))

    assert response.status == "failed"
    assert response.content_candidates == []
    assert content_candidate_repository.saved == []
    generation_run = generation_run_repository.saved[0]
    assert generation_run.status == "failed"
    assert generation_run.output_json is not None
    assert (
        generation_run.output_json["error_code"]
        == "content_generation_validation_failed"
    )


def test_generation_service_saves_channel_specific_fields() -> None:
    email_service = GenerationService(
        generation_input_builder=StaticGenerationInputBuilder(
            [target_segment_input()],
            channel=ContentChannel.EMAIL,
        )
    )
    sms_service = GenerationService(
        generation_input_builder=StaticGenerationInputBuilder(
            [target_segment_input()],
            channel=ContentChannel.SMS,
        )
    )

    email_response = email_service.generate(generation_request(content_option_count=1))
    sms_response = sms_service.generate(generation_request(content_option_count=1))

    email_candidate = email_response.content_candidates[0]
    assert email_candidate.source.creative_format == "email_html"
    assert email_candidate.source.subject
    assert email_candidate.source.preheader
    assert email_candidate.source.text_body
    assert "호텔" in email_candidate.source.subject
    assert email_candidate.attribution.target_url

    sms_candidate = sms_response.content_candidates[0]
    assert sms_candidate.source.creative_format == "sms_text"
    assert sms_candidate.source.message
    assert "호텔" in sms_candidate.source.message
    assert "{{redirect_url}}" in sms_candidate.source.message
    assert sms_candidate.attribution.target_url


def test_generation_service_saves_source_report_references() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        generation_input_builder=StaticGenerationInputBuilder(
            [
                target_segment_input(
                    generated_sql=(
                        "SELECT user_id FROM hotel_detail_events "
                        "WHERE repeat_view_count >= 2"
                    ),
                    query_preview_id="seg_query_preview_001",
                )
            ]
        ),
    )

    service.generate(generation_request(content_option_count=1))

    candidate = content_candidate_repository.saved[0]
    metadata = candidate.metadata_json
    assert metadata["source_segment_definition_id"] == "seg_repeat_hotel_001"
    assert metadata["source_query_preview_id"] == "seg_query_preview_001"
    assert metadata["generated_sql_summary"] == (
        "SELECT user_id FROM hotel_detail_events WHERE repeat_view_count >= 2"
    )
    assert metadata["data_evidence"]["top_common_features"] == [
        "same_hotel_repeat_view",
        "near_checkin",
    ]
    assert metadata["data_evidence"]["booking_conversion_rate"] == 0.018
    assert metadata["data_evidence"][
        "comparison_group_conversion_rate"
    ] == 0.034

    output_json = generation_run_repository.saved[0].output_json
    assert output_json is not None
    assert output_json["content_report_summaries"][0][
        "generated_sql_summary"
    ] == metadata["generated_sql_summary"]
    assert output_json["segment_summaries"][0]["operator_instruction"] == (
        "Make the banner direct and concise."
    )


def test_generation_service_report_filters_behavior_metrics_from_v2_brief() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        generation_input_builder=StaticGenerationInputBuilder(
            [
                replace(
                    target_segment_input(),
                    content_brief_json={
                        "schema_version": "content_brief.v2",
                        "readiness": {
                            "level": "partial",
                            "available_sections": [
                                "fallback_guidance",
                                "audience_evidence",
                            ],
                            "missing_sections": [],
                        },
                        "fallback_guidance": {
                            "message_direction": "Use a hotel booking message.",
                            "keywords": ["must_not_be_common_feature"],
                            "source": "legacy_segment_content_hints",
                        },
                        "top_common_features": ["must_not_pass"],
                        "booking_conversion_rate": 0.99,
                        "comparison_group_conversion_rate": 0.98,
                        "audience_evidence": {
                            "primary_signals": [
                                "same_hotel_repeat_view",
                                "near_checkin",
                            ],
                            "score_components": {
                                "promotion_cluster_similarity": 0.92,
                            },
                            "promotion_vector_basis": {
                                "channel": "onsite_banner",
                                "goal_metric": "booking_conversion_rate",
                            },
                            "promotion_matched_features": [
                                "same_hotel_repeat_view",
                                "near_checkin",
                            ],
                            "behavior_metrics": {
                                "booking_conversion_rate": 0.018,
                            },
                        },
                    },
                )
            ]
        ),
    )

    service.generate(generation_request(content_option_count=1))

    metadata = content_candidate_repository.saved[0].metadata_json
    assert metadata["content_brief_readiness"] == {
        "level": "evidence_ready",
        "missing_sections": [],
        "available_sections": ["fallback_guidance", "audience_evidence"],
    }
    assert metadata["fallback_guidance_present"] is True
    assert metadata["fallback_guidance_used"] is False
    assert metadata["data_evidence"]["fallback_guidance_present"] is True
    assert metadata["data_evidence"]["fallback_guidance_used"] is False
    assert "content_brief_keywords" not in metadata["data_evidence"]
    assert metadata["data_evidence"]["audience_evidence"] == {
        "primary_signals": ["same_hotel_repeat_view", "near_checkin"],
        "score_components": {
            "promotion_cluster_similarity": 0.92,
        },
        "promotion_vector_basis": {
            "channel": "onsite_banner",
            "goal_metric": "booking_conversion_rate",
        },
        "promotion_matched_features": [
            "same_hotel_repeat_view",
            "near_checkin",
        ],
    }
    assert "top_common_features" not in metadata["data_evidence"]
    assert "booking_conversion_rate" not in metadata["data_evidence"]
    assert "comparison_group_conversion_rate" not in metadata["data_evidence"]
    assert "must_not_pass" not in str(metadata)
    assert "behavior_metrics" not in str(metadata)
    assert "behavior_metrics" not in str(generation_run_repository.saved[0].output_json)


def test_generation_service_persists_candidate_specific_prompt_and_strategy_metadata() -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_builder=StaticGenerationInputBuilder(
            [
                replace(
                    target_segment_input(index=1, content_slug="jeju_ocean"),
                    segment_name="Jeju ocean hotel viewers",
                    content_brief_json={
                        "schema_version": "content_brief.v2",
                        "fallback_guidance": {
                            "message_direction": "Use a booking reminder.",
                            "keywords": ["should-not-be-used"],
                        },
                        "hotel_profile": {
                            "hotel_cluster": "jeju_ocean",
                            "booking_count": 120,
                        },
                        "audience_evidence": {
                            "primary_signals": ["jeju_destination_search"],
                            "score_components": {"final_score": 0.91},
                        },
                    },
                ),
                replace(
                    target_segment_input(index=2, content_slug="seoul_business"),
                    segment_name="Seoul business hotel viewers",
                    content_brief_json={
                        "schema_version": "content_brief.v2",
                        "fallback_guidance": {
                            "message_direction": "Use a booking reminder.",
                            "keywords": ["should-not-be-used"],
                        },
                        "hotel_profile": {
                            "hotel_cluster": "seoul_business",
                            "booking_count": 240,
                        },
                        "audience_evidence": {
                            "primary_signals": ["weekday_business_search"],
                            "score_components": {"final_score": 0.87},
                        },
                    },
                ),
            ]
        ),
    )

    response = service.generate(generation_request(content_option_count=1))

    assert len(response.content_candidates) == 2
    assert len(content_candidate_repository.saved) == 2
    candidates_by_segment = {
        candidate.segment_id: candidate
        for candidate in content_candidate_repository.saved
    }
    jeju_candidate = candidates_by_segment["seg_jeju_ocean_001"]
    seoul_candidate = candidates_by_segment["seg_seoul_business_002"]

    assert jeju_candidate.generation_prompt != seoul_candidate.generation_prompt
    assert "jeju_destination_search" in jeju_candidate.generation_prompt
    assert "hotel_cluster=jeju_ocean" in jeju_candidate.generation_prompt
    assert "weekday_business_search" in seoul_candidate.generation_prompt
    assert "hotel_cluster=seoul_business" in seoul_candidate.generation_prompt

    for candidate in (jeju_candidate, seoul_candidate):
        metadata = candidate.metadata_json
        assert metadata["content_brief_readiness"]["level"] == "evidence_ready"
        assert metadata["fallback_guidance_present"] is True
        assert metadata["fallback_guidance_used"] is False
        assert candidate.data_evidence_json == metadata["data_evidence"]
        assert "content_brief_keywords" not in candidate.data_evidence_json

    assert jeju_candidate.data_evidence_json["audience_evidence"] != (
        seoul_candidate.data_evidence_json["audience_evidence"]
    )


def test_generation_service_applies_candidate_strategy_to_content_and_metadata() -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_builder=StaticGenerationInputBuilder(
            [
                replace(
                    target_segment_input(),
                    content_brief_json={
                        "schema_version": "content_brief.v2",
                        "fallback_guidance": {
                            "message_direction": "Use a hotel booking message.",
                            "keywords": ["hotel booking"],
                        },
                        "hotel_profile": {
                            "event_count": 5000,
                            "booking_count": 120,
                        },
                        "audience_evidence": {
                            "primary_signals": ["near_checkin", "mobile"],
                            "score_components": {"final_score": 0.91},
                            "promotion_matched_features": [
                                "free_cancellation",
                                "breakfast_included",
                            ],
                        },
                    },
                )
            ]
        ),
    )

    service.generate(generation_request(content_option_count=2))

    first_candidate, second_candidate = content_candidate_repository.saved
    first_metadata = first_candidate.metadata_json
    second_metadata = second_candidate.metadata_json

    assert first_metadata["brief_fingerprint"] == second_metadata[
        "brief_fingerprint"
    ]
    assert first_metadata["brief_fingerprint"].startswith("sha256:")
    assert first_metadata["strategy_key"] == "booking_confidence__near_checkin"
    assert second_metadata["strategy_key"] == "booking_confidence__mobile"
    assert first_metadata["evidence_refs"] == [
        "primary_signals[0]",
        "promotion_matched_features[0]",
    ]
    assert second_metadata["evidence_refs"] == [
        "primary_signals[1]",
        "promotion_matched_features[1]",
    ]
    assert first_metadata["evidence_refs"] == first_metadata["strategy_plan"][
        "evidence_refs"
    ]
    assert first_metadata["strategy_plan"]["benefit_focus"] == []
    assert second_metadata["strategy_plan"]["benefit_focus"] == []
    assert first_metadata["missing_sections"] == []
    assert first_candidate.message_strategy == second_candidate.message_strategy

    first_base, first_strategy = first_candidate.generation_prompt.split(
        CANDIDATE_STRATEGY_BLOCK_HEADER,
        maxsplit=1,
    )
    second_base, second_strategy = second_candidate.generation_prompt.split(
        CANDIDATE_STRATEGY_BLOCK_HEADER,
        maxsplit=1,
    )
    assert first_base == second_base
    assert first_strategy != second_strategy

    assert first_candidate.body is not None
    assert second_candidate.body is not None
    assert "체크인 일정이 가까운 고객" in first_candidate.body
    assert "모바일로 호텔을 찾는 고객" in second_candidate.body
    assert first_candidate.image_prompt is not None
    assert second_candidate.image_prompt is not None
    assert "generic hotel booking travel scene" in first_candidate.image_prompt
    assert "traveler reviewing an accommodation booking" in (
        second_candidate.image_prompt
    )
    assert "goal_metric=booking_conversion_rate" in first_candidate.image_prompt
    assert "Audience focus: near_checkin, free_cancellation" in (
        first_candidate.image_prompt
    )
    assert "Verified hotel visual context: none" in first_candidate.image_prompt
    assert "no visible text" in first_candidate.image_prompt


def test_generation_service_allows_repeated_strategy_and_content_when_evidence_is_sparse() -> None:
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        content_candidate_repository=content_candidate_repository,
        generation_input_builder=StaticGenerationInputBuilder(
            [
                replace(
                    target_segment_input(),
                    content_brief_json={
                        "schema_version": "content_brief.v2",
                        "audience_evidence": {
                            "primary_signals": ["near_checkin"],
                            "score_components": {"final_score": 0.91},
                        },
                    },
                )
            ]
        ),
    )

    response = service.generate(generation_request(content_option_count=4))

    candidates = content_candidate_repository.saved
    assert len(response.content_candidates) == 4
    assert len(candidates) == 4
    assert {candidate.metadata_json["strategy_key"] for candidate in candidates} == {
        "booking_confidence__near_checkin"
    }
    assert {
        tuple(candidate.metadata_json["evidence_refs"])
        for candidate in candidates
    } == {("primary_signals[0]",)}
    assert all(
        candidate.metadata_json["strategy_plan"]["benefit_focus"] == []
        for candidate in candidates
    )
    assert candidates[0].title == candidates[3].title
    assert candidates[0].body == candidates[3].body
    assert candidates[0].image_prompt == candidates[3].image_prompt


class StaticGenerationInputBuilder:
    def __init__(
        self,
        target_segments: list[TargetSegmentPromptInput],
        *,
        channel: ContentChannel = ContentChannel.ONSITE_BANNER,
    ) -> None:
        self._target_segments = target_segments
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
                target_segment=target_segment,
            )
            for target_segment in self._target_segments
        ]


class StaticGenerationInputReader:
    def __init__(
        self,
        target_segments: list[TargetSegmentPromptInput],
        *,
        channel: ContentChannel = ContentChannel.ONSITE_BANNER,
        landing_url: str | None = "https://demo-stay.example.com/summer",
        focus_target_segments: list[TargetSegmentPromptInput] | None = None,
    ) -> None:
        self._target_segments = target_segments
        self._focus_target_segments = focus_target_segments or target_segments
        self._channel = channel
        self._landing_url = landing_url

    def get_promotion_input(
        self,
        request: GenerationRequest,
    ) -> PromotionPromptInput:
        return PromotionPromptInput(
            project_id=request.project_id,
            campaign_id=request.campaign_id,
            promotion_id=request.promotion_id,
            channel=self._channel,
            goal_metric="booking_conversion_rate",
            goal_target_value="0.030000",
            goal_basis="all_segments",
            message_brief="Drive hotel booking conversion for summer stays.",
            landing_url=self._landing_url,
        )

    def list_target_segment_inputs(
        self,
        request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        del request
        return list(self._target_segments)

    def list_focus_target_segment_inputs(
        self,
        request: GenerationRequest,
    ) -> list[TargetSegmentPromptInput]:
        del request
        return list(self._focus_target_segments)


class FailingContentGenerator:
    def generate(
        self,
        *,
        prompt_input: GenerationPromptInput,
        prompt_result: PromptBuildResult,
        option_index: int,
    ) -> GeneratedContent:
        del prompt_input, prompt_result, option_index
        raise RuntimeError("provider failed with secret-token-value")


class MissingImagePromptContentGenerator:
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
            landing_url="https://demo-stay.example.com/summer",
        )


def target_segment_input(
    *,
    index: int = 1,
    analysis_id: str = "analysis_banner_001",
    segment_id: str | None = None,
    content_slug: str = "repeat_hotel",
    generated_sql: str | None = None,
    query_preview_id: str | None = None,
    status: str | None = None,
) -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id=analysis_id,
        promotion_id="promo_banner_001",
        segment_id=segment_id or f"seg_{content_slug}_{index:03d}",
        segment_name=f"Hotel audience segment {index}",
        content_slug=content_slug,
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
        segment_vector_id=f"segvec_{content_slug}_{index:03d}",
        estimated_size=1000 + index,
        priority="high",
        natural_language_query="hotel visitors without booking",
        generated_sql=generated_sql,
        sample_ratio="0.018000",
        source="system_default",
        query_preview_id=query_preview_id,
        status=status,
    )
