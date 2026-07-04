from decimal import Decimal

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
    PromotionPromptRecord,
    PromotionTargetSegmentRecord,
)
from app.generation.schemas import (
    ContentChannel,
    GenerationRequest,
)
from app.generation.service import GenerationService
from app.generation.service import NextLoopFocusGenerationRequest


class FakeGenerationRunRepository:
    def __init__(self) -> None:
        self.saved: list[GenerationRunRecord] = []

    def create(self, record: GenerationRunRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"generation_id": record.generation_id}


class FakeContentCandidateRepository:
    def __init__(self) -> None:
        self.saved: list[ContentCandidateRecord] = []

    def create(self, record: ContentCandidateRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"content_id": record.content_id}


class FakePromotionRepository:
    def __init__(self, promotion: PromotionPromptRecord | None = None) -> None:
        self.promotion = promotion or promotion_prompt_record()

    def get_for_generation(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionPromptRecord | None:
        if (
            project_id,
            campaign_id,
            promotion_id,
        ) != (
            self.promotion.project_id,
            self.promotion.campaign_id,
            self.promotion.promotion_id,
        ):
            return None
        return self.promotion


class FakePromotionTargetSegmentRepository:
    def __init__(self, segments: list[PromotionTargetSegmentRecord]) -> None:
        self.segments = segments

    def list_for_analysis(self, analysis_id: str) -> list[PromotionTargetSegmentRecord]:
        return [
            segment for segment in self.segments if segment.analysis_id == analysis_id
        ]


class FakeGenerationRunReader:
    def __init__(self, *, content_option_count: int = 3) -> None:
        self.content_option_count = content_option_count

    def get_by_id(self, generation_id: str) -> dict[str, object] | None:
        if generation_id != "generation_banner_001":
            return None
        return {
            "generation_id": generation_id,
            "content_option_count": self.content_option_count,
        }


def generation_request(
    *,
    content_option_count: int = 2,
    operator_instruction: str | None = "Make the banner direct and concise.",
) -> GenerationRequest:
    return GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=content_option_count,
        operator_instruction=operator_instruction,
    )


def promotion_prompt_record() -> PromotionPromptRecord:
    return PromotionPromptRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel="onsite_banner",
        goal_metric="booking_conversion_rate",
        goal_target_value=Decimal("0.030000"),
        goal_basis="all_segments",
        message_brief="Drive hotel booking conversion for summer stays.",
        landing_url="https://demo-stay.example.com/summer",
    )


def target_segment_record(
    segment_id: str,
    *,
    analysis_id: str = "analysis_banner_001_loop_2",
) -> PromotionTargetSegmentRecord:
    return PromotionTargetSegmentRecord(
        analysis_id=analysis_id,
        promotion_id="promo_banner_001",
        segment_id=segment_id,
        segment_name=segment_id.replace("_", " ").title(),
        content_brief_json={
            "message_direction": "Use a focused hotel booking message.",
            "keywords": ["hotel booking", "breakfast"],
        },
        data_evidence_json={
            "source": "system_default",
            "sample_ratio": "0.020000",
        },
        segment_vector_id=f"segvec_{segment_id}_v1",
        estimated_size=1200,
        priority="high",
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
    assert generation_run.output_json["report_version"] == "dec-c4.v1"
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
        "prompt_builder": "dec-c2.v1",
        "content_generator": "dec-c3.deterministic.v1",
        "report_builder": "dec-c4.v1",
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
    assert "title, body, cta, image_prompt, landing_url" in (
        first_candidate.generation_prompt
    )
    assert first_candidate.reason_summary
    assert first_candidate.message_strategy
    assert first_candidate.data_evidence_json["segment_id"] == (
        "seg_repeat_hotel_no_booking"
    )
    assert first_candidate.metadata_json["content_id"] == first_candidate.content_id
    assert first_candidate.metadata_json["channel"] == "onsite_banner"
    assert first_candidate.metadata_json["report_version"] == "dec-c4.v1"
    assert first_candidate.metadata_json["prompt_builder_version"] == "dec-c2.v1"
    assert (
        first_candidate.metadata_json["content_generator_version"]
        == "dec-c3.deterministic.v1"
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
    assert first_candidate.image_url is None
    assert first_candidate.metadata_json["image_url"] is None
    assert first_candidate.metadata_json["source_query_preview_id"] is None
    assert first_candidate.metadata_json["generated_sql_summary"] is None


def test_generation_service_can_generate_response_without_repositories() -> None:
    service = GenerationService()

    response = service.generate(generation_request(content_option_count=1))

    assert response.generation_id == "generation_banner_001"
    assert len(response.content_candidates) == 1
    assert response.content_candidates[0].content_option_id == (
        "banner_repeat_hotel_option_001"
    )


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
        candidate.content_id for candidate in response.content_candidates
    } == {
        "content_banner_failed_focus_001",
        "content_banner_failed_focus_002",
        "content_banner_failed_focus_003",
    }


def test_generation_service_generates_next_loop_focus_candidates_as_draft() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
        promotion_repository=FakePromotionRepository(),
        promotion_target_segment_repository=FakePromotionTargetSegmentRepository(
            [
                target_segment_record("seg_near_checkin"),
                target_segment_record("seg_mobile_user"),
            ]
        ),
        generation_run_reader=FakeGenerationRunReader(content_option_count=3),
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
    assert generation_run.input_json["content_option_count"] == 3
    assert generation_run.input_json["next_loop"] == {
        "loop_count": 2,
        "source_promotion_run_id": "prun_banner_001_loop_1",
        "source_generation_id": "generation_banner_001",
        "focus_segment_ids": ["seg_near_checkin"],
    }
    assert len(content_candidate_repository.saved) == 3
    assert {
        candidate.segment_id for candidate in content_candidate_repository.saved
    } == {"seg_near_checkin"}
    assert {
        candidate.status for candidate in content_candidate_repository.saved
    } == {"draft"}


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
        "report_version": "dec-c4.v1",
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
        "prompt_builder": "dec-c2.v1",
        "content_generator": "dec-c3.deterministic.v1",
        "report_builder": "dec-c4.v1",
        "error_code": "content_generation_failed",
    }
    assert content_candidate_repository.saved == []
    assert "secret" not in str(generation_run.output_json)
    assert "secret" not in str(generation_run.generation_report_json)


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
    assert email_candidate.subject
    assert email_candidate.preheader
    assert email_candidate.body
    assert email_candidate.cta
    assert email_candidate.landing_url

    sms_candidate = sms_response.content_candidates[0]
    assert sms_candidate.message
    assert sms_candidate.landing_url


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


def target_segment_input(
    *,
    index: int = 1,
    content_slug: str = "repeat_hotel",
    generated_sql: str | None = None,
    query_preview_id: str | None = None,
) -> TargetSegmentPromptInput:
    return TargetSegmentPromptInput(
        analysis_id="analysis_banner_001",
        promotion_id="promo_banner_001",
        segment_id=f"seg_{content_slug}_{index:03d}",
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
    )
