from __future__ import annotations

import re
import threading
from dataclasses import dataclass, replace
from decimal import Decimal
from typing import Any, Mapping, Sequence

import pytest

from app.audience_allocation import (
    ConfirmationAllocationResult,
    FinalAudienceAllocation,
)
from app.audience_contract import (
    CUSTOM_STRUCTURED_ANCHOR_POLICY_ID,
    CUSTOM_STRUCTURED_CANDIDATE_TYPE,
    CUSTOM_STRUCTURED_CONDITION_KEY,
    CUSTOM_STRUCTURED_PARAMETER_POLICY_ID,
    CUSTOM_STRUCTURED_SELECTION_POLICY_ID,
    CUSTOM_STRUCTURED_TEMPLATE_HASH,
    CUSTOM_STRUCTURED_TEMPLATE_ID,
    CUSTOM_STRUCTURED_TEMPLATE_VERSION,
    CUSTOM_STRUCTURED_WINDOW_DAYS,
    SEGMENT_AUDIENCE_CONTRACT,
)

from app.analysis.repositories import (
    BookingTrainingRecord,
    HotelMarketingProfileRecord,
    PromotionAnalysisWrite,
    PromotionRecord,
    PromotionSegmentSuggestionWrite,
    PromotionTargetSegmentWrite,
    SegmentDefinitionRecord,
    SegmentSuggestionAudienceBindingRecord,
)
from app.analysis.audience_v2 import AudienceV2Preparation
from app.analysis.audience_snapshot_repository import AudienceSnapshotBindingError
from app.analysis.segment_audience_templates import (
    RegisteredSegmentAudienceBinder,
)
from app.analysis.report_generator import SegmentSuggestionReportInput
from app.analysis.schemas import AnalysisRequest, SegmentAnalysisRequest
from app.analysis.service import (
    NextLoopFocusAnalysisRequest,
    PromotionAnalysisService,
    PromotionNotFoundError,
    SegmentSelectionError,
    _bounded_next_loop_lineage_id,
)
from app.analysis.vector_service import (
    SegmentVectorBuildRequest,
    SegmentVectorBuildResult,
)


@dataclass
class SavedAnalysis:
    analysis: PromotionAnalysisWrite | None = None
    target_segments: list[PromotionTargetSegmentWrite] | None = None
    segment_suggestions: list[PromotionSegmentSuggestionWrite] | None = None


class FakePromotionRepository:
    def __init__(self, promotion: PromotionRecord | None) -> None:
        self.promotion = promotion
        self.calls: list[Mapping[str, str]] = []

    def get_for_analysis(
        self,
        *,
        project_id: str,
        campaign_id: str,
        promotion_id: str,
    ) -> PromotionRecord | None:
        self.calls.append(
            {
                "project_id": project_id,
                "campaign_id": campaign_id,
                "promotion_id": promotion_id,
            }
        )
        return self.promotion


class FakeSegmentDefinitionRepository:
    def __init__(self, segments: list[SegmentDefinitionRecord]) -> None:
        self.segments = segments
        self.calls: list[Mapping[str, Any]] = []
        self.saved_ai_suggested: list[SegmentDefinitionRecord] = []

    def list_active(
        self,
        *,
        project_id: str,
        campaign_id: str | None = None,
        promotion_id: str | None = None,
        sources: Sequence[str] | None = None,
    ) -> list[SegmentDefinitionRecord]:
        self.calls.append(
            {
                "project_id": project_id,
                "campaign_id": campaign_id,
                "promotion_id": promotion_id,
                "sources": sources,
            }
        )
        return self.segments

    def save_ai_suggested(
        self,
        segments: Sequence[SegmentDefinitionRecord],
    ) -> None:
        self.saved_ai_suggested.extend(segments)


class FakeHotelProfileRepository:
    def __init__(
        self,
        profiles: list[HotelMarketingProfileRecord],
        *,
        user_profile_summaries: Mapping[str, HotelMarketingProfileRecord] | None = None,
        booking_training_records: list[BookingTrainingRecord] | None = None,
    ) -> None:
        self.profiles = profiles
        self.user_profile_summaries = user_profile_summaries or {}
        self.booking_training_records = booking_training_records or []
        self.calls: list[Mapping[str, str]] = []
        self.user_profile_calls: list[Mapping[str, Any]] = []
        self.training_calls: list[Mapping[str, int]] = []

    def list_marketing_profiles(
        self,
        *,
        project_id: str,
    ) -> list[HotelMarketingProfileRecord]:
        self.calls.append({"project_id": project_id})
        return self.profiles

    def summarize_user_ids(
        self,
        *,
        project_id: str,
        profile_name: str,
        user_ids: Sequence[str],
    ) -> HotelMarketingProfileRecord | None:
        self.user_profile_calls.append(
            {
                "project_id": project_id,
                "profile_name": profile_name,
                "user_ids": list(user_ids),
            }
        )
        return self.user_profile_summaries.get(profile_name)

    def list_booking_training_records(
        self,
        *,
        limit: int = 500,
    ) -> list[BookingTrainingRecord]:
        self.training_calls.append({"limit": limit})
        return self.booking_training_records


class FakePromotionAnalysisRepository:
    def __init__(
        self,
        bindings: Sequence[SegmentSuggestionAudienceBindingRecord] = (),
    ) -> None:
        self.saved = SavedAnalysis()
        self.events: list[str] = []
        self.bindings = list(bindings)

    def save_analysis(self, analysis: PromotionAnalysisWrite) -> None:
        self.saved.analysis = analysis
        self.events.append("analysis")

    def save_target_segments(
        self,
        target_segments: Sequence[PromotionTargetSegmentWrite],
    ) -> None:
        self.saved.target_segments = list(target_segments)
        self.events.append("target_segments")

    def save_segment_suggestions(
        self,
        suggestions: Sequence[PromotionSegmentSuggestionWrite],
    ) -> None:
        self.saved.segment_suggestions = list(suggestions)
        self.events.append("segment_suggestions")

    def get_latest_audience_bindings(self, **kwargs: object):
        segment_ids = set(kwargs.get("segment_ids", ()))
        return [
            binding
            for binding in self.bindings
            if not segment_ids or binding.segment_id in segment_ids
        ]


class FakeAudienceV2Coordinator:
    def __init__(self) -> None:
        self.prepare_calls: list[dict[str, object]] = []
        self.prepare_many_calls = 0

    def prepare(self, **kwargs: object) -> AudienceV2Preparation:
        self.prepare_calls.append(dict(kwargs))
        snapshot_id = str(kwargs["audience_snapshot_id"])
        segment = kwargs["segment"]
        return AudienceV2Preparation(
            audience_snapshot_id=snapshot_id,
            segment_vector_id=f"vector_{segment.segment_id}",
            vector_generation_id="generation_active",
            vector_version="hotel_behavior.v2",
            total_eligible_user_count=100,
            matching_user_count=20,
            selected_user_count=10,
            selection_method="exact",
            estimated_recall=1.0,
            recall_lower_bound=1.0,
            recall_target=1.0,
            meets_min_sample_size=False,
        )

    def prepare_many(self, **_kwargs: object):
        self.prepare_many_calls += 1
        raise AssertionError("confirmation must not search or create a new snapshot")


class FakePreparingAudienceV2Coordinator:
    def __init__(
        self,
        *,
        total_eligible_user_count: int = 100,
        matching_user_count: int = 20,
        selected_user_count: int = 10,
        meets_min_sample_size: bool = False,
        counts_by_segment: Mapping[str, tuple[int, int, int, bool]] | None = None,
    ) -> None:
        self.total_eligible_user_count = total_eligible_user_count
        self.matching_user_count = matching_user_count
        self.selected_user_count = selected_user_count
        self.meets_min_sample_size = meets_min_sample_size
        self.counts_by_segment = counts_by_segment or {}
        self.prepare_many_calls: list[dict[str, object]] = []

    def prepare(self, **kwargs: object) -> AudienceV2Preparation:
        segment = kwargs["segment"]
        eligible, matching, selected, meets_minimum = self.counts_by_segment.get(
            segment.segment_id,
            (
                self.total_eligible_user_count,
                self.matching_user_count,
                self.selected_user_count,
                self.meets_min_sample_size,
            ),
        )
        return AudienceV2Preparation(
            audience_snapshot_id=str(kwargs["audience_snapshot_id"]),
            segment_vector_id=f"vector_{segment.segment_id}",
            vector_generation_id="generation_active",
            vector_version="hotel_behavior.v2",
            total_eligible_user_count=eligible,
            matching_user_count=matching,
            selected_user_count=selected,
            selection_method="allocation_snapshot_reuse",
            estimated_recall=1.0,
            recall_lower_bound=1.0,
            recall_target=1.0,
            meets_min_sample_size=meets_minimum,
        )

    def prepare_many(self, **kwargs: object):
        self.prepare_many_calls.append(dict(kwargs))
        preparations: dict[str, AudienceV2Preparation] = {}
        for segment in kwargs["segments"]:
            eligible, matching, selected, meets_minimum = self.counts_by_segment.get(
                segment.segment_id,
                (
                    self.total_eligible_user_count,
                    self.matching_user_count,
                    self.selected_user_count,
                    self.meets_min_sample_size,
                ),
            )
            preparations[segment.segment_id] = AudienceV2Preparation(
                audience_snapshot_id=f"snapshot_{segment.segment_id}",
                segment_vector_id=f"vector_{segment.segment_id}",
                vector_generation_id="generation_active",
                vector_version="hotel_behavior.v2",
                total_eligible_user_count=eligible,
                matching_user_count=matching,
                selected_user_count=selected,
                selection_method="exact",
                estimated_recall=1.0,
                recall_lower_bound=1.0,
                recall_target=1.0,
                meets_min_sample_size=meets_minimum,
            )
        return preparations


class FakeAudienceAllocationService:
    def __init__(
        self,
        bindings: Sequence[SegmentSuggestionAudienceBindingRecord] = (),
    ) -> None:
        self.bindings = tuple(bindings)
        self.confirm_calls: list[dict[str, object]] = []
        self.preview_calls: list[dict[str, object]] = []

    def refresh_recommendation_previews(self, **kwargs: object):
        self.preview_calls.append(dict(kwargs))
        return {"allocation_previews": {}}

    def confirm_selection(self, **kwargs: object) -> ConfirmationAllocationResult:
        self.confirm_calls.append(dict(kwargs))
        segment_ids = tuple(sorted(set(kwargs["segment_ids"])))
        source_analysis_id = kwargs.get("source_analysis_id")
        if source_analysis_id is not None:
            matching = [
                binding
                for binding in self.bindings
                if binding.analysis_id == source_analysis_id
                and binding.segment_id in segment_ids
            ]
            if not matching and not self.bindings:
                matching = [
                    SegmentSuggestionAudienceBindingRecord(
                        suggestion_id=f"internal_{segment_id}",
                        analysis_id=str(source_analysis_id),
                        segment_id=segment_id,
                        audience_snapshot_id=f"snapshot_{segment_id}",
                    )
                    for segment_id in segment_ids
                ]
        else:
            matching = [
                binding
                for binding in self.bindings
                if binding.segment_id in segment_ids
            ]
        if {binding.segment_id for binding in matching} != set(segment_ids):
            missing = next(
                segment_id
                for segment_id in segment_ids
                if segment_id not in {binding.segment_id for binding in matching}
            )
            raise AudienceSnapshotBindingError(
                "recommendation snapshot binding is required",
                code="segment_audience_snapshot_binding_required",
                segment_id=missing,
            )
        analysis_ids = {binding.analysis_id for binding in matching}
        if len(analysis_ids) != 1:
            raise AudienceSnapshotBindingError(
                "selected snapshots must belong to one recommendation analysis",
                code="segment_audience_snapshot_binding_required",
            )
        source_analysis_id = next(iter(analysis_ids))
        plan_id = "allocation_test"
        allocations = {
            binding.segment_id: FinalAudienceAllocation(
                segment_id=binding.segment_id,
                source_analysis_id=binding.analysis_id,
                source_snapshot_id=binding.audience_snapshot_id,
                final_snapshot_id=f"final_{binding.audience_snapshot_id}",
                allocation_plan_id=plan_id,
                final_user_count=10,
                meets_min_sample_size=False,
                audience_status="insufficient_sample",
                exclusion_revision=1,
            )
            for binding in matching
        }
        return ConfirmationAllocationResult(
            source_analysis_id=source_analysis_id,
            allocation_plan_id=plan_id,
            exclusion_revision=1,
            allocations=allocations,
        )


class FakeSegmentVectorService:
    def __init__(self) -> None:
        self.calls: list[SegmentVectorBuildRequest] = []

    def prepare_segment_vector(
        self,
        request: SegmentVectorBuildRequest,
    ) -> SegmentVectorBuildResult:
        self.calls.append(request)
        return SegmentVectorBuildResult(
            segment_id=request.segment_id,
            segment_vector_id=f"segvec_{request.segment_id}_v1",
            vector_values=[1.0, *([0.0] * 63)],
            source="decision_analysis",
        )


class FakeSegmentSuggester:
    def __init__(self, segments: list[SegmentDefinitionRecord]) -> None:
        self.segments = segments
        self.calls: list[tuple[PromotionRecord, str | None]] = []

    def suggest_segments(
        self,
        *,
        promotion: PromotionRecord,
        segment_instruction: str | None = None,
    ) -> list[SegmentDefinitionRecord]:
        self.calls.append((promotion, segment_instruction))
        return self.segments


class ConcurrentSegmentReportGenerator:
    def __init__(self, expected_calls: int) -> None:
        self.barrier = threading.Barrier(expected_calls)
        self.calls: list[str] = []

    def generate_report(
        self,
        report_input: SegmentSuggestionReportInput,
    ) -> dict[str, Any]:
        self.barrier.wait(timeout=2)
        segment_id = report_input.segment.segment_id
        self.calls.append(segment_id)
        return {"source": "test", "title": segment_id}


def promotion_record(
    *,
    channel: str = "onsite_banner",
    goal_metric: str = "booking_conversion_rate",
    min_sample_size: int = 1000,
) -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id=f"promo_{channel}_001",
        channel=channel,
        goal_metric=goal_metric,
        goal_target_value=Decimal("0.030000"),
        goal_basis="all_segments",
        min_sample_size=min_sample_size,
        landing_url="https://demo-stay.example.com/summer",
        message_brief="Drive summer hotel booking.",
    )


def segment_record(
    segment_id: str,
    *,
    source: str = "system_default",
    sample_size: int = 2000,
    sample_ratio: Decimal = Decimal("0.020000"),
    rule_json: Mapping[str, Any] | None = None,
) -> SegmentDefinitionRecord:
    return SegmentDefinitionRecord(
        segment_id=segment_id,
        project_id="hotel-client-a",
        segment_name=segment_id.replace("_", " ").title(),
        source=source,
        query_preview_id=None,
        natural_language_query=f"{segment_id} hotel audience",
        generated_sql=None,
        rule_json=rule_json or {"segment_id": segment_id},
        profile_json={"primary_segment": segment_id},
        sample_size=sample_size,
        total_eligible_user_count=74200,
        sample_ratio=sample_ratio,
        status="active",
    )


def profile_record(
    profile_name: str,
    *,
    event_count: int = 2000,
) -> HotelMarketingProfileRecord:
    return HotelMarketingProfileRecord(
        project_id="hotel-client-a",
        profile_name=profile_name,
        profile_json={
            "event_count": event_count,
            "booking_count": 120,
            "mobile_ratio": 0.65,
            "package_ratio": 0.25,
            "avg_stay_nights": 2.4,
            "avg_days_until_checkin": 14.2,
        },
    )


def analysis_request(
    *,
    promotion_id: str,
    operator_instruction: str | None = None,
    segment_instruction: str | None = None,
) -> AnalysisRequest:
    return AnalysisRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id=promotion_id,
        operator_instruction=operator_instruction,
        segment_instruction=segment_instruction,
    )


def build_service(
    *,
    promotion: PromotionRecord | None,
    segments: list[SegmentDefinitionRecord],
    profiles: list[HotelMarketingProfileRecord] | None = None,
    user_profile_summaries: Mapping[str, HotelMarketingProfileRecord] | None = None,
    booking_training_records: list[BookingTrainingRecord] | None = None,
    segment_vector_service: FakeSegmentVectorService | None = None,
    segment_suggester: FakeSegmentSuggester | None = None,
    segment_report_generator: ConcurrentSegmentReportGenerator | None = None,
    audience_v2_coordinator: FakeAudienceV2Coordinator | None = None,
    audience_bindings: Sequence[SegmentSuggestionAudienceBindingRecord] = (),
    audience_allocation_service: FakeAudienceAllocationService | None = None,
    configured_candidate_limit: int = 3,
) -> tuple[
    PromotionAnalysisService,
    FakePromotionAnalysisRepository,
    FakeSegmentDefinitionRepository,
]:
    analysis_repository = FakePromotionAnalysisRepository(audience_bindings)
    segment_definition_repository = FakeSegmentDefinitionRepository(segments)
    service = PromotionAnalysisService(
        promotion_repository=FakePromotionRepository(promotion),
        segment_definition_repository=segment_definition_repository,
        hotel_profile_repository=FakeHotelProfileRepository(
            profiles or [],
            user_profile_summaries=user_profile_summaries,
            booking_training_records=booking_training_records,
        ),
        promotion_analysis_repository=analysis_repository,
        segment_vector_service=segment_vector_service or FakeSegmentVectorService(),
        segment_suggester=segment_suggester,
        segment_report_generator=segment_report_generator,
        audience_v2_coordinator=audience_v2_coordinator,
        audience_allocation_service=(
            audience_allocation_service
            or FakeAudienceAllocationService(audience_bindings)
        ),
        max_default_target_segments=configured_candidate_limit,
    )
    return service, analysis_repository, segment_definition_repository


def default_segments() -> list[SegmentDefinitionRecord]:
    return [
        segment_record("seg_mobile_user"),
        segment_record("seg_family_trip"),
        segment_record("seg_near_checkin"),
        segment_record("seg_existing_all"),
        segment_record("seg_repeat_hotel_no_booking"),
        segment_record("seg_long_stay"),
    ]


def segment_ids(
    target_segments: Sequence[PromotionTargetSegmentWrite],
) -> list[str]:
    return [segment.segment_id for segment in target_segments]


def suggestion_segment_ids(
    suggestions: Sequence[PromotionSegmentSuggestionWrite],
) -> list[str]:
    return [suggestion.segment_id for suggestion in suggestions]


def test_service_analyzes_email_promotion_with_configured_candidate_limit() -> None:
    promotion = promotion_record(channel="email")
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        profiles=[profile_record("seg_mobile_user", event_count=5000)],
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert result.analysis.status == "completed"
    assert re.fullmatch(
        rf"analysis_{promotion.promotion_id}_run_[0-9a-f]{{8}}",
        result.analysis.analysis_id,
    )
    assert segment_ids(result.target_segments) == [
        "seg_mobile_user",
        "seg_family_trip",
        "seg_near_checkin",
    ]
    assert analysis_repository.saved.analysis == result.analysis
    assert analysis_repository.saved.segment_suggestions == result.segment_suggestions
    assert suggestion_segment_ids(result.segment_suggestions) == segment_ids(
        result.target_segments
    )
    assert result.analysis.profile_summary_json["selected_segment_count"] == 3
    assert result.target_segments[0].content_brief_json["hotel_profile"]["event_count"] == 5000
    assert "primary_signals" not in result.target_segments[0].profile_json


@pytest.mark.parametrize("configured_candidate_limit", [1, 2, 3])
def test_recommendation_card_batch_respects_configured_candidate_limit(
    configured_candidate_limit: int,
) -> None:
    promotion = promotion_record(channel="email")
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        configured_candidate_limit=configured_candidate_limit,
    )

    result = service.recommend_segments(
        analysis_request(promotion_id=promotion.promotion_id)
    )

    assert len(result.target_segments) == configured_candidate_limit
    assert len(result.segment_suggestions) == configured_candidate_limit


def test_service_creates_new_analysis_id_for_repeated_ai_recommendations() -> None:
    promotion = promotion_record(channel="email")
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
    )

    first = service.analyze(analysis_request(promotion_id=promotion.promotion_id))
    second = service.analyze(analysis_request(promotion_id=promotion.promotion_id))

    assert first.analysis.analysis_id != second.analysis.analysis_id
    assert re.fullmatch(
        rf"analysis_{promotion.promotion_id}_run_[0-9a-f]{{8}}",
        first.analysis.analysis_id,
    )
    assert re.fullmatch(
        rf"analysis_{promotion.promotion_id}_run_[0-9a-f]{{8}}",
        second.analysis.analysis_id,
    )
    assert {
        suggestion.analysis_id for suggestion in first.segment_suggestions
    } == {first.analysis.analysis_id}
    assert {
        suggestion.analysis_id for suggestion in second.segment_suggestions
    } == {second.analysis.analysis_id}


def test_service_prioritizes_related_custom_segment_for_onsite_banner() -> None:
    promotion = promotion_record(channel="onsite_banner")
    segments = [
        segment_record(
            "seg_repeat_hotel_no_booking",
            source="custom_chatkit",
            sample_size=1342,
        ),
        segment_record("seg_family_trip"),
        segment_record("seg_mobile_user"),
        segment_record("seg_near_checkin"),
        segment_record("seg_existing_all"),
    ]
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=segments,
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_repeat_hotel_no_booking",
        "seg_family_trip",
        "seg_mobile_user",
    ]
    assert result.target_segments[0].priority == "high"
    assert {segment.status for segment in result.target_segments} == {"planned"}
    assert analysis_repository.saved.target_segments is None
    assert analysis_repository.events == ["analysis", "segment_suggestions"]


def test_service_applies_sms_default_segment_order() -> None:
    promotion = promotion_record(channel="sms")
    service, _, _ = build_service(promotion=promotion, segments=default_segments())

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_near_checkin",
        "seg_mobile_user",
        "seg_family_trip",
    ]


def test_service_analyzes_focus_segment_ids_only() -> None:
    promotion = promotion_record(channel="onsite_banner")
    vector_service = FakeSegmentVectorService()
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_vector_service=vector_service,
    )

    result = service.analyze_focus(
        NextLoopFocusAnalysisRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id=promotion.promotion_id,
            focus_segment_ids=["seg_near_checkin"],
            loop_count=2,
            source_promotion_run_id="prun_banner_001_loop_1",
            source_failed_ad_experiment_ids=["adexp_near_checkin_001"],
            operator_instruction="Stress breakfast.",
        ),
    )

    assert segment_ids(result.target_segments) == ["seg_near_checkin"]
    assert analysis_repository.saved.target_segments == result.target_segments
    assert {segment.status for segment in result.target_segments} == {"planned"}
    assert analysis_repository.events == [
        "analysis",
        "target_segments",
    ]
    assert result.segment_suggestions == []
    saved_analysis = analysis_repository.saved.analysis
    assert saved_analysis is not None
    assert saved_analysis.focus_segment_ids_json == ["seg_near_checkin"]
    assert saved_analysis.input_snapshot_json["focus_segment_ids"] == [
        "seg_near_checkin"
    ]
    assert saved_analysis.operator_instruction == "Stress breakfast."
    assert saved_analysis.profile_summary_json["selection_mode"] == "focus"
    assert vector_service.calls == [
        SegmentVectorBuildRequest(
                project_id="hotel-client-a",
                promotion_id=promotion.promotion_id,
                analysis_id=result.analysis.analysis_id,
                segment_id="seg_near_checkin",
                candidate_user_ids=[],
            )
    ]


def test_service_can_approve_automatic_next_loop_focus_segments() -> None:
    promotion = promotion_record(channel="onsite_banner")
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
    )

    result = service.analyze_focus(
        NextLoopFocusAnalysisRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id=promotion.promotion_id,
            focus_segment_ids=["seg_near_checkin"],
            loop_count=2,
            source_promotion_run_id="prun_banner_001_loop_1",
            source_failed_ad_experiment_ids=["adexp_near_checkin_001"],
        ),
        target_status="approved",
    )

    assert analysis_repository.saved.target_segments == result.target_segments
    assert {segment.status for segment in result.target_segments} == {"approved"}


def test_next_loop_analysis_id_separates_and_bounds_source_lineage() -> None:
    common = {
        "prefix": "analysis",
        "promotion_id": "promo_" + ("long_hotel_promotion_" * 10),
        "loop_count": 2,
    }

    first = _bounded_next_loop_lineage_id(
        **common,
        source_promotion_run_id="prun_scope_a",
    )
    second = _bounded_next_loop_lineage_id(
        **common,
        source_promotion_run_id="prun_scope_b",
    )

    assert first != second
    assert len(first) <= 100
    assert len(second) <= 100


def test_service_analyzes_requested_segments_without_refreshing_suggestions() -> None:
    promotion = promotion_record(channel="email")
    fresh_ai_segment = replace(
        segment_record(
            "seg_ai_raw_should_not_be_created",
            source="ai_suggested",
            rule_json={"candidate_user_ids": ["user_999"]},
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    suggester = FakeSegmentSuggester([fresh_ai_segment])
    service, analysis_repository, segment_definition_repository = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=suggester,
    )

    result = service.analyze_segments(
        SegmentAnalysisRequest(
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_ids=["seg_family_trip"],
        )
    )

    assert segment_ids(result.target_segments) == ["seg_family_trip"]
    assert {segment.status for segment in result.target_segments} == {"approved"}
    assert analysis_repository.saved.target_segments == result.target_segments
    assert result.segment_suggestions == []
    assert suggester.calls == []
    assert segment_definition_repository.saved_ai_suggested == []
    assert analysis_repository.events == ["analysis", "target_segments"]


def test_v2_recommendation_projects_snapshot_counts_to_card_metadata() -> None:
    promotion = promotion_record(channel="onsite_banner", min_sample_size=20)
    segment = _v2_ai_segment(
        promotion=promotion,
        segment_id="seg_ai_registered_intent",
        sample_size=40,
        raw_audience={
            "total_eligible_user_count": 74_200,
            "matching_user_count": 40,
            "selected_user_count": 40,
            "selection_basis": "candidate_condition_match",
            "selected_user_role": "recommended_audience",
        },
    )
    coordinator = FakePreparingAudienceV2Coordinator()
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[],
        segment_suggester=FakeSegmentSuggester([segment]),
        audience_v2_coordinator=coordinator,
    )

    result = service.recommend_segments(
        analysis_request(promotion_id=promotion.promotion_id)
    )

    assert len(coordinator.prepare_many_calls) == 1
    target = result.target_segments[0]
    evidence = target.data_evidence_json
    assert target.estimated_size == 10
    assert evidence["candidate_generation_user_count"] == 40
    assert evidence["sample_size"] == 10
    assert evidence["sample_ratio"] == 0.1
    assert evidence["matching_user_ratio"] == 0.2
    assert evidence["selection_ratio_within_matching"] == 0.5
    assert evidence["targetable"] is True
    assert evidence["audience_status"] == "insufficient_sample"

    suggestion = result.segment_suggestions[0]
    assert suggestion.score_json["estimated_size"] == 10
    assert suggestion.audience_snapshot_id == f"snapshot_{segment.segment_id}"
    display_copy = suggestion.metadata_json["display_copy"]
    assert display_copy["audience_summary"] == (
        "분석 가능 사용자 100명 · 행동 조건 부합 20명 · "
        "실험 대상 사용자 10명"
    )
    assert display_copy["audience"] == {
        "total_eligible_user_count": 100,
        "matching_user_count": 20,
        "selected_user_count": 10,
        "matching_user_ratio": 0.2,
        "selected_user_ratio": 0.1,
        "selection_ratio_within_matching": 0.5,
        "selection_limited": True,
        "selection_basis": "hard_predicate_and_exact_cosine",
        "selected_user_role": "final_experiment_audience",
    }
    assert analysis_repository.saved.segment_suggestions == result.segment_suggestions


def test_v2_recommendation_projects_each_snapshot_to_its_own_card() -> None:
    promotion = promotion_record(channel="onsite_banner", min_sample_size=20)
    segments = [
        _v2_ai_segment(
            promotion=promotion,
            segment_id=segment_id,
            sample_size=40,
            raw_audience={
                "total_eligible_user_count": 74_200,
                "matching_user_count": 40,
                "selected_user_count": 40,
            },
        )
        for segment_id in ("seg_ai_first", "seg_ai_second", "seg_ai_third")
    ]
    expected = {
        "seg_ai_first": (100, 60, 30, True),
        "seg_ai_second": (100, 20, 5, False),
        "seg_ai_third": (100, 0, 0, False),
    }
    coordinator = FakePreparingAudienceV2Coordinator(
        counts_by_segment=expected,
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=[],
        segment_suggester=FakeSegmentSuggester(segments),
        audience_v2_coordinator=coordinator,
    )

    result = service.recommend_segments(
        analysis_request(promotion_id=promotion.promotion_id)
    )

    assert len(coordinator.prepare_many_calls) == 1
    assert len(result.segment_suggestions) == 3
    for suggestion in result.segment_suggestions:
        eligible, matching, selected, _meets_minimum = expected[
            suggestion.segment_id
        ]
        audience = suggestion.metadata_json["display_copy"]["audience"]
        assert audience["total_eligible_user_count"] == eligible
        assert audience["matching_user_count"] == matching
        assert audience["selected_user_count"] == selected
        assert suggestion.audience_snapshot_id == (
            f"snapshot_{suggestion.segment_id}"
        )

    targets = {target.segment_id: target for target in result.target_segments}
    assert targets["seg_ai_first"].data_evidence_json["audience_status"] == (
        "targetable"
    )
    assert targets["seg_ai_second"].data_evidence_json["audience_status"] == (
        "insufficient_sample"
    )
    assert targets["seg_ai_third"].data_evidence_json["audience_status"] == (
        "no_eligible_audience"
    )


def test_v2_recommendation_keeps_zero_audience_without_legacy_card_values() -> None:
    promotion = promotion_record(channel="onsite_banner", min_sample_size=20)
    segment = _v2_ai_segment(
        promotion=promotion,
        segment_id="seg_ai_registered_zero",
        sample_size=40,
        raw_audience={
            "total_eligible_user_count": 74_200,
            "matching_user_count": 40,
            "selected_user_count": 40,
        },
    )
    coordinator = FakePreparingAudienceV2Coordinator(selected_user_count=0)
    service, _, _ = build_service(
        promotion=promotion,
        segments=[],
        segment_suggester=FakeSegmentSuggester([segment]),
        audience_v2_coordinator=coordinator,
    )

    result = service.recommend_segments(
        analysis_request(promotion_id=promotion.promotion_id)
    )

    target = result.target_segments[0]
    assert target.estimated_size == 0
    assert target.data_evidence_json["sample_size"] == 0
    assert target.data_evidence_json["sample_ratio"] == 0.0
    assert target.data_evidence_json["targetable"] is False
    assert target.data_evidence_json["audience_status"] == "no_eligible_audience"
    audience = result.segment_suggestions[0].metadata_json["display_copy"][
        "audience"
    ]
    assert audience["selected_user_count"] == 0
    assert audience["selected_user_ratio"] == 0.0
    assert audience["selection_ratio_within_matching"] == 0.0


def test_legacy_recommendation_preserves_existing_card_audience() -> None:
    promotion = promotion_record(channel="onsite_banner")
    raw_audience = {
        "total_eligible_user_count": 100,
        "matching_user_count": 40,
        "selected_user_count": 25,
        "selection_basis": "candidate_condition_match",
        "selected_user_role": "recommended_audience",
    }
    segment = replace(
        segment_record(
            "seg_ai_legacy",
            source="ai_suggested",
            sample_size=25,
            rule_json={"candidate_user_ids": ["user_1"]},
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_legacy",
            "display_copy": {
                "title": "기존 고객군",
                "audience_summary": "기존 요약",
                "audience": raw_audience,
            },
        },
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=[],
        segment_suggester=FakeSegmentSuggester([segment]),
    )

    result = service.recommend_segments(
        analysis_request(promotion_id=promotion.promotion_id)
    )

    display_copy = result.segment_suggestions[0].metadata_json["display_copy"]
    assert display_copy["audience_summary"] == "기존 요약"
    assert display_copy["audience"] == raw_audience
    assert "candidate_generation_user_count" not in (
        result.target_segments[0].data_evidence_json
    )


def test_v2_confirmation_reuses_latest_snapshot_without_searching_again() -> None:
    promotion = promotion_record(channel="onsite_banner")
    segment_id = "seg_ai_registered_intent"
    segment = replace(
        segment_record(
            segment_id,
            source="ai_suggested",
            rule_json={
                "audience_resolution_contract": "segment_audience.v1",
                "segment_audience_spec": dict(
                    RegisteredSegmentAudienceBinder().bind(
                        candidate_type="intent_matched"
                    )
                ),
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    coordinator = FakeAudienceV2Coordinator()
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[segment],
        audience_v2_coordinator=coordinator,
        audience_bindings=(
            SegmentSuggestionAudienceBindingRecord(
                suggestion_id="suggestion_1",
                analysis_id="recommendation_analysis_1",
                segment_id=segment_id,
                audience_snapshot_id="snapshot_1",
            ),
        ),
    )

    result = service.analyze_segments(
        SegmentAnalysisRequest(
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_ids=[segment_id],
        )
    )

    assert coordinator.prepare_many_calls == 0
    assert len(coordinator.prepare_calls) == 1
    assert result.target_segments[0].audience_snapshot_id == "final_snapshot_1"
    assert result.target_segments[0].allocation_plan_id == "allocation_test"
    assert analysis_repository.saved.target_segments == result.target_segments


def test_v2_confirmation_retry_uses_the_same_source_bound_analysis_id() -> None:
    promotion = promotion_record(channel="onsite_banner")
    segment_id = "seg_ai_registered_retry"
    segment = replace(
        segment_record(
            segment_id,
            source="ai_suggested",
            rule_json={
                "audience_resolution_contract": "segment_audience.v1",
                "segment_audience_spec": dict(
                    RegisteredSegmentAudienceBinder().bind(
                        candidate_type="promotion_responsive"
                    )
                ),
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    binding = SegmentSuggestionAudienceBindingRecord(
        suggestion_id="suggestion_retry",
        analysis_id="recommendation_analysis_retry",
        segment_id=segment_id,
        audience_snapshot_id="snapshot_retry",
    )
    coordinator = FakeAudienceV2Coordinator()
    allocation_service = FakeAudienceAllocationService((binding,))
    service, _, _ = build_service(
        promotion=promotion,
        segments=[segment],
        audience_v2_coordinator=coordinator,
        audience_bindings=(binding,),
        audience_allocation_service=allocation_service,
    )
    request = SegmentAnalysisRequest(
        project_id=promotion.project_id,
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        segment_ids=[segment_id],
    )

    first = service.analyze_segments(request)
    second = service.analyze_segments(request)
    changed_request = service.analyze_segments(
        request.model_copy(update={"operator_instruction": "Use a new message."})
    )

    assert first.analysis.analysis_id == second.analysis.analysis_id
    assert changed_request.analysis.analysis_id != first.analysis.analysis_id
    assert first.target_segments == second.target_segments
    assert len(allocation_service.confirm_calls) == 3
    assert {
        call["source_analysis_id"]
        for call in allocation_service.confirm_calls
    } == {"recommendation_analysis_retry"}
    assert coordinator.prepare_many_calls == 0


def test_v2_confirmation_rejects_missing_recommendation_snapshot() -> None:
    promotion = promotion_record(channel="onsite_banner")
    segment_id = "seg_ai_registered_missing_snapshot"
    segment = replace(
        segment_record(
            segment_id,
            source="ai_suggested",
            rule_json={
                "audience_resolution_contract": "segment_audience.v1",
                "segment_audience_spec": dict(
                    RegisteredSegmentAudienceBinder().bind(
                        candidate_type="promotion_responsive"
                    )
                ),
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    coordinator = FakeAudienceV2Coordinator()
    service, _, _ = build_service(
        promotion=promotion,
        segments=[segment],
        audience_v2_coordinator=coordinator,
        audience_bindings=(),
    )

    with pytest.raises(AudienceSnapshotBindingError) as error:
        service.analyze_segments(
            SegmentAnalysisRequest(
                project_id=promotion.project_id,
                campaign_id=promotion.campaign_id,
                promotion_id=promotion.promotion_id,
                segment_ids=[segment_id],
            )
        )

    assert error.value.code == "segment_audience_snapshot_binding_required"
    assert error.value.segment_id == segment_id
    assert coordinator.prepare_calls == []
    assert coordinator.prepare_many_calls == 0


def test_v2_confirmation_prepares_custom_snapshot_and_allocates_with_ai_candidate() -> None:
    promotion = promotion_record(channel="onsite_banner")
    ai_segment = _v2_ai_segment(
        promotion=promotion,
        segment_id="seg_ai_confirmed",
        sample_size=20,
        raw_audience={},
    )
    custom_segment = _v2_custom_segment(
        promotion=promotion,
        segment_id="seg_custom_confirmed",
    )
    source_analysis_id = "recommendation_analysis_mixed"
    bindings = (
        SegmentSuggestionAudienceBindingRecord(
            suggestion_id="suggestion_ai",
            analysis_id=source_analysis_id,
            segment_id=ai_segment.segment_id,
            audience_snapshot_id="snapshot_ai",
        ),
        SegmentSuggestionAudienceBindingRecord(
            suggestion_id="internal_custom",
            analysis_id=source_analysis_id,
            segment_id=custom_segment.segment_id,
            audience_snapshot_id="snapshot_custom",
        ),
    )
    coordinator = FakePreparingAudienceV2Coordinator()
    allocation_service = FakeAudienceAllocationService(bindings)
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[ai_segment, custom_segment],
        audience_v2_coordinator=coordinator,
        audience_bindings=bindings,
        audience_allocation_service=allocation_service,
    )

    result = service.analyze_segments(
        SegmentAnalysisRequest(
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            segment_ids=[ai_segment.segment_id, custom_segment.segment_id],
        )
    )

    assert len(coordinator.prepare_many_calls) == 1
    prepare_call = coordinator.prepare_many_calls[0]
    assert prepare_call["analysis_id"] == source_analysis_id
    assert [segment.segment_id for segment in prepare_call["segments"]] == [
        custom_segment.segment_id
    ]
    assert len(allocation_service.confirm_calls) == 1
    assert allocation_service.confirm_calls[0]["source_analysis_id"] == source_analysis_id
    assert set(allocation_service.confirm_calls[0]["segment_ids"]) == {
        ai_segment.segment_id,
        custom_segment.segment_id,
    }
    assert {target.audience_snapshot_id for target in result.target_segments} == {
        "final_snapshot_ai",
        "final_snapshot_custom",
    }
    assert analysis_repository.saved.target_segments == result.target_segments


def test_v2_next_loop_creates_final_allocation_and_reservation_binding() -> None:
    promotion = promotion_record(channel="onsite_banner")
    segment_id = "seg_ai_next_loop_funnel"
    segment = _v2_ai_segment(
        promotion=promotion,
        segment_id=segment_id,
        sample_size=25,
        raw_audience={},
    )
    coordinator = FakePreparingAudienceV2Coordinator()
    allocation_service = FakeAudienceAllocationService()
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[segment],
        audience_v2_coordinator=coordinator,
        audience_allocation_service=allocation_service,
    )

    result = service.analyze_focus(
        NextLoopFocusAnalysisRequest(
            project_id=promotion.project_id,
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
            focus_segment_ids=[segment_id],
            loop_count=2,
            source_promotion_run_id="run_previous",
            source_failed_ad_experiment_ids=["adexp_failed"],
        ),
        target_status="approved",
    )

    target = result.target_segments[0]
    assert target.audience_snapshot_id == f"final_snapshot_{segment_id}"
    assert target.allocation_plan_id == "allocation_test"
    assert allocation_service.confirm_calls[0]["source_analysis_id"] == (
        result.analysis.analysis_id
    )
    assert analysis_repository.saved.target_segments == result.target_segments


def _v2_ai_segment(
    *,
    promotion: PromotionRecord,
    segment_id: str,
    sample_size: int,
    raw_audience: Mapping[str, Any],
) -> SegmentDefinitionRecord:
    return replace(
        segment_record(
            segment_id,
            source="ai_suggested",
            sample_size=sample_size,
            rule_json={
                "source": "raw_event_intent",
                "audience_resolution_contract": "segment_audience.v1",
                "segment_audience_spec": dict(
                    RegisteredSegmentAudienceBinder().bind(
                        candidate_type="intent_matched"
                    )
                ),
                "candidate_user_ids": ["candidate_user"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": segment_id,
            "display_copy": {
                "title": "기존 후보 카드",
                "audience_summary": "과거 후보 계산 요약",
                "audience": dict(raw_audience),
            },
        },
    )


def _v2_custom_segment(
    *,
    promotion: PromotionRecord,
    segment_id: str,
) -> SegmentDefinitionRecord:
    conditions = [
        {
            "event_name": "booking_start",
            "label": "예약 시작",
            "minimum_count": 1,
            "maximum_count": None,
            "destination": "jeju",
            "checkin_months": [],
            "property_filters": [],
        },
        {
            "event_name": "booking_complete",
            "label": "예약 완료 없음",
            "minimum_count": 0,
            "maximum_count": 0,
            "destination": None,
            "checkin_months": [],
            "property_filters": [],
        },
    ]
    return replace(
        segment_record(
            segment_id,
            source="custom_chatkit",
            rule_json={
                "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
                "segment_audience_spec": {
                    "schema_version": "hotel_behavior.v2",
                    "template_id": CUSTOM_STRUCTURED_TEMPLATE_ID,
                    "template_version": CUSTOM_STRUCTURED_TEMPLATE_VERSION,
                    "template_semantic_hash": CUSTOM_STRUCTURED_TEMPLATE_HASH,
                    "candidate_type": CUSTOM_STRUCTURED_CANDIDATE_TYPE,
                    "condition_keys": [CUSTOM_STRUCTURED_CONDITION_KEY],
                    "query_signal_keys": [
                        "booking_start_intensity",
                        "booking_start_without_complete",
                    ],
                    "hard_predicate_keys": [CUSTOM_STRUCTURED_CONDITION_KEY],
                    "parameters": {
                        "lookback_days": CUSTOM_STRUCTURED_WINDOW_DAYS,
                        "conditions": conditions,
                    },
                    "parameter_policy_id": CUSTOM_STRUCTURED_PARAMETER_POLICY_ID,
                    "semantic_selection_policy_id": CUSTOM_STRUCTURED_SELECTION_POLICY_ID,
                    "semantic_anchor_policy_id": CUSTOM_STRUCTURED_ANCHOR_POLICY_ID,
                    "observation_window_days": CUSTOM_STRUCTURED_WINDOW_DAYS,
                },
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )


def test_v2_confirmation_rejects_stale_selection_from_different_analyses() -> None:
    promotion = promotion_record(channel="onsite_banner")
    binder = RegisteredSegmentAudienceBinder()
    segments = [
        replace(
            segment_record(
                segment_id,
                source="ai_suggested",
                rule_json={
                    "audience_resolution_contract": "segment_audience.v1",
                    "segment_audience_spec": dict(
                        binder.bind(candidate_type=candidate_type)
                    ),
                },
            ),
            campaign_id=promotion.campaign_id,
            promotion_id=promotion.promotion_id,
        )
        for segment_id, candidate_type in (
            ("seg_ai_responsive", "promotion_responsive"),
            ("seg_ai_explorer", "general_destination_explorer"),
        )
    ]
    coordinator = FakeAudienceV2Coordinator()
    service, _, _ = build_service(
        promotion=promotion,
        segments=segments,
        audience_v2_coordinator=coordinator,
        audience_bindings=(
            SegmentSuggestionAudienceBindingRecord(
                suggestion_id="suggestion_1",
                analysis_id="recommendation_analysis_1",
                segment_id=segments[0].segment_id,
                audience_snapshot_id="snapshot_1",
            ),
            SegmentSuggestionAudienceBindingRecord(
                suggestion_id="suggestion_2",
                analysis_id="recommendation_analysis_2",
                segment_id=segments[1].segment_id,
                audience_snapshot_id="snapshot_2",
            ),
        ),
    )

    with pytest.raises(AudienceSnapshotBindingError) as error:
        service.analyze_segments(
            SegmentAnalysisRequest(
                project_id=promotion.project_id,
                campaign_id=promotion.campaign_id,
                promotion_id=promotion.promotion_id,
                segment_ids=[segment.segment_id for segment in segments],
            )
        )

    assert error.value.code == "segment_audience_source_batch_mismatch"
    assert coordinator.prepare_calls == []
    assert coordinator.prepare_many_calls == 0


def test_service_keeps_stored_ai_focus_segment_without_refreshing_suggestions() -> None:
    promotion = promotion_record(channel="onsite_banner")
    stored_segment_id = (
        "seg_ai_raw_promo_onsite_banner_001_"
        "target_destination_affinity_membership"
    )
    refreshed_segment_id = (
        "seg_ai_raw_promo_onsite_banner_001_"
        "benefit_value_seeker_other_membership"
    )
    stored_segment = replace(
        segment_record(
            stored_segment_id,
            source="ai_suggested",
            rule_json={
                "source": "raw_event_intent",
                "candidate_user_ids": ["user_101", "user_102"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    refreshed_segment = replace(
        segment_record(
            refreshed_segment_id,
            source="ai_suggested",
            rule_json={
                "source": "raw_event_intent",
                "candidate_user_ids": ["user_101", "user_102"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    suggester = FakeSegmentSuggester([refreshed_segment])
    service, _, segment_definition_repository = build_service(
        promotion=promotion,
        segments=[stored_segment, *default_segments()],
        segment_suggester=suggester,
    )

    result = service.analyze_focus(
        NextLoopFocusAnalysisRequest(
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id=promotion.promotion_id,
            focus_segment_ids=[stored_segment_id],
            loop_count=2,
            source_promotion_run_id="prun_banner_001_loop_1",
            source_failed_ad_experiment_ids=["adexp_target_affinity_001"],
            operator_instruction=None,
        ),
    )

    assert segment_ids(result.target_segments) == [stored_segment_id]
    assert suggester.calls == []
    assert segment_definition_repository.saved_ai_suggested == []
    assert result.analysis.input_snapshot_json["focus_segment_ids"] == [
        stored_segment_id
    ]


def test_service_populates_segment_vector_ids_when_vector_service_is_configured() -> None:
    promotion = promotion_record(channel="onsite_banner")
    vector_service = FakeSegmentVectorService()
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[
            segment_record(
                "seg_mobile_user",
                rule_json={
                    "candidate_user_ids": ["user_001", "user_002"],
                },
            ),
        ],
        segment_vector_service=vector_service,
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert result.target_segments[0].segment_vector_id == "segvec_seg_mobile_user_v1"
    assert analysis_repository.events == ["analysis", "segment_suggestions"]
    assert vector_service.calls == [
        SegmentVectorBuildRequest(
            project_id="hotel-client-a",
            promotion_id=promotion.promotion_id,
            analysis_id=result.analysis.analysis_id,
            segment_id="seg_mobile_user",
            candidate_user_ids=["user_001", "user_002"],
        )
    ]


def test_service_prioritizes_ai_suggested_cluster_segments() -> None:
    promotion = promotion_record(channel="onsite_banner")
    ai_segment = segment_record(
        "seg_ai_cluster_promo_onsite_banner_001_1_abcdef1234",
        source="ai_suggested",
        sample_size=1800,
        rule_json={
            "source": "user_vector_clustering",
            "candidate_user_ids": ["user_101", "user_102"],
        },
    )
    ai_segment = replace(
        ai_segment,
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": ai_segment.segment_id,
            "source": "user_vector_clustering",
            "cluster_score": 0.99,
            "top_common_features": [
                "Booking conversion ready users",
                "Promotion-engaged hotel users",
                "Hotel page viewers",
            ],
        },
    )
    vector_service = FakeSegmentVectorService()
    suggester = FakeSegmentSuggester([ai_segment])
    service, analysis_repository, segment_definition_repository = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_vector_service=vector_service,
        segment_suggester=suggester,
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    expected_segment_ids = [
        ai_segment.segment_id,
        "seg_family_trip",
        "seg_mobile_user",
    ]
    assert segment_ids(result.target_segments) == expected_segment_ids
    assert segment_definition_repository.saved_ai_suggested == [ai_segment]
    assert suggester.calls == [(promotion, None)]
    assert analysis_repository.events == ["analysis", "segment_suggestions"]
    assert result.analysis.output_json == {
        "selected_segment_ids": expected_segment_ids,
        "target_segment_count": 3,
    }
    assert result.analysis.profile_summary_json["candidate_segment_count"] == 7
    assert result.analysis.input_snapshot_json["available_segment_definitions"][6] == {
        "segment_id": ai_segment.segment_id,
        "campaign_id": promotion.campaign_id,
        "promotion_id": promotion.promotion_id,
        "segment_name": ai_segment.segment_name,
        "source": "ai_suggested",
        "sample_size": 1800,
        "total_eligible_user_count": 74200,
        "sample_ratio": "0.020000",
        "status": "active",
    }
    assert result.target_segments[0].data_evidence_json["source"] == "ai_suggested"
    assert result.target_segments[0].profile_json["cluster_score"] == 0.99
    assert "primary_signals" not in result.target_segments[0].profile_json
    assert "audience_evidence" not in result.target_segments[0].content_brief_json
    assert result.target_segments[0].content_brief_json["readiness"] == {
        "level": "fallback_only",
        "missing_sections": ["primary_signals", "score_components"],
        "available_sections": [
            "segment_snapshot",
            "promotion_context",
            "fallback_guidance",
            "source_refs",
        ],
    }
    assert result.segment_suggestions[0].suggestion_source == "ai_generated"
    assert result.segment_suggestions[0].status == "suggested"
    assert result.segment_suggestions[0].suggested_rank == 1
    assert result.segment_suggestions[0].reason_json["primary_signals"] == [
        "booking_conversion_ready",
        "promotion_engaged",
        "hotel_browsing",
    ]
    assert result.segment_suggestions[0].metadata_json["display_copy"] == {
        "title": "예약 가능성이 높은 프로모션 반응 고객",
        "audience_summary": "분석 대상 74200명 중 1800명 · 2%",
        "signal_chips": ["예약 가능성 높음", "프로모션 반응", "호텔 탐색"],
        "reason": "예약 전환 목표에 가까운 행동 패턴을 보인 고객군입니다.",
        "action_hint": "사이트 내 배너로 호텔 혜택을 노출하기 적합합니다.",
    }
    ai_report = result.segment_suggestions[0].metadata_json["ai_report"]
    assert ai_report["version"] == "dec.segment-report.v3"
    assert ai_report["title"] == "예약 가능성이 높은 프로모션 반응 고객"
    assert ai_report["summary"]
    assert ai_report["promotion_interpretation"]
    assert ai_report["why_recommended"]
    assert ai_report["evidence"]
    assert ai_report["candidate_strengths"]
    assert ai_report["selection_considerations"]
    assert ai_report["action_hint"]
    assert ai_report["confidence_label"] in {"high", "medium", "low"}
    assert all(
        forbidden not in str(ai_report)
        for forbidden in ("벡터", "군집", "클러스터", "centroid", "유사도")
    )
    assert vector_service.calls[0] == SegmentVectorBuildRequest(
        project_id="hotel-client-a",
        promotion_id=promotion.promotion_id,
        analysis_id=result.analysis.analysis_id,
        segment_id=ai_segment.segment_id,
        candidate_user_ids=["user_101", "user_102"],
    )


def test_service_ignores_stale_ai_suggested_segments_when_new_suggestions_exist() -> None:
    promotion = promotion_record(channel="onsite_banner")
    stale_ai_segment = replace(
        segment_record(
            "seg_ai_cluster_promo_onsite_banner_001_old",
            source="ai_suggested",
            sample_size=8000,
            rule_json={
                "source": "raw_event_intent",
                "candidate_user_ids": ["old_user_001", "old_user_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_onsite_banner_001_old",
            "source": "raw_event_intent",
            "recommendation_score": 1.0,
        },
    )
    fresh_ai_segment = replace(
        segment_record(
            "seg_ai_cluster_promo_onsite_banner_001_fresh",
            source="ai_suggested",
            sample_size=160,
            rule_json={
                "source": "raw_event_intent",
                "candidate_user_ids": ["fresh_user_001", "fresh_user_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_onsite_banner_001_fresh",
            "source": "raw_event_intent",
            "recommendation_score": 0.8,
        },
    )
    service, _, segment_definition_repository = build_service(
        promotion=promotion,
        segments=[stale_ai_segment, *default_segments()],
        segment_suggester=FakeSegmentSuggester([fresh_ai_segment]),
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    selected_segment_ids = segment_ids(result.target_segments)
    assert fresh_ai_segment.segment_id in selected_segment_ids
    assert stale_ai_segment.segment_id not in selected_segment_ids
    assert segment_definition_repository.saved_ai_suggested == [fresh_ai_segment]
    assert all(
        segment["segment_id"] != stale_ai_segment.segment_id
        for segment in result.analysis.input_snapshot_json[
            "available_segment_definitions"
        ]
    )


def test_service_scopes_conversational_recommendation_to_fresh_segments() -> None:
    promotion = promotion_record(channel="onsite_banner")
    instruction = "최근 제주 숙소를 반복 탐색한 고객을 찾아줘"
    stale_ai_segment = replace(
        segment_record(
            "seg_ai_stale",
            source="ai_suggested",
            rule_json={
                "source": "raw_event_intent",
                "candidate_user_ids": ["old_user_001"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    fresh_ai_segment = replace(
        segment_record(
            "seg_ai_jeju_repeat",
            source="ai_suggested",
            sample_size=120,
            rule_json={
                "source": "raw_event_intent",
                "candidate_user_ids": ["jeju_user_001", "jeju_user_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_jeju_repeat",
            "source": "raw_event_intent",
            "recommendation_score": 0.81,
        },
    )
    segment_suggester = FakeSegmentSuggester([fresh_ai_segment])
    service, _, segment_definition_repository = build_service(
        promotion=promotion,
        segments=[stale_ai_segment, *default_segments()],
        segment_suggester=segment_suggester,
    )

    result = service.recommend_segments(
        analysis_request(
            promotion_id=promotion.promotion_id,
            segment_instruction=instruction,
        ),
    )

    assert segment_suggester.calls == [(promotion, instruction)]
    assert segment_ids(result.target_segments) == [fresh_ai_segment.segment_id]
    assert suggestion_segment_ids(result.segment_suggestions) == [
        fresh_ai_segment.segment_id
    ]
    assert segment_definition_repository.saved_ai_suggested == [fresh_ai_segment]


def test_service_does_not_replace_empty_conversational_result_with_defaults() -> None:
    promotion = promotion_record(channel="onsite_banner")
    segment_suggester = FakeSegmentSuggester([])
    service, analysis_repository, segment_definition_repository = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=segment_suggester,
    )

    with pytest.raises(
        SegmentSelectionError,
        match="no segment candidates matched segment instruction",
    ):
        service.recommend_segments(
            analysis_request(
                promotion_id=promotion.promotion_id,
                segment_instruction="최근 부산에서 반려동물 동반 객실을 본 고객",
            ),
        )

    assert analysis_repository.events == []
    assert segment_definition_repository.saved_ai_suggested == []


def test_service_builds_ai_reports_concurrently_without_mixing_segments() -> None:
    promotion = promotion_record(channel="onsite_banner")
    first_segment = replace(
        segment_record(
            "seg_ai_first",
            source="ai_suggested",
            sample_size=200,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_001", "user_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    second_segment = replace(
        segment_record(
            "seg_ai_second",
            source="ai_suggested",
            sample_size=100,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_003", "user_004"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    report_generator = ConcurrentSegmentReportGenerator(expected_calls=2)
    service, _, _ = build_service(
        promotion=promotion,
        segments=[first_segment, second_segment],
        segment_report_generator=report_generator,
    )

    result = service.recommend_segments(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert len(report_generator.calls) == 2
    assert set(report_generator.calls) == {
        first_segment.segment_id,
        second_segment.segment_id,
    }
    assert [
        suggestion.metadata_json["ai_report"]["title"]
        for suggestion in result.segment_suggestions
    ] == [suggestion.segment_id for suggestion in result.segment_suggestions]


def test_conversational_recommendation_skips_additional_openai_report_calls() -> None:
    promotion = promotion_record(channel="email")
    suggested_segment = replace(
        segment_record(
            "seg_ai_conversation_destination",
            source="ai_suggested",
            sample_size=80,
            rule_json={
                "source": "raw_event_intent",
                "candidate_user_ids": ["user_001", "user_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
    )
    report_generator = ConcurrentSegmentReportGenerator(expected_calls=1)
    service, _, _ = build_service(
        promotion=promotion,
        segments=[],
        segment_suggester=FakeSegmentSuggester([suggested_segment]),
        segment_report_generator=report_generator,
    )

    result = service.recommend_segments(
        analysis_request(
            promotion_id=promotion.promotion_id,
            segment_instruction="최근 제주 숙소를 반복 검색한 고객",
        ),
    )

    assert report_generator.calls == []
    assert result.segment_suggestions[0].metadata_json["ai_report"]["source"] == (
        "deterministic"
    )


def test_service_uses_promotion_matched_features_for_ai_suggestion_copy() -> None:
    promotion = promotion_record(channel="email")
    ai_segment = replace(
        segment_record(
            "seg_ai_cluster_promo_email_001_1_intent",
            source="ai_suggested",
            sample_size=120,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_101", "user_102"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_email_001_1_intent",
            "source": "user_vector_clustering",
            "cluster_score": 0.7,
            "promotion_cluster_similarity": 0.92,
            "cluster_quality_score": 0.7,
            "sample_size_score": 0.5,
            "recommendation_score": 0.88,
            "score_components": {
                "promotion_cluster_similarity": 0.92,
                "cluster_quality": 0.7,
                "sample_size": 0.5,
                "final_score": 0.88,
            },
            "behavior_metrics": {
                "booking_conversion_rate": 0.018,
            },
            "promotion_matched_features": [
                "Campaign redirect users",
                "Hotel market bucket 2 affinity users",
                "Free cancellation seekers",
            ],
            "top_common_features": ["Hotel search users"],
            "promotion_vector_basis": {
                "channel": "email",
                "goal_metric": "booking_conversion_rate",
            },
        },
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=FakeSegmentSuggester([ai_segment]),
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    first_suggestion = result.segment_suggestions[0]
    assert first_suggestion.segment_id == ai_segment.segment_id
    assert first_suggestion.reason_json["primary_signals"] == [
        "campaign_redirect",
        "hotel_market_affinity",
        "free_cancellation",
    ]
    assert first_suggestion.score_json["promotion_cluster_similarity"] == 0.92
    assert first_suggestion.score_json["recommendation_score"] == 0.88
    assert first_suggestion.metadata_json["promotion_matched_features"] == [
        "Campaign redirect users",
        "Hotel market bucket 2 affinity users",
        "Free cancellation seekers",
    ]
    assert first_suggestion.metadata_json["display_copy"] == {
        "title": "이메일 링크 반응 고객",
        "audience_summary": "분석 대상 74200명 중 120명 · 2%",
        "signal_chips": ["이메일 링크 클릭", "지역 선호", "무료 취소 선호"],
        "reason": "예약 전환 목표에 가까운 행동 패턴을 보인 고객군입니다.",
        "action_hint": "이메일 예약 혜택 메시지의 우선 타겟으로 적합합니다.",
    }
    target_profile = result.target_segments[0].profile_json
    assert "primary_signals" not in target_profile
    assert target_profile["score_components"] == {
        "promotion_cluster_similarity": 0.92,
        "cluster_quality": 0.7,
        "sample_size": 0.5,
        "final_score": 0.88,
    }
    content_brief = result.target_segments[0].content_brief_json
    assert content_brief["readiness"]["level"] == "partial"
    assert content_brief["readiness"]["missing_sections"] == ["primary_signals"]
    assert content_brief["audience_evidence"] == {
        "score_components": {
            "promotion_cluster_similarity": 0.92,
            "cluster_quality": 0.7,
            "sample_size": 0.5,
            "final_score": 0.88,
        },
        "promotion_vector_basis": {
            "channel": "email",
            "goal_metric": "booking_conversion_rate",
        },
        "promotion_matched_features": [
            "Campaign redirect users",
            "Hotel market bucket 2 affinity users",
            "Free cancellation seekers",
        ],
    }
    assert content_brief["audience_evidence"]["score_components"] == {
        "promotion_cluster_similarity": 0.92,
        "cluster_quality": 0.7,
        "sample_size": 0.5,
        "final_score": 0.88,
    }
    assert "behavior_metrics" not in content_brief["audience_evidence"]
    assert "behavior_metrics" not in str(content_brief)
    assert "top_common_features" not in str(content_brief)
    assert "recommendation_score" not in str(content_brief)


def test_service_preserves_existing_generation_primary_signals() -> None:
    promotion = promotion_record(channel="email")
    ai_segment = replace(
        segment_record(
            "seg_ai_cluster_promo_email_001_1_explicit",
            source="ai_suggested",
            sample_size=120,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_101", "user_102"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_email_001_1_explicit",
            "source": "user_vector_clustering",
            "cluster_score": 0.7,
            "primary_signals": ["explicit_analysis_signal"],
            "score_components": {"explicit_score": 0.73},
            "promotion_vector_basis": {"goal_metric": "inflow_rate"},
            "promotion_matched_features": ["Explicit matched feature"],
            "top_common_features": ["Booking conversion ready users"],
            "recommendation_score": 0.99,
            "behavior_metrics": {"booking_conversion_rate": 0.42},
        },
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=FakeSegmentSuggester([ai_segment]),
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert result.target_segments[0].profile_json["primary_signals"] == [
        "explicit_analysis_signal"
    ]
    assert result.target_segments[0].content_brief_json["audience_evidence"][
        "primary_signals"
    ] == ["explicit_analysis_signal"]
    assert result.target_segments[0].content_brief_json["audience_evidence"] == {
        "primary_signals": ["explicit_analysis_signal"],
        "score_components": {"explicit_score": 0.73},
        "promotion_vector_basis": {"goal_metric": "inflow_rate"},
        "promotion_matched_features": ["Explicit matched feature"],
    }
    assert result.target_segments[0].content_brief_json["readiness"] == {
        "level": "evidence_ready",
        "missing_sections": [],
        "available_sections": [
            "segment_snapshot",
            "promotion_context",
            "fallback_guidance",
            "source_refs",
            "audience_evidence",
        ],
    }


def test_service_excludes_empty_and_unstructured_generation_evidence() -> None:
    promotion = promotion_record(channel="email")
    invalid_profile = {
        "primary_segment": "seg_ai_cluster_promo_email_001_1_invalid",
        "source": "user_vector_clustering",
        "primary_signals": "not-a-sequence",
        "score_components": ["not-a-mapping"],
        "promotion_vector_basis": {},
        "promotion_matched_features": [],
        "top_common_features": ["Campaign redirect users"],
        "recommendation_score": 0.91,
    }
    ai_segment = replace(
        segment_record(
            "seg_ai_cluster_promo_email_001_1_invalid",
            source="ai_suggested",
            sample_size=120,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_101", "user_102"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json=invalid_profile,
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=FakeSegmentSuggester([ai_segment]),
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    target_segment = result.target_segments[0]
    assert target_segment.profile_json == invalid_profile
    assert "audience_evidence" not in target_segment.content_brief_json
    assert target_segment.content_brief_json["readiness"]["level"] == "fallback_only"
    assert target_segment.content_brief_json["readiness"]["missing_sections"] == [
        "primary_signals",
        "score_components",
    ]


def test_service_does_not_derive_raw_event_conditions_for_generation_evidence() -> None:
    promotion = promotion_record(channel="email")
    ai_segment = replace(
        segment_record(
            "seg_ai_raw_promo_email_001_1_funnel_recovery",
            source="ai_suggested",
            sample_size=42,
            rule_json={
                "source": "raw_event_intent",
                "candidate_type": "funnel_recovery",
                "compiled_conditions": [
                    "booking_start_without_complete",
                    "hotel_detail_view",
                    "recent_destination_search",
                    "price_sensitive",
                ],
                "candidate_user_ids": ["user_101", "user_102"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_raw_promo_email_001_1_funnel_recovery",
            "source": "raw_event_intent",
            "rank_role": "예약 이탈 회수형",
            "candidate_type": "funnel_recovery",
            "recommendation_score": 0.84,
            "score_components": {
                "promotion_condition_match": 0.8,
                "expected_goal_performance": 0.7,
                "final_score": 0.84,
            },
            "matched_conditions": [
                "예약 시작 후 미완료",
                "호텔 상세 조회",
                "목적지 숙소 검색",
            ],
            "signal_chips": ["예약 시작", "예약 미완료", "호텔 상세 조회"],
            "signal_metrics": {
                "booking_start_user_count": 42,
                "booking_complete_user_count": 0,
            },
        },
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=FakeSegmentSuggester([ai_segment]),
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    target_segment = result.target_segments[0]
    assert "primary_signals" not in target_segment.profile_json
    assert target_segment.content_brief_json["audience_evidence"][
        "score_components"
    ] == {
        "promotion_condition_match": 0.8,
        "expected_goal_performance": 0.7,
        "final_score": 0.84,
    }
    assert target_segment.content_brief_json["readiness"]["level"] == "partial"
    assert target_segment.content_brief_json["readiness"]["missing_sections"] == [
        "primary_signals"
    ]
    assert "signal_chips" not in target_segment.content_brief_json["audience_evidence"]
    assert "matched_conditions" not in target_segment.content_brief_json[
        "audience_evidence"
    ]


def test_service_ranks_ai_clusters_by_booking_propensity_model() -> None:
    promotion = promotion_record(channel="onsite_banner")
    high_cluster_low_booking = replace(
        segment_record(
            "seg_ai_cluster_promo_onsite_banner_001_1_highcluster",
            source="ai_suggested",
            sample_size=1800,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_low_001", "user_low_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_onsite_banner_001_1_highcluster",
            "source": "user_vector_clustering",
            "cluster_score": 0.99,
        },
    )
    lower_cluster_high_booking = replace(
        segment_record(
            "seg_ai_cluster_promo_onsite_banner_001_2_ml",
            source="ai_suggested",
            sample_size=1800,
            rule_json={
                "source": "user_vector_clustering",
                "candidate_user_ids": ["user_ml_001", "user_ml_002"],
            },
        ),
        campaign_id=promotion.campaign_id,
        promotion_id=promotion.promotion_id,
        profile_json={
            "primary_segment": "seg_ai_cluster_promo_onsite_banner_001_2_ml",
            "source": "user_vector_clustering",
            "cluster_score": 0.55,
        },
    )
    service, _, _ = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=FakeSegmentSuggester(
            [high_cluster_low_booking, lower_cluster_high_booking]
        ),
        booking_training_records=[
            BookingTrainingRecord(
                is_mobile=0,
                is_package=0,
                stay_nights=1,
                days_until_checkin=50,
                event_count=100,
                booking_count=4,
            ),
            BookingTrainingRecord(
                is_mobile=1,
                is_package=1,
                stay_nights=4,
                days_until_checkin=3,
                event_count=100,
                booking_count=72,
            ),
        ],
        user_profile_summaries={
            high_cluster_low_booking.segment_id: HotelMarketingProfileRecord(
                project_id="hotel-client-a",
                profile_name=high_cluster_low_booking.segment_id,
                profile_json={
                    "event_count": 80,
                    "booking_count": 2,
                    "mobile_ratio": 0.0,
                    "package_ratio": 0.0,
                    "avg_stay_nights": 1.0,
                    "avg_days_until_checkin": 50.0,
                },
            ),
            lower_cluster_high_booking.segment_id: HotelMarketingProfileRecord(
                project_id="hotel-client-a",
                profile_name=lower_cluster_high_booking.segment_id,
                profile_json={
                    "event_count": 80,
                    "booking_count": 50,
                    "mobile_ratio": 1.0,
                    "package_ratio": 1.0,
                    "avg_stay_nights": 4.0,
                    "avg_days_until_checkin": 3.0,
                },
            ),
        },
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments)[:2] == [
        lower_cluster_high_booking.segment_id,
        high_cluster_low_booking.segment_id,
    ]
    first_score = result.segment_suggestions[0].score_json
    second_score = result.segment_suggestions[1].score_json
    assert first_score["cluster_score"] < second_score["cluster_score"]
    assert first_score["booking_propensity_score"] > second_score[
        "booking_propensity_score"
    ]
    assert first_score["booking_propensity_model"] == "booking_propensity_logistic_v1"
    assert first_score["booking_propensity_training_sample_count"] == 2
    assert result.segment_suggestions[0].reason_json["ml_model"] == (
        "booking_propensity_logistic_v1"
    )
    assert result.segment_suggestions[0].reason_json["ml_features"] == {
        "mobile_ratio": 1.0,
        "package_ratio": 1.0,
        "stay_nights_scaled": 0.285714,
        "near_checkin_score": 0.95,
    }


def test_service_falls_back_to_default_segments_when_no_ai_suggestions_exist() -> None:
    promotion = promotion_record(channel="onsite_banner")
    suggester = FakeSegmentSuggester([])
    service, _, segment_definition_repository = build_service(
        promotion=promotion,
        segments=default_segments(),
        segment_suggester=suggester,
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_family_trip",
        "seg_mobile_user",
        "seg_repeat_hotel_no_booking",
    ]
    assert segment_definition_repository.saved_ai_suggested == []
    assert suggester.calls == [(promotion, None)]


def test_service_skips_zero_size_default_segments() -> None:
    promotion = promotion_record(channel="email")
    service, analysis_repository, _ = build_service(
        promotion=promotion,
        segments=[
            segment_record("seg_mobile_user", sample_size=0),
            segment_record("seg_family_trip", sample_size=0),
            segment_record("seg_near_checkin", sample_size=35),
            segment_record("seg_existing_all", sample_size=20),
        ],
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert segment_ids(result.target_segments) == [
        "seg_near_checkin",
        "seg_existing_all",
    ]
    assert suggestion_segment_ids(result.segment_suggestions) == [
        "seg_near_checkin",
        "seg_existing_all",
    ]
    assert all(
        suggestion.score_json["estimated_size"] > 0
        for suggestion in analysis_repository.saved.segment_suggestions or []
    )


def test_service_reflects_operator_instruction_in_content_brief() -> None:
    promotion = promotion_record(channel="onsite_banner")
    service, _, _ = build_service(promotion=promotion, segments=default_segments())

    result = service.analyze(
        analysis_request(
            promotion_id=promotion.promotion_id,
            operator_instruction="Emphasize breakfast and same-day availability.",
        ),
    )

    assert result.analysis.operator_instruction == (
        "Emphasize breakfast and same-day availability."
    )
    assert result.target_segments[0].content_brief_json["operator_instruction"] == (
        "Emphasize breakfast and same-day availability."
    )


def test_service_marks_small_segments_as_low_priority() -> None:
    promotion = promotion_record(channel="email", min_sample_size=1000)
    service, _, _ = build_service(
        promotion=promotion,
        segments=[
            segment_record("seg_mobile_user", sample_size=250),
            segment_record("seg_family_trip", sample_size=2000),
            segment_record("seg_near_checkin", sample_size=2000),
            segment_record("seg_existing_all", sample_size=2000),
        ],
    )

    result = service.analyze(
        analysis_request(promotion_id=promotion.promotion_id),
    )

    assert result.target_segments[0].segment_id == "seg_mobile_user"
    assert result.target_segments[0].priority == "low"


def test_service_raises_when_promotion_is_missing() -> None:
    service, _, _ = build_service(
        promotion=None,
        segments=default_segments(),
    )

    with pytest.raises(PromotionNotFoundError):
        service.analyze(analysis_request(promotion_id="promo_missing"))


def test_service_raises_when_no_candidate_matches() -> None:
    promotion = promotion_record(channel="email")
    service, _, _ = build_service(
        promotion=promotion,
        segments=[],
    )

    with pytest.raises(SegmentSelectionError):
        service.analyze(analysis_request(promotion_id=promotion.promotion_id))
