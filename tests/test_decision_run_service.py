from __future__ import annotations

from decimal import Decimal

import pytest

from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWrite,
    ContentCandidateRecord,
    GenerationRunRecord,
    PromotionAnalysisRecord,
    PromotionRecord,
    PromotionRunRecord,
    PromotionRunWrite,
    PromotionTargetSegmentRecord,
)
from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.schemas import (
    AdExperimentStatus,
    Channel,
    GoalBasis,
    GoalMetric,
    PromotionRunStatus,
    RunCreateRequest,
)
from app.decision.service import (
    PromotionRunService,
    RunConflictError,
    RunSegmentScopeValidationError,
    RunValidationError,
    build_bounded_decision_id,
    build_promotion_run_id,
    build_segment_scope_fingerprint,
)


DEFAULT_LATEST = object()


def test_run_service_uses_latest_completed_analysis_and_generation() -> None:
    service, repos = make_service()

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(),
    )

    assert repos.analyses.latest_calls == ["promo_banner_001"]
    assert repos.generations.latest_calls == ["promo_banner_001"]
    assert repos.target_segments.calls == []
    assert repos.target_segments.approved_calls == [
        ("analysis_banner_001", None)
    ]
    assert response.analysis_id == "analysis_banner_001"
    assert response.generation_id == "generation_banner_001"
    assert response.status == PromotionRunStatus.PLANNED
    assert len(repos.runs.inserted) == 1
    assert repos.runs.inserted[0].goal_snapshot_json == {
        "source": "promotions",
        "promotion_id": "promo_banner_001",
        "channel": Channel.ONSITE_BANNER.value,
        "goal_metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
        "goal_target_value": "0.030000",
        "goal_basis": GoalBasis.ALL_SEGMENTS.value,
        "min_sample_size": 1000,
        "max_loop_count": 3,
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
    }


def test_run_service_rejects_generation_for_different_analysis_without_writes() -> None:
    service, repos = make_service(
        generation=generation_record(analysis_id="analysis_other_001"),
    )

    with pytest.raises(RunValidationError, match="selected promotion analysis"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(
                analysis_id="analysis_banner_001",
                generation_id="generation_banner_001",
            ),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


@pytest.mark.parametrize(
    ("missing_dependency", "message"),
    [
        ("analysis", "completed promotion analysis"),
        ("generation", "completed generation run"),
    ],
)
def test_run_service_requires_latest_completed_dependencies(
    missing_dependency: str,
    message: str,
) -> None:
    service, repos = make_service(
        latest_analysis=None
        if missing_dependency == "analysis"
        else DEFAULT_LATEST,
        latest_generation=None
        if missing_dependency == "generation"
        else DEFAULT_LATEST,
    )

    with pytest.raises(RunValidationError, match=message):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_reuses_identical_scope_without_duplicate_writes() -> None:
    service, repos = make_service()

    first = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(loop_count=2),
    )
    second = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(loop_count=2),
    )

    assert second == first
    assert len(repos.runs.inserted) == 1
    assert len(repos.ad_experiments.inserted_batches) == 1


def test_run_service_creates_distinct_runs_for_different_segment_scopes() -> None:
    generation = generation_record(target_segment_ids=["seg_a", "seg_b"])
    service, repos = make_service(
        generation=generation,
        target_segments=[
            target_segment_record(segment_id="seg_a"),
            target_segment_record(segment_id="seg_b"),
        ],
        candidates=[
            content_candidate_record(
                generation_id=generation.generation_id,
                segment_id="seg_a",
            ),
            content_candidate_record(
                generation_id=generation.generation_id,
                segment_id="seg_b",
            ),
        ],
    )

    first = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            generation_id=generation.generation_id,
            segment_ids=["seg_a"],
        ),
    )
    second = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            generation_id=generation.generation_id,
            segment_ids=["seg_b"],
        ),
    )

    assert first.promotion_run_id != second.promotion_run_id
    assert first.segment_ids == ["seg_a"]
    assert second.segment_ids == ["seg_b"]
    assert [run.loop_count for run in repos.runs.inserted] == [1, 1]


def test_run_service_reuses_scope_when_segment_input_order_changes() -> None:
    target_segments = [
        target_segment_record(segment_id="seg_a"),
        target_segment_record(segment_id="seg_b"),
    ]
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_a", "seg_b"]),
        target_segments=target_segments,
        candidates=[
            content_candidate_record(segment_id="seg_a"),
            content_candidate_record(segment_id="seg_b"),
        ],
    )

    first = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            segment_ids=[" seg_b ", "seg_a", "seg_a"]
        ),
    )
    second = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(segment_ids=["seg_a", "seg_b"]),
    )

    assert first == second
    assert first.segment_ids == ["seg_a", "seg_b"]
    assert len(repos.runs.inserted) == 1


def test_run_service_rejects_corrupted_existing_scope() -> None:
    service, repos = make_service()
    request = RunCreateRequest()
    service.create_run(promotion_id="promo_banner_001", request=request)
    repos.ad_experiments.records = [
        experiment
        for experiment in repos.ad_experiments.records
        if experiment.segment_id != FALLBACK_SEGMENT_ID
    ]

    with pytest.raises(RunConflictError, match="do not match its segment scope"):
        service.create_run(promotion_id="promo_banner_001", request=request)


def test_run_service_reuses_scope_after_concurrent_insert_loses_race() -> None:
    service, repos = make_service()
    fingerprint = build_segment_scope_fingerprint(
        segment_ids=["seg_family_trip"],
    )
    existing_run = PromotionRunRecord(
        promotion_run_id=build_bounded_decision_id(
            "prun",
            "promo_banner_001",
            "loop_1",
            fingerprint[:24],
        ),
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        loop_count=1,
        status=PromotionRunStatus.PLANNED.value,
        goal_snapshot_json={"source": "promotions"},
        segment_scope_json=["seg_family_trip"],
        segment_scope_fingerprint=fingerprint,
    )
    repos.ad_experiments.records = [
        ad_experiment_record(
            promotion_run_id=existing_run.promotion_run_id,
            segment_id="seg_family_trip",
        ),
        ad_experiment_record(
            promotion_run_id=existing_run.promotion_run_id,
            segment_id=FALLBACK_SEGMENT_ID,
        ),
    ]
    scope_lookup_count = 0

    def get_by_scope(**_scope: object) -> PromotionRunRecord | None:
        nonlocal scope_lookup_count
        scope_lookup_count += 1
        return None if scope_lookup_count == 1 else existing_run

    repos.runs.get_by_scope = get_by_scope  # type: ignore[method-assign]
    repos.runs.insert_if_absent = lambda _run: False  # type: ignore[method-assign]

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(),
    )

    assert response.promotion_run_id == existing_run.promotion_run_id
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


@pytest.mark.parametrize(
    "candidate_count",
    [0, 2],
)
def test_run_service_requires_exactly_one_content_candidate_per_segment(
    candidate_count: int,
) -> None:
    candidates = [
        content_candidate_record(content_id=f"content_family_{index:03d}")
        for index in range(candidate_count)
    ]
    service, repos = make_service(candidates=candidates)

    with pytest.raises(RunValidationError, match="exactly one approved or active"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_rejects_content_candidate_context_mismatch() -> None:
    service, repos = make_service(
        candidates=[
            content_candidate_record(project_id="other-project"),
        ],
    )

    with pytest.raises(RunValidationError, match="project_id"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_creates_run_and_fallback_ad_experiment() -> None:
    target_segments = [
        target_segment_record(
            segment_id="seg_family_trip",
            segment_name="Family hotel trip",
        ),
        target_segment_record(
            segment_id="seg_mobile_user",
            segment_name="Mobile hotel users",
        ),
    ]
    service, repos = make_service(
        target_segments=target_segments,
        candidates=[
            content_candidate_record(
                segment_id="seg_family_trip",
                content_id="content_family_001",
                content_option_id="family_option_a",
            ),
            content_candidate_record(
                segment_id="seg_mobile_user",
                content_id="content_mobile_001",
                content_option_id="mobile_option_a",
            ),
        ],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            analysis_id="analysis_banner_001",
            generation_id="generation_banner_001",
            loop_count=3,
        ),
    )

    assert len(repos.runs.inserted) == 1
    run = repos.runs.inserted[0]
    assert run.loop_count == 3
    assert run.status == PromotionRunStatus.PLANNED.value
    assert len(repos.ad_experiments.inserted_batches) == 1
    experiments = repos.ad_experiments.inserted_batches[0]
    assert [experiment.segment_id for experiment in experiments] == [
        "seg_family_trip",
        "seg_mobile_user",
        FALLBACK_SEGMENT_ID,
    ]
    assert all(experiment.loop_count == run.loop_count for experiment in experiments)
    assert all(
        experiment.status == AdExperimentStatus.PLANNED.value
        for experiment in experiments
    )
    assert experiments[0].content_id == "content_family_001"
    assert experiments[1].content_option_id == "mobile_option_a"
    assert experiments[2].content_id == "content_family_001"
    assert experiments[2].content_option_id == "family_option_a"
    assert response.promotion_run_id == run.promotion_run_id
    assert response.segment_ids == ["seg_family_trip", "seg_mobile_user"]
    assert [item.segment_id for item in response.ad_experiments] == [
        "seg_family_trip",
        "seg_mobile_user",
        FALLBACK_SEGMENT_ID,
    ]
    assert [item.is_fallback for item in response.ad_experiments] == [
        False,
        False,
        True,
    ]


def test_run_service_creates_experiments_only_for_requested_approved_segments() -> None:
    target_segments = [
        target_segment_record(
            segment_id="seg_family_trip",
            segment_name="Family hotel trip",
            status="approved",
        ),
        target_segment_record(
            segment_id="seg_mobile_user",
            segment_name="Mobile hotel users",
            status="approved",
        ),
    ]
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_mobile_user"]),
        target_segments=target_segments,
        candidates=[
            content_candidate_record(
                segment_id="seg_family_trip",
                content_id="content_family_001",
                content_option_id="family_option_a",
            ),
            content_candidate_record(
                segment_id="seg_mobile_user",
                content_id="content_mobile_001",
                content_option_id="mobile_option_a",
            ),
        ],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            analysis_id="analysis_banner_001",
            generation_id="generation_banner_001",
            segment_ids=["seg_mobile_user"],
        ),
    )

    assert repos.target_segments.calls == []
    assert repos.target_segments.approved_calls == [
        ("analysis_banner_001", ["seg_mobile_user"])
    ]
    assert [experiment.segment_id for experiment in response.ad_experiments] == [
        "seg_mobile_user",
        FALLBACK_SEGMENT_ID,
    ]


def test_run_service_rejects_requested_segment_ids_that_are_not_approved_without_writes() -> None:
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_family_trip"]),
        target_segments=[target_segment_record(status="planned")],
    )

    with pytest.raises(RunValidationError, match="segment_ids"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(segment_ids=["seg_family_trip"]),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_rejects_generation_snapshot_mismatch_without_writes() -> None:
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_family_trip"]),
        target_segments=[
            target_segment_record(
                segment_id="seg_mobile_user",
                status="approved",
            )
        ],
    )

    with pytest.raises(RunValidationError, match="generation target_segment_ids snapshot"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(segment_ids=["seg_mobile_user"]),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_omitted_segment_ids_uses_generation_snapshot_scope() -> None:
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_family_trip"]),
        target_segments=[
            target_segment_record(
                segment_id="seg_family_trip",
                status="approved",
            ),
            target_segment_record(
                segment_id="seg_mobile_user",
                status="planned",
            ),
        ],
        candidates=[content_candidate_record(segment_id="seg_family_trip")],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(),
    )

    assert repos.target_segments.approved_calls == [
        ("analysis_banner_001", ["seg_family_trip"])
    ]
    assert [experiment.segment_id for experiment in response.ad_experiments] == [
        "seg_family_trip",
        FALLBACK_SEGMENT_ID,
    ]


def test_run_service_excludes_fallback_from_snapshot_scope_and_fingerprint() -> None:
    service, repos = make_service(
        generation=generation_record(
            target_segment_ids=[FALLBACK_SEGMENT_ID, "seg_family_trip"]
        ),
        target_segments=[
            target_segment_record(segment_id="seg_family_trip"),
            target_segment_record(segment_id=FALLBACK_SEGMENT_ID),
        ],
        candidates=[
            content_candidate_record(segment_id="seg_family_trip"),
            content_candidate_record(segment_id=FALLBACK_SEGMENT_ID),
        ],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(),
    )

    assert response.segment_ids == ["seg_family_trip"]
    assert repos.runs.inserted[0].segment_scope_fingerprint == (
        build_segment_scope_fingerprint(segment_ids=["seg_family_trip"])
    )
    assert [item.segment_id for item in response.ad_experiments] == [
        "seg_family_trip",
        FALLBACK_SEGMENT_ID,
    ]


def test_run_service_omitted_segment_ids_requires_snapshot_segments_to_be_approved() -> None:
    service, repos = make_service(
        generation=generation_record(target_segment_ids=["seg_mobile_user"]),
        target_segments=[target_segment_record(status="approved")],
    )

    with pytest.raises(RunValidationError, match="approved promotion_target_segments"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_uses_dedicated_fallback_content_when_available() -> None:
    service, repos = make_service(
        target_segments=[
            target_segment_record(
                segment_id="seg_family_trip",
                segment_name="Family hotel trip",
            ),
        ],
        candidates=[
            content_candidate_record(
                segment_id="seg_family_trip",
                content_id="content_family_001",
                content_option_id="family_option_a",
            ),
            content_candidate_record(
                segment_id=FALLBACK_SEGMENT_ID,
                content_id="content_existing_all_001",
                content_option_id="existing_all_option_a",
            ),
        ],
    )

    service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(),
    )

    experiments = repos.ad_experiments.inserted_batches[0]
    assert [experiment.segment_id for experiment in experiments] == [
        "seg_family_trip",
        FALLBACK_SEGMENT_ID,
    ]
    assert experiments[1].content_id == "content_existing_all_001"
    assert experiments[1].content_option_id == "existing_all_option_a"


def test_run_service_creates_next_loop_run_from_approved_focus_candidate() -> None:
    analysis_id = "analysis_banner_001_loop_2"
    generation_id = "generation_banner_001_loop_2"
    service, repos = make_service(
        analysis=analysis_record(analysis_id=analysis_id),
        generation=generation_record(
            generation_id=generation_id,
            analysis_id=analysis_id,
        ),
        target_segments=[
            target_segment_record(
                analysis_id=analysis_id,
                segment_id="seg_luxury",
                segment_name="Luxury hotel users",
            ),
        ],
        candidates=[
            content_candidate_record(
                analysis_id=analysis_id,
                generation_id=generation_id,
                segment_id="seg_luxury",
                content_id="content_luxury_approved_001",
                content_option_id="luxury_option_a",
                status="approved",
            ),
        ],
    )

    response = service.create_run(
        promotion_id="promo_banner_001",
        request=RunCreateRequest(
            analysis_id=analysis_id,
            generation_id=generation_id,
            loop_count=2,
        ),
    )

    assert response.analysis_id == analysis_id
    assert response.generation_id == generation_id
    assert response.loop_count == 2
    assert len(repos.runs.inserted) == 1
    assert repos.runs.inserted[0].loop_count == 2
    experiments = repos.ad_experiments.inserted_batches[0]
    assert [experiment.segment_id for experiment in experiments] == [
        "seg_luxury",
        FALLBACK_SEGMENT_ID,
    ]
    assert experiments[0].content_id == "content_luxury_approved_001"
    assert experiments[1].content_id == "content_luxury_approved_001"


def test_bounded_decision_id_is_stable_and_under_contract_length() -> None:
    long_promotion_id = "promo_" + ("very_long_hotel_campaign_" * 10)

    first = build_bounded_decision_id("prun", long_promotion_id, "loop_12")
    second = build_bounded_decision_id("prun", long_promotion_id, "loop_12")

    assert first == second
    assert first.startswith("prun_")
    assert len(first) <= 100


def test_promotion_run_id_preserves_scope_fingerprint_for_long_promotion_id() -> None:
    long_promotion_id = "promo_" + ("very_long_hotel_campaign_" * 10)
    fingerprint = "abcdef0123456789abcdef01" + ("2" * 40)

    first = build_promotion_run_id(
        project_id="hotel-client-a",
        promotion_id=long_promotion_id,
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        loop_count=12,
        segment_scope_fingerprint=fingerprint,
    )
    second = build_promotion_run_id(
        project_id="hotel-client-a",
        promotion_id=long_promotion_id,
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        loop_count=12,
        segment_scope_fingerprint=fingerprint,
    )

    assert first == second
    assert f"_loop_12_{fingerprint[:24]}_" in first
    assert fingerprint[:24] in first
    assert len(first) <= 100


def test_segment_scope_fingerprint_is_order_and_duplicate_invariant() -> None:
    assert build_segment_scope_fingerprint(
        segment_ids=["seg_b", FALLBACK_SEGMENT_ID, "seg_a", "seg_a"],
    ) == build_segment_scope_fingerprint(
        segment_ids=["seg_a", "seg_b"],
    )


@pytest.mark.parametrize(
    ("changed_field", "changed_value"),
    [
        ("analysis_id", "analysis_banner_002"),
        ("generation_id", "generation_banner_002"),
        ("loop_count", 2),
    ],
)
def test_promotion_run_id_changes_with_non_segment_scope_fields(
    changed_field: str,
    changed_value: object,
) -> None:
    base: dict[str, object] = {
        "project_id": "hotel-client-a",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
    }
    changed = {**base, changed_field: changed_value}
    fingerprint = build_segment_scope_fingerprint(segment_ids=["seg_a"])
    assert build_promotion_run_id(
        project_id=str(base["project_id"]),
        promotion_id=str(base["promotion_id"]),
        analysis_id=str(base["analysis_id"]),
        generation_id=str(base["generation_id"]),
        loop_count=int(base["loop_count"]),
        segment_scope_fingerprint=fingerprint,
    ) != build_promotion_run_id(
        project_id=str(changed["project_id"]),
        promotion_id=str(changed["promotion_id"]),
        analysis_id=str(changed["analysis_id"]),
        generation_id=str(changed["generation_id"]),
        loop_count=int(changed["loop_count"]),
        segment_scope_fingerprint=fingerprint,
    )


def test_segment_scope_fingerprint_and_run_id_change_with_segment_scope() -> None:
    first_fingerprint = build_segment_scope_fingerprint(segment_ids=["seg_a"])
    second_fingerprint = build_segment_scope_fingerprint(segment_ids=["seg_b"])
    common = {
        "project_id": "hotel-client-a",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "generation_id": "generation_banner_001",
        "loop_count": 1,
    }

    assert first_fingerprint != second_fingerprint
    assert build_promotion_run_id(
        **common,
        segment_scope_fingerprint=first_fingerprint,
    ) != build_promotion_run_id(
        **common,
        segment_scope_fingerprint=second_fingerprint,
    )


@pytest.mark.parametrize("segment_ids", [[], ["   "], ["seg_a", "  "]])
def test_run_service_rejects_empty_or_blank_explicit_scope_without_writes(
    segment_ids: list[str],
) -> None:
    service, repos = make_service()

    with pytest.raises(RunSegmentScopeValidationError, match="segment_ids"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(segment_ids=segment_ids),
        )

    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_rejects_fallback_in_explicit_scope_without_writes() -> None:
    service, repos = make_service()

    with pytest.raises(RunSegmentScopeValidationError, match="fallback"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(
                segment_ids=["seg_family_trip", FALLBACK_SEGMENT_ID]
            ),
        )

    assert repos.promotions.calls == []
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


def test_run_service_rejects_explicit_scope_when_feature_flag_is_off() -> None:
    service, repos = make_service(partial_segment_scope_enabled=False)

    with pytest.raises(RunConflictError, match="scope is disabled"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(segment_ids=["seg_family_trip"]),
        )

    assert repos.promotions.calls == []
    assert repos.runs.inserted == []
    assert repos.ad_experiments.inserted_batches == []


class FakePromotionRepository:
    def __init__(self, promotion: PromotionRecord | None) -> None:
        self.promotion = promotion
        self.calls: list[str] = []

    def get_by_id(self, promotion_id: str) -> PromotionRecord | None:
        self.calls.append(promotion_id)
        return self.promotion


class FakePromotionAnalysisRepository:
    def __init__(
        self,
        *,
        analysis: PromotionAnalysisRecord | None,
        latest: PromotionAnalysisRecord | None,
    ) -> None:
        self.analysis = analysis
        self.latest = latest
        self.get_calls: list[str] = []
        self.latest_calls: list[str] = []

    def get_by_id(self, analysis_id: str) -> PromotionAnalysisRecord | None:
        self.get_calls.append(analysis_id)
        return self.analysis

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> PromotionAnalysisRecord | None:
        self.latest_calls.append(promotion_id)
        return self.latest


class FakeGenerationRunRepository:
    def __init__(
        self,
        *,
        generation: GenerationRunRecord | None,
        latest: GenerationRunRecord | None,
    ) -> None:
        self.generation = generation
        self.latest = latest
        self.get_calls: list[str] = []
        self.latest_calls: list[str] = []

    def get_by_id(self, generation_id: str) -> GenerationRunRecord | None:
        self.get_calls.append(generation_id)
        return self.generation

    def get_latest_completed_for_promotion(
        self,
        promotion_id: str,
    ) -> GenerationRunRecord | None:
        self.latest_calls.append(promotion_id)
        return self.latest


class FakePromotionTargetSegmentRepository:
    def __init__(self, segments: list[PromotionTargetSegmentRecord]) -> None:
        self.segments = segments
        self.calls: list[str] = []
        self.approved_calls: list[tuple[str, list[str] | None]] = []

    def list_for_analysis(
        self,
        analysis_id: str,
    ) -> list[PromotionTargetSegmentRecord]:
        self.calls.append(analysis_id)
        return self.segments

    def list_approved_for_analysis(
        self,
        analysis_id: str,
        segment_ids: list[str] | None = None,
    ) -> list[PromotionTargetSegmentRecord]:
        self.approved_calls.append(
            (analysis_id, list(segment_ids) if segment_ids is not None else None)
        )
        if segment_ids is None:
            return [segment for segment in self.segments if segment.status == "approved"]
        requested_ids = set(segment_ids)
        return [
            segment
            for segment in self.segments
            if segment.segment_id in requested_ids and segment.status == "approved"
        ]


class FakeContentCandidateRepository:
    def __init__(self, candidates: list[ContentCandidateRecord]) -> None:
        self.candidates = candidates
        self.calls: list[str] = []

    def list_approved_or_active_for_generation(
        self,
        generation_id: str,
    ) -> list[ContentCandidateRecord]:
        self.calls.append(generation_id)
        return self.candidates


class FakePromotionRunRepository:
    def __init__(self) -> None:
        self.inserted: list[PromotionRunWrite] = []
        self.records: list[PromotionRunRecord] = []

    def insert_if_absent(self, run: PromotionRunWrite) -> bool:
        if self.get_by_scope(
            project_id=run.project_id,
            promotion_id=run.promotion_id,
            analysis_id=run.analysis_id,
            generation_id=run.generation_id,
            segment_scope_fingerprint=run.segment_scope_fingerprint,
            loop_count=run.loop_count,
        ) is not None:
            return False
        self.inserted.append(run)
        self.records.append(PromotionRunRecord(**vars(run)))
        return True

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        return next(
            (
                run
                for run in self.records
                if run.promotion_run_id == promotion_run_id
            ),
            None,
        )

    def get_by_scope(
        self,
        *,
        project_id: str,
        promotion_id: str,
        analysis_id: str,
        generation_id: str,
        segment_scope_fingerprint: str,
        loop_count: int,
    ) -> PromotionRunRecord | None:
        return next(
            (
                run
                for run in self.records
                if run.project_id == project_id
                and run.promotion_id == promotion_id
                and run.analysis_id == analysis_id
                and run.generation_id == generation_id
                and run.segment_scope_fingerprint == segment_scope_fingerprint
                and run.loop_count == loop_count
            ),
            None,
        )


class FakeAdExperimentRepository:
    def __init__(self, existing_segments: set[str] | None = None) -> None:
        self.existing_segments = existing_segments or set()
        self.exists_calls: list[tuple[str, str]] = []
        self.inserted_batches: list[list[AdExperimentWrite]] = []
        self.records: list[AdExperimentRecord] = []

    def insert_many(self, experiments: list[AdExperimentWrite]) -> None:
        self.inserted_batches.append(list(experiments))
        self.records.extend(AdExperimentRecord(**vars(item)) for item in experiments)

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        return [
            experiment
            for experiment in self.records
            if experiment.promotion_run_id == promotion_run_id
        ]

    def exists_for_run_segment(
        self,
        *,
        promotion_run_id: str,
        segment_id: str,
    ) -> bool:
        self.exists_calls.append((promotion_run_id, segment_id))
        return segment_id in self.existing_segments


class FakeRepositoryBundle:
    def __init__(
        self,
        *,
        promotion: PromotionRecord | None = None,
        analysis: PromotionAnalysisRecord | None = None,
        latest_analysis: PromotionAnalysisRecord | None | object = DEFAULT_LATEST,
        generation: GenerationRunRecord | None = None,
        latest_generation: GenerationRunRecord | None | object = DEFAULT_LATEST,
        target_segments: list[PromotionTargetSegmentRecord] | None = None,
        candidates: list[ContentCandidateRecord] | None = None,
    ) -> None:
        resolved_analysis = analysis if analysis is not None else analysis_record()
        resolved_generation = (
            generation if generation is not None else generation_record()
        )
        self.promotions = FakePromotionRepository(promotion or promotion_record())
        self.analyses = FakePromotionAnalysisRepository(
            analysis=resolved_analysis,
            latest=(
                resolved_analysis
                if latest_analysis is DEFAULT_LATEST
                else latest_analysis
            ),
        )
        self.generations = FakeGenerationRunRepository(
            generation=resolved_generation,
            latest=(
                resolved_generation
                if latest_generation is DEFAULT_LATEST
                else latest_generation
            ),
        )
        self.target_segments = FakePromotionTargetSegmentRepository(
            target_segments
            if target_segments is not None
            else [
                target_segment_record(),
            ],
        )
        self.contents = FakeContentCandidateRepository(
            candidates
            if candidates is not None
            else [
                content_candidate_record(),
            ],
        )
        self.runs = FakePromotionRunRepository()
        self.ad_experiments = FakeAdExperimentRepository()


def make_service(
    *,
    promotion: PromotionRecord | None = None,
    analysis: PromotionAnalysisRecord | None = None,
    latest_analysis: PromotionAnalysisRecord | None | object = DEFAULT_LATEST,
    generation: GenerationRunRecord | None = None,
    latest_generation: GenerationRunRecord | None | object = DEFAULT_LATEST,
    target_segments: list[PromotionTargetSegmentRecord] | None = None,
    candidates: list[ContentCandidateRecord] | None = None,
    partial_segment_scope_enabled: bool = True,
) -> tuple[PromotionRunService, FakeRepositoryBundle]:
    repos = FakeRepositoryBundle(
        promotion=promotion,
        analysis=analysis,
        latest_analysis=latest_analysis,
        generation=generation,
        latest_generation=latest_generation,
        target_segments=target_segments,
        candidates=candidates,
    )
    return (
        PromotionRunService(
            promotion_repository=repos.promotions,
            promotion_analysis_repository=repos.analyses,
            promotion_target_segment_repository=repos.target_segments,
            generation_run_repository=repos.generations,
            content_candidate_repository=repos.contents,
            promotion_run_repository=repos.runs,
            ad_experiment_repository=repos.ad_experiments,
            partial_segment_scope_enabled=partial_segment_scope_enabled,
        ),
        repos,
    )


def promotion_record() -> PromotionRecord:
    return PromotionRecord(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        channel=Channel.ONSITE_BANNER.value,
        goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        goal_target_value=Decimal("0.030000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
        min_sample_size=1000,
        max_loop_count=3,
    )


def analysis_record(
    *,
    analysis_id: str = "analysis_banner_001",
    status: str = "completed",
) -> PromotionAnalysisRecord:
    return PromotionAnalysisRecord(
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        focus_segment_ids_json=None,
        operator_instruction=None,
        input_snapshot_json={"promotion_id": "promo_banner_001"},
        profile_summary_json={"selected_segment_count": 1},
        output_json={"target_segment_count": 1},
        status=status,
    )


def generation_record(
    *,
    generation_id: str = "generation_banner_001",
    analysis_id: str = "analysis_banner_001",
    target_segment_ids: list[str] | None = None,
    status: str = "completed",
) -> GenerationRunRecord:
    input_json = {"analysis_id": analysis_id}
    if target_segment_ids is not None:
        input_json["target_segment_ids"] = target_segment_ids
    return GenerationRunRecord(
        generation_id=generation_id,
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        content_option_count=2,
        operator_instruction=None,
        input_json=input_json,
        output_json={"content_count": 1},
        generation_report_json={"status": "completed"},
        status=status,
    )


def target_segment_record(
    *,
    analysis_id: str = "analysis_banner_001",
    segment_id: str = "seg_family_trip",
    segment_name: str = "Family hotel trip",
    status: str = "approved",
) -> PromotionTargetSegmentRecord:
    return PromotionTargetSegmentRecord(
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        segment_id=segment_id,
        segment_name=segment_name,
        segment_vector_id=f"segvec_{segment_id}_v1",
        rule_json={"segment_id": segment_id},
        profile_json={"segment_id": segment_id},
        content_brief_json={"message_direction": "Highlight hotel benefits."},
        data_evidence_json={"event_count": 120},
        estimated_size=1200,
        priority="high",
        status=status,
    )


def content_candidate_record(
    *,
    content_id: str = "content_family_001",
    content_option_id: str = "family_option_a",
    analysis_id: str = "analysis_banner_001",
    generation_id: str = "generation_banner_001",
    segment_id: str = "seg_family_trip",
    project_id: str = "hotel-client-a",
    status: str = "approved",
) -> ContentCandidateRecord:
    return ContentCandidateRecord(
        content_id=content_id,
        content_option_id=content_option_id,
        generation_id=generation_id,
        analysis_id=analysis_id,
        project_id=project_id,
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        segment_id=segment_id,
        channel=Channel.ONSITE_BANNER.value,
        status=status,
    )


def ad_experiment_record(
    *,
    promotion_run_id: str,
    segment_id: str,
) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=f"adexp_{segment_id}",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id=promotion_run_id,
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        segment_id=segment_id,
        segment_name=segment_id,
        content_id="content_family_001",
        content_option_id="family_option_a",
        channel=Channel.ONSITE_BANNER.value,
        loop_count=1,
        status=AdExperimentStatus.PLANNED.value,
        goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        goal_target_value=Decimal("0.030000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
    )
