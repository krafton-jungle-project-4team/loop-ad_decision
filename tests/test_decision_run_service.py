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
    RunValidationError,
    build_bounded_decision_id,
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


def test_run_service_rejects_duplicate_promotion_loop_without_writes() -> None:
    service, repos = make_service(run_exists=True)

    with pytest.raises(RunConflictError, match="promotion_id and loop_count"):
        service.create_run(
            promotion_id="promo_banner_001",
            request=RunCreateRequest(loop_count=2),
        )

    assert repos.runs.exists_calls == [("promo_banner_001", 2)]
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


def test_run_service_creates_run_and_one_ad_experiment_per_target_segment() -> None:
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
    ]
    assert all(experiment.loop_count == run.loop_count for experiment in experiments)
    assert all(
        experiment.status == AdExperimentStatus.PLANNED.value
        for experiment in experiments
    )
    assert experiments[0].content_id == "content_family_001"
    assert experiments[1].content_option_id == "mobile_option_a"
    assert response.promotion_run_id == run.promotion_run_id
    assert [item.segment_id for item in response.ad_experiments] == [
        "seg_family_trip",
        "seg_mobile_user",
    ]


def test_run_service_creates_next_loop_run_after_admin_approval() -> None:
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
    assert [experiment.segment_id for experiment in experiments] == ["seg_luxury"]
    assert experiments[0].content_id == "content_luxury_approved_001"


def test_bounded_decision_id_is_stable_and_under_contract_length() -> None:
    long_promotion_id = "promo_" + ("very_long_hotel_campaign_" * 10)

    first = build_bounded_decision_id("prun", long_promotion_id, "loop_12")
    second = build_bounded_decision_id("prun", long_promotion_id, "loop_12")

    assert first == second
    assert first.startswith("prun_")
    assert len(first) <= 100


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

    def list_for_analysis(
        self,
        analysis_id: str,
    ) -> list[PromotionTargetSegmentRecord]:
        self.calls.append(analysis_id)
        return self.segments


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
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.exists_calls: list[tuple[str, int]] = []
        self.inserted: list[PromotionRunWrite] = []

    def insert(self, run: PromotionRunWrite) -> None:
        self.inserted.append(run)

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        return None

    def exists_for_promotion_loop(self, *, promotion_id: str, loop_count: int) -> bool:
        self.exists_calls.append((promotion_id, loop_count))
        return self.exists


class FakeAdExperimentRepository:
    def __init__(self, existing_segments: set[str] | None = None) -> None:
        self.existing_segments = existing_segments or set()
        self.exists_calls: list[tuple[str, str]] = []
        self.inserted_batches: list[list[AdExperimentWrite]] = []

    def insert_many(self, experiments: list[AdExperimentWrite]) -> None:
        self.inserted_batches.append(list(experiments))

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        return []

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
        run_exists: bool = False,
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
        self.runs = FakePromotionRunRepository(exists=run_exists)
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
    run_exists: bool = False,
) -> tuple[PromotionRunService, FakeRepositoryBundle]:
    repos = FakeRepositoryBundle(
        promotion=promotion,
        analysis=analysis,
        latest_analysis=latest_analysis,
        generation=generation,
        latest_generation=latest_generation,
        target_segments=target_segments,
        candidates=candidates,
        run_exists=run_exists,
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
    status: str = "completed",
) -> GenerationRunRecord:
    return GenerationRunRecord(
        generation_id=generation_id,
        analysis_id=analysis_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        content_option_count=2,
        operator_instruction=None,
        input_json={"analysis_id": analysis_id},
        output_json={"content_count": 1},
        generation_report_json={"status": "completed"},
        status=status,
    )


def target_segment_record(
    *,
    analysis_id: str = "analysis_banner_001",
    segment_id: str = "seg_family_trip",
    segment_name: str = "Family hotel trip",
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
        status="planned",
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
