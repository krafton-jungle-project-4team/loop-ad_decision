from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.decision.evaluation_service import (
    AdExperimentEvaluationNotFoundError,
    AdExperimentEvaluationService,
    AdExperimentEvaluationValidationError,
    EvaluationContext,
)
from app.decision.repositories import (
    AdExperimentRecord,
    ContentCandidateRecord,
    MetricCountRecord,
    PromotionEvaluationWrite,
    PromotionRunRecord,
)
from app.decision.schemas import (
    AdExperimentEvaluateRequest,
    AdExperimentStatus,
    Channel,
    GoalBasis,
    GoalMetric,
    PromotionEvaluationStatus,
    PromotionRunStatus,
)


DEFAULT_EXPERIMENT = object()
DEFAULT_CANDIDATE = object()


def test_ad_experiment_evaluation_calculates_inflow_goal_met() -> None:
    repos = FakeEvaluationRepos(
        experiment=ad_experiment_record(goal_metric=GoalMetric.INFLOW_RATE.value),
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_target_value": "0.500000",
                "min_sample_size": 10,
            }
        ),
        counts=MetricCountRecord(numerator_count=5, denominator_count=10),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )
    assert response.metric == GoalMetric.INFLOW_RATE
    assert response.promotion_id == "promo_banner_001"
    assert response.segment_id == "seg_family_trip"
    assert response.actual_value == Decimal("0.500000")
    assert response.target_gap == Decimal("0.000000")
    assert response.status_reason == "target_met"
    assert response.strategy_snapshot.strategy_key == "booking_confidence__family"
    assert response.status == PromotionEvaluationStatus.GOAL_MET
    assert response.next_loop_required is False
    assert response.feedback is None
    assert len(repos.evaluations.inserted) == 1
    inserted = repos.evaluations.inserted[0]
    assert inserted.basis == GoalBasis.ALL_SEGMENTS.value
    assert inserted.target_value == Decimal("0.500000")
    assert inserted.project_id == "hotel-client-a"
    assert inserted.segment_id == "seg_family_trip"
    assert inserted.content_id == "content_family_trip_001"
    assert inserted.result_json["event_names"] == {
        "numerator": "campaign_landing",
        "denominator": "campaign_redirect_click",
    }
    assert inserted.result_json["evaluation_mode"] == "target_threshold"
    assert inserted.result_json["evaluation_scope"] == "ad_experiment"
    assert inserted.result_json["target_gap"] == "0.000000"
    assert inserted.result_json["sample_size"] == 10
    assert inserted.result_json["strategy_snapshot"]["evidence_refs"] == [
        "content_brief.benefit_evidence.free_cancellation"
    ]
    assert repos.experiments.status_updates == [
        ("adexp_family_trip_001", AdExperimentStatus.GOAL_MET.value)
    ]


def test_ad_experiment_evaluation_calculates_booking_goal_not_met() -> None:
    repos = FakeEvaluationRepos(
        experiment=ad_experiment_record(
            goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value
        ),
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_target_value": "0.300000",
                "min_sample_size": "10",
            }
        ),
        counts=MetricCountRecord(numerator_count=2, denominator_count=10),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.actual_value == Decimal("0.200000")
    assert response.target_gap == Decimal("-0.100000")
    assert response.status == PromotionEvaluationStatus.GOAL_NOT_MET
    assert response.next_loop_required is True
    inserted = repos.evaluations.inserted[0]
    assert inserted.next_loop_required is True
    assert inserted.result_json["event_names"] == {
        "numerator": "booking_complete",
        "denominator": "promotion_click",
    }
    assert repos.experiments.status_updates == [
        ("adexp_family_trip_001", AdExperimentStatus.GOAL_NOT_MET.value)
    ]


def test_ad_experiment_evaluation_calculates_positive_target_gap() -> None:
    repos = FakeEvaluationRepos(
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_target_value": "0.100000",
                "min_sample_size": 10,
            }
        ),
        counts=MetricCountRecord(numerator_count=2, denominator_count=10),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.target_gap == Decimal("0.100000")
    assert repos.evaluations.inserted[0].result_json["target_gap"] == "0.100000"


def test_email_booking_conversion_uses_campaign_landing_denominator() -> None:
    repos = FakeEvaluationRepos(
        experiment=ad_experiment_record(
            goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
            channel=Channel.EMAIL.value,
        ),
        counts=MetricCountRecord(numerator_count=1, denominator_count=2),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.actual_value == Decimal("0.500000")
    inserted = repos.evaluations.inserted[0]
    assert inserted.result_json["event_names"] == {
        "numerator": "booking_complete",
        "denominator": "campaign_landing",
    }


def test_ad_experiment_evaluation_preserves_denominator_when_no_bookings() -> None:
    repos = FakeEvaluationRepos(
        counts=MetricCountRecord(numerator_count=0, denominator_count=10),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.numerator_count == 0
    assert response.denominator_count == 10
    assert response.sample_size == 10
    assert response.status == PromotionEvaluationStatus.GOAL_NOT_MET


def test_ad_experiment_evaluation_marks_denominator_zero_insufficient() -> None:
    repos = FakeEvaluationRepos(
        counts=MetricCountRecord(numerator_count=0, denominator_count=0),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.actual_value == Decimal("0.000000")
    assert response.status == PromotionEvaluationStatus.INSUFFICIENT_DATA
    assert response.next_loop_required is False
    assert repos.evaluations.inserted[0].result_json["status_reason"] == (
        "denominator_zero"
    )


def test_ad_experiment_evaluation_marks_min_sample_insufficient() -> None:
    repos = FakeEvaluationRepos(
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_target_value": "0.100000",
                "min_sample_size": 50,
            }
        ),
        counts=MetricCountRecord(numerator_count=10, denominator_count=20),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.status == PromotionEvaluationStatus.INSUFFICIENT_DATA
    assert response.next_loop_required is False
    assert repos.evaluations.inserted[0].result_json["status_reason"] == (
        "sample_size_below_minimum"
    )


def test_ad_experiment_evaluation_rejects_funnel_step_rate_without_writes() -> None:
    repos = FakeEvaluationRepos(
        experiment=ad_experiment_record(goal_metric=GoalMetric.FUNNEL_STEP_RATE.value)
    )
    service = make_service(repos)

    with pytest.raises(AdExperimentEvaluationValidationError, match="funnel_step_rate"):
        service.evaluate(
            ad_experiment_id="adexp_family_trip_001",
            request=AdExperimentEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []
    assert repos.experiments.status_updates == []


def test_ad_experiment_evaluation_rejects_missing_snapshot_target() -> None:
    repos = FakeEvaluationRepos(
        run=promotion_run_record(goal_snapshot_json={"min_sample_size": 10})
    )
    service = make_service(repos)

    with pytest.raises(AdExperimentEvaluationValidationError, match="goal_target_value"):
        service.evaluate(
            ad_experiment_id="adexp_family_trip_001",
            request=AdExperimentEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []


@pytest.mark.parametrize(
    ("field_name", "mismatched_value"),
    [
        ("content_option_id", "option_b"),
        ("project_id", "other-project"),
        ("campaign_id", "other-campaign"),
        ("promotion_id", "other-promotion"),
        ("segment_id", "other-segment"),
        ("generation_id", "other-generation"),
    ],
)
def test_ad_experiment_evaluation_rejects_candidate_context_mismatch_without_writes(
    field_name: str,
    mismatched_value: str,
) -> None:
    candidate_values = {field_name: mismatched_value}
    repos = FakeEvaluationRepos(
        candidate=content_candidate_record(**candidate_values),
    )
    service = make_service(repos)

    with pytest.raises(
        AdExperimentEvaluationValidationError,
        match=field_name,
    ):
        service.evaluate(
            ad_experiment_id="adexp_family_trip_001",
            request=AdExperimentEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []
    assert repos.experiments.status_updates == []
    assert repos.metrics.cutoffs == []


def test_ad_experiment_evaluation_rejects_missing_candidate_without_writes() -> None:
    repos = FakeEvaluationRepos(candidate=None)
    service = make_service(repos)

    with pytest.raises(AdExperimentEvaluationValidationError, match="not found"):
        service.evaluate(
            ad_experiment_id="adexp_family_trip_001",
            request=AdExperimentEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []
    assert repos.experiments.status_updates == []


def test_strategy_snapshot_distinguishes_missing_keys_from_explicit_empty_values() -> None:
    repos = FakeEvaluationRepos(
        candidate=content_candidate_record(
            metadata_json={
                "strategy_plan": {},
                "evidence_refs": [],
            }
        )
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    snapshot = repos.evaluations.inserted[0].result_json["strategy_snapshot"]
    assert snapshot == {
        "strategy_key": None,
        "strategy_plan": {},
        "evidence_refs": [],
        "brief_fingerprint": None,
        "prompt_builder_version": None,
        "fallback_guidance_used": None,
    }
    assert response.strategy_snapshot.model_dump() == snapshot


def test_strategy_snapshot_deep_copies_nested_metadata() -> None:
    metadata = {
        "strategy_key": "booking_confidence__family",
        "strategy_plan": {"benefit_focus": ["free_cancellation"]},
        "evidence_refs": ["content_brief.benefit_evidence.free_cancellation"],
    }
    repos = FakeEvaluationRepos(
        candidate=content_candidate_record(metadata_json=metadata)
    )
    service = make_service(repos)

    service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )
    metadata["strategy_plan"]["benefit_focus"].append("breakfast")
    metadata["evidence_refs"].append("content_brief.benefit_evidence.breakfast")

    snapshot = repos.evaluations.inserted[0].result_json["strategy_snapshot"]
    assert snapshot["strategy_plan"] == {
        "benefit_focus": ["free_cancellation"]
    }
    assert snapshot["evidence_refs"] == [
        "content_brief.benefit_evidence.free_cancellation"
    ]


def test_ad_experiment_evaluation_maps_missing_experiment() -> None:
    repos = FakeEvaluationRepos(experiment=None)
    service = make_service(repos)

    with pytest.raises(AdExperimentEvaluationNotFoundError):
        service.evaluate(
            ad_experiment_id="missing",
            request=AdExperimentEvaluateRequest(),
        )


def test_ad_experiment_evaluation_uses_fixed_context_and_serializes_utc_ms() -> None:
    repos = FakeEvaluationRepos()
    service = make_service(repos)
    context = EvaluationContext(
        evaluation_cutoff_at=datetime(
            2026,
            7,
            10,
            12,
            34,
            56,
            567890,
            tzinfo=UTC,
        ),
        window_start=datetime(
            2026,
            7,
            9,
            12,
            34,
            56,
            123456,
            tzinfo=UTC,
        ),
        evaluator_version="evaluator-test-v1",
        metric_sql_version="metric-sql-test-v1",
    )

    evaluation = service.evaluate_with_context(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
        context=context,
    )

    assert repos.metrics.cutoffs == [
        datetime(2026, 7, 10, 12, 34, 56, 567000, tzinfo=UTC)
    ]
    assert evaluation.result_json["evaluation_cutoff_at"] == (
        "2026-07-10T12:34:56.567Z"
    )
    assert evaluation.result_json["window_start"] == "2026-07-09T12:34:56.123Z"
    assert evaluation.result_json["evaluator_version"] == "evaluator-test-v1"
    assert evaluation.result_json["metric_sql_version"] == "metric-sql-test-v1"


def test_evaluation_context_rejects_naive_datetime() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        EvaluationContext(evaluation_cutoff_at=datetime(2026, 7, 10))


@pytest.mark.parametrize(
    "counts",
    [
        MetricCountRecord(numerator_count=-1, denominator_count=10),
        MetricCountRecord(numerator_count=1, denominator_count=-1),
        MetricCountRecord(numerator_count=11, denominator_count=10),
    ],
)
def test_ad_experiment_evaluation_rejects_invalid_counts_without_writes(
    counts: MetricCountRecord,
) -> None:
    repos = FakeEvaluationRepos(counts=counts)
    service = make_service(repos)

    with pytest.raises(AdExperimentEvaluationValidationError, match="metric"):
        service.evaluate(
            ad_experiment_id="adexp_family_trip_001",
            request=AdExperimentEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []
    assert repos.experiments.status_updates == []


def make_service(repos: "FakeEvaluationRepos") -> AdExperimentEvaluationService:
    return AdExperimentEvaluationService(
        ad_experiment_repository=repos.experiments,
        promotion_run_repository=repos.runs,
        content_candidate_repository=repos.candidates,
        promotion_evaluation_repository=repos.evaluations,
        evaluation_metric_repository=repos.metrics,
    )


class FakeEvaluationRepos:
    def __init__(
        self,
        *,
        experiment: AdExperimentRecord | None | object = DEFAULT_EXPERIMENT,
        candidate: ContentCandidateRecord | None | object = DEFAULT_CANDIDATE,
        run: PromotionRunRecord | None = None,
        counts: MetricCountRecord | None = None,
    ) -> None:
        self.experiments = FakeAdExperimentRepository(
            ad_experiment_record()
            if experiment is DEFAULT_EXPERIMENT
            else experiment
        )
        self.runs = FakePromotionRunRepository(run or promotion_run_record())
        self.candidates = FakeContentCandidateRepository(
            content_candidate_record()
            if candidate is DEFAULT_CANDIDATE
            else candidate
        )
        self.evaluations = FakePromotionEvaluationRepository()
        self.metrics = FakeEvaluationMetricRepository(
            counts or MetricCountRecord(numerator_count=2, denominator_count=10)
        )


class FakeAdExperimentRepository:
    def __init__(self, experiment: AdExperimentRecord | None) -> None:
        self.experiment = experiment
        self.status_updates: list[tuple[str, str]] = []

    def get_by_id(self, ad_experiment_id: str) -> AdExperimentRecord | None:
        if self.experiment is None:
            return None
        if self.experiment.ad_experiment_id != ad_experiment_id:
            return None
        return self.experiment

    def update_status(self, *, ad_experiment_id: str, status: str) -> None:
        self.status_updates.append((ad_experiment_id, status))


class FakePromotionRunRepository:
    def __init__(self, run: PromotionRunRecord | None) -> None:
        self.run = run

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        if self.run is None:
            return None
        if self.run.promotion_run_id != promotion_run_id:
            return None
        return self.run


class FakeContentCandidateRepository:
    def __init__(self, candidate: ContentCandidateRecord | None) -> None:
        self.candidate = candidate

    def get_by_id(self, content_id: str) -> ContentCandidateRecord | None:
        if self.candidate is None or self.candidate.content_id != content_id:
            return None
        return self.candidate


class FakePromotionEvaluationRepository:
    def __init__(self) -> None:
        self.inserted: list[PromotionEvaluationWrite] = []

    def insert(self, evaluation: PromotionEvaluationWrite) -> None:
        self.inserted.append(evaluation)


class FakeEvaluationMetricRepository:
    def __init__(self, counts: MetricCountRecord) -> None:
        self.counts = counts
        self.cutoffs: list[datetime] = []

    def count_inflow_rate(
        self,
        _experiment: AdExperimentRecord,
        *,
        evaluation_cutoff_at: datetime,
    ) -> MetricCountRecord:
        self.cutoffs.append(evaluation_cutoff_at)
        return self.counts

    def count_booking_conversion_rate(
        self,
        _experiment: AdExperimentRecord,
        *,
        evaluation_cutoff_at: datetime,
    ) -> MetricCountRecord:
        self.cutoffs.append(evaluation_cutoff_at)
        return self.counts


def ad_experiment_record(
    *,
    goal_metric: str = GoalMetric.BOOKING_CONVERSION_RATE.value,
    channel: str = Channel.ONSITE_BANNER.value,
) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id="adexp_family_trip_001",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        segment_id="seg_family_trip",
        segment_name="Family hotel trip",
        content_id="content_family_trip_001",
        content_option_id="option_a",
        channel=channel,
        loop_count=1,
        status=AdExperimentStatus.RUNNING.value,
        goal_metric=goal_metric,
        goal_target_value=Decimal("0.030000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
    )


def promotion_run_record(
    *,
    goal_snapshot_json: dict[str, object] | None = None,
) -> PromotionRunRecord:
    return PromotionRunRecord(
        promotion_run_id="prun_banner_001_loop_1",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        loop_count=1,
        status=PromotionRunStatus.RUNNING.value,
        goal_snapshot_json=goal_snapshot_json
        if goal_snapshot_json is not None
        else {
            "goal_target_value": "0.300000",
            "min_sample_size": 10,
        },
    )


def content_candidate_record(
    *,
    content_option_id: str = "option_a",
    generation_id: str = "generation_banner_001",
    project_id: str = "hotel-client-a",
    campaign_id: str = "camp_summer_2026",
    promotion_id: str = "promo_banner_001",
    segment_id: str = "seg_family_trip",
    metadata_json: dict[str, object] | None = None,
) -> ContentCandidateRecord:
    return ContentCandidateRecord(
        content_id="content_family_trip_001",
        content_option_id=content_option_id,
        generation_id=generation_id,
        analysis_id="analysis_banner_001",
        project_id=project_id,
        campaign_id=campaign_id,
        promotion_id=promotion_id,
        segment_id=segment_id,
        channel=Channel.ONSITE_BANNER.value,
        status="approved",
        metadata_json=metadata_json
        if metadata_json is not None
        else {
            "strategy_key": "booking_confidence__family",
            "strategy_plan": {"benefit_focus": ["free_cancellation"]},
            "evidence_refs": [
                "content_brief.benefit_evidence.free_cancellation"
            ],
            "brief_fingerprint": "sha256:brief",
            "prompt_builder_version": "dec-c2.v4",
            "fallback_guidance_used": False,
        },
    )
