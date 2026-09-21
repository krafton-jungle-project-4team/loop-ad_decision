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
    BookingIntentCohortRecord,
    EvaluationFunnelRecord,
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
    assert repos.experiments.status_updates == []
    assert repos.experiments.experiment is not None
    assert repos.experiments.experiment.status == AdExperimentStatus.RUNNING.value


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
        counts=MetricCountRecord(
            numerator_count=2,
            denominator_count=10,
            funnel=EvaluationFunnelRecord(
                response_count=10,
                hotel_search_count=9,
                hotel_detail_view_count=8,
                booking_start_count=5,
                booking_complete_count=2,
            ),
        ),
    )
    service = make_service(repos)

    response = service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.actual_value == Decimal("0.200000")
    assert response.status == PromotionEvaluationStatus.GOAL_NOT_MET
    assert response.next_loop_required is True
    assert response.feedback is not None
    assert "예약 시작에서 예약 완료" in response.feedback
    assert "10.00%p" in response.feedback
    inserted = repos.evaluations.inserted[0]
    assert inserted.next_loop_required is True
    assert inserted.result_json["event_names"] == {
        "numerator": "booking_complete",
        "denominator": "promotion_click",
    }
    diagnosis = inserted.result_json["diagnosis"]
    assert diagnosis["version"] == "dec.evaluation-diagnosis.v4"
    assert diagnosis["status"] == "goal_not_met"
    assert diagnosis["summary"] == response.feedback
    assert diagnosis["observed_bottleneck"] == (
        "booking_start_to_booking_complete"
    )
    assert diagnosis["largest_dropoff"] == {
        "from_stage_key": "booking_start",
        "from_stage_label": "예약 시작",
        "to_stage_key": "booking_complete",
        "to_stage_label": "예약 완료",
        "from_count": 5,
        "to_count": 2,
        "dropoff_count": 3,
        "dropoff_rate": "0.600000",
    }
    assert [stage["user_count"] for stage in diagnosis["funnel"]["stages"]] == [
        10,
        9,
        8,
        5,
        2,
    ]
    assert diagnosis["evidence_strength"]["level"] == "limited"
    assert diagnosis["data_origin"] == {
        "kind": "observed",
        "label": "수집 이벤트",
    }
    assert "상세 실패 이벤트" in diagnosis["limitations"][-1]
    assert diagnosis["audience_intent_analysis"] is None
    assert repos.experiments.status_updates == []
    assert repos.experiments.experiment is not None
    assert repos.experiments.experiment.status == AdExperimentStatus.RUNNING.value


def test_goal_not_met_includes_booking_intent_cohort_analysis() -> None:
    repos = FakeEvaluationRepos(
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_target_value": "0.100000",
                "min_sample_size": 10,
                "outcome_spec": {
                    "outcome_filter": {
                        "destination_ids": ["okinawa", "jeju"],
                    }
                },
                "audience_context": {"age_groups": ["20대", "30대"]},
            }
        ),
        counts=MetricCountRecord(
            numerator_count=24,
            denominator_count=480,
            funnel=EvaluationFunnelRecord(
                response_count=480,
                hotel_search_count=216,
                hotel_detail_view_count=144,
                booking_start_count=64,
                booking_complete_count=24,
            ),
        ),
        cohorts=BookingIntentCohortRecord(
            ad_click_count=216,
            repeat_view_user_count=130,
            repeat_view_booking_count=16,
            comparison_user_count=350,
            comparison_booking_count=9,
            booking_abandon_user_count=40,
            booking_complete_user_count=24,
            booking_abandon_median_revenue=Decimal("720000"),
            booking_complete_median_revenue=Decimal("510000"),
            high_price_booking_start_user_count=103,
            high_price_booking_abandon_user_count=82,
            high_price_booking_complete_user_count=21,
            booking_abandon_median_nightly_price=Decimal("238000"),
            booking_complete_median_nightly_price=Decimal("184000"),
        ),
    )

    make_service(repos).evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    diagnosis = repos.evaluations.inserted[0].result_json["diagnosis"]
    assert diagnosis["audience_intent_analysis"] is None
    analysis = diagnosis["price_abandonment_analysis"]
    assert analysis["title"] == (
        "높은 1박 가격이 예약 완료에 부담이 되었을 가능성이 있습니다"
    )
    assert analysis["price_abandonment"] == {
        "currency": "KRW",
        "nightly_price_threshold": "200000",
        "booking_start_user_count": 103,
        "booking_abandon_user_count": 82,
        "booking_complete_user_count": 21,
        "booking_abandon_median_nightly_price": "238000",
        "booking_complete_median_nightly_price": "184000",
    }
    assert analysis["next_segment_hypothesis"]["condition_labels"] == [
        "20~30대",
        "최근 7일 제주·오키나와 숙소 1박 가격 20만 원 초과",
        "예약 시작 후 미완료",
    ]
    assert any("가능성이 있습니다" in item for item in analysis["paragraphs"])
    assert repos.metrics.cohort_destination_ids == [("jeju", "okinawa")]


def test_booking_intent_cohort_failure_does_not_block_evaluation() -> None:
    repos = FakeEvaluationRepos(
        counts=MetricCountRecord(
            numerator_count=2,
            denominator_count=10,
            funnel=EvaluationFunnelRecord(10, 9, 8, 5, 2),
        ),
        cohort_error=RuntimeError("supplemental query failed"),
    )

    response = make_service(repos).evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    assert response.status == PromotionEvaluationStatus.GOAL_NOT_MET
    assert len(repos.evaluations.inserted) == 1
    assert (
        repos.evaluations.inserted[0].result_json["diagnosis"][
            "audience_intent_analysis"
        ]
        is None
    )


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


def test_ad_experiment_evaluation_marks_fixture_backed_funnel() -> None:
    repos = FakeEvaluationRepos(
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_target_value": "0.300000",
                "min_sample_size": 1,
            }
        ),
        counts=MetricCountRecord(
            numerator_count=20,
            denominator_count=100,
            funnel=EvaluationFunnelRecord(
                response_count=100,
                hotel_search_count=90,
                hotel_detail_view_count=80,
                booking_start_count=70,
                booking_complete_count=20,
                fixture_response_count=100,
            ),
        ),
    )

    service = make_service(repos)
    service.evaluate(
        ad_experiment_id="adexp_family_trip_001",
        request=AdExperimentEvaluateRequest(),
    )

    diagnosis = repos.evaluations.inserted[0].result_json["diagnosis"]
    assert diagnosis["data_origin"] == {
        "kind": "demo_fixture",
        "label": "시연 데이터",
    }
    assert diagnosis["evidence_strength"]["level"] == "sufficient"


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
    assert repos.experiments.status_updates == []
    assert repos.experiments.experiment is not None
    assert repos.experiments.experiment.status == AdExperimentStatus.RUNNING.value


def test_ad_experiment_evaluation_uses_event_denominator_for_min_sample() -> None:
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
    assert response.denominator_count == 20
    assert response.sample_size == 20
    assert response.next_loop_required is False
    assert repos.evaluations.inserted[0].result_json["status_reason"] == (
        "sample_size_below_minimum"
    )
    assert repos.experiments.status_updates == []
    assert repos.experiments.experiment is not None
    assert repos.experiments.experiment.status == AdExperimentStatus.RUNNING.value


def test_insufficient_data_evaluation_can_be_repeated_without_stopping_experiment() -> None:
    repos = FakeEvaluationRepos(
        counts=MetricCountRecord(numerator_count=0, denominator_count=0),
    )
    service = make_service(repos)

    responses = [
        service.evaluate(
            ad_experiment_id="adexp_family_trip_001",
            request=AdExperimentEvaluateRequest(),
        )
        for _ in range(2)
    ]

    assert [response.status for response in responses] == [
        PromotionEvaluationStatus.INSUFFICIENT_DATA,
        PromotionEvaluationStatus.INSUFFICIENT_DATA,
    ]
    assert len(repos.evaluations.inserted) == 2
    assert all(
        evaluation.status == PromotionEvaluationStatus.INSUFFICIENT_DATA.value
        for evaluation in repos.evaluations.inserted
    )
    assert repos.experiments.status_updates == []
    assert repos.experiments.experiment is not None
    assert repos.experiments.experiment.status == AdExperimentStatus.RUNNING.value


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
        promotion_evaluation_repository=repos.evaluations,
        evaluation_metric_repository=repos.metrics,
    )


class FakeEvaluationRepos:
    def __init__(
        self,
        *,
        experiment: AdExperimentRecord | None | object = DEFAULT_EXPERIMENT,
        run: PromotionRunRecord | None = None,
        counts: MetricCountRecord | None = None,
        cohorts: BookingIntentCohortRecord | None = None,
        cohort_error: Exception | None = None,
    ) -> None:
        self.experiments = FakeAdExperimentRepository(
            ad_experiment_record()
            if experiment is DEFAULT_EXPERIMENT
            else experiment
        )
        self.runs = FakePromotionRunRepository(run or promotion_run_record())
        self.evaluations = FakePromotionEvaluationRepository()
        self.metrics = FakeEvaluationMetricRepository(
            counts or MetricCountRecord(numerator_count=2, denominator_count=10),
            cohorts=cohorts,
            cohort_error=cohort_error,
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


class FakePromotionEvaluationRepository:
    def __init__(self) -> None:
        self.inserted: list[PromotionEvaluationWrite] = []

    def insert(self, evaluation: PromotionEvaluationWrite) -> None:
        self.inserted.append(evaluation)


class FakeEvaluationMetricRepository:
    def __init__(
        self,
        counts: MetricCountRecord,
        *,
        cohorts: BookingIntentCohortRecord | None,
        cohort_error: Exception | None,
    ) -> None:
        self.counts = counts
        self.cohorts = cohorts or BookingIntentCohortRecord(
            0, 0, 0, 0, 0, 0, 0, None, None
        )
        self.cohort_error = cohort_error
        self.cutoffs: list[datetime] = []
        self.cohort_destination_ids: list[tuple[str, ...]] = []

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

    def analyze_booking_intent_cohorts(
        self,
        _experiment: AdExperimentRecord,
        *,
        destination_ids: tuple[str, ...],
        evaluation_cutoff_at: datetime,
        lookback_days: int,
    ) -> BookingIntentCohortRecord:
        del evaluation_cutoff_at, lookback_days
        if self.cohort_error is not None:
            raise self.cohort_error
        self.cohort_destination_ids.append(tuple(destination_ids))
        return self.cohorts


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
        segment_scope_json=["seg_family_trip"],
        segment_scope_fingerprint="a" * 64,
    )
