from __future__ import annotations

from datetime import datetime
from decimal import Decimal
from typing import Sequence

import pytest

from app.decision.evaluation_service import (
    EvaluationContext,
    PromotionRunEvaluationNotFoundError,
    PromotionRunEvaluationService,
    PromotionRunEvaluationValidationError,
)
from app.decision.repositories import (
    AdExperimentRecord,
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
    PromotionRunEvaluateRequest,
    PromotionRunStatus,
)


def test_promotion_run_evaluation_aggregates_all_segments_mixed_status() -> None:
    repos = FakeRunEvaluationRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_MET.value,
                actual_value=Decimal("0.400000"),
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
                actual_value=Decimal("0.100000"),
            ),
        ],
    )
    service = make_service(repos)

    response = service.evaluate(
        promotion_run_id="prun_banner_001_loop_1",
        request=PromotionRunEvaluateRequest(),
    )

    assert response.status == PromotionRunStatus.PARTIAL_GOAL_MET
    assert response.next_loop_required is True
    assert response.target_gap is None
    assert response.failed_segment_ids == ["seg_luxury"]
    assert response.failed_ad_experiment_ids == ["adexp_luxury_001"]
    assert [item.status for item in response.ad_experiment_results] == [
        PromotionEvaluationStatus.GOAL_MET,
        PromotionEvaluationStatus.GOAL_NOT_MET,
    ]
    inserted = repos.evaluations.inserted[0]
    assert inserted.metric == GoalMetric.BOOKING_CONVERSION_RATE.value
    assert inserted.target_value == Decimal("0.300000")
    assert inserted.basis == GoalBasis.ALL_SEGMENTS.value
    assert inserted.ad_experiment_id is None
    assert inserted.segment_id is None
    assert inserted.actual_value == Decimal("0.000000")
    assert inserted.numerator_count == 0
    assert inserted.denominator_count == 0
    assert inserted.sample_size == 0
    assert inserted.result_json["failed_segment_ids"] == ["seg_luxury"]
    assert inserted.result_json["evaluation_scope"] == "promotion_run_aggregate"
    assert inserted.result_json["target_gap"] is None
    assert "strategy_snapshot" not in inserted.result_json
    assert inserted.result_json["ad_experiment_results"][0]["target_gap"] == (
        "0.100000"
    )
    assert inserted.result_json["ad_experiment_results"][0][
        "strategy_snapshot"
    ]["strategy_key"] is None
    assert repos.runs.status_updates == [
        ("prun_banner_001_loop_1", PromotionRunStatus.PARTIAL_GOAL_MET.value)
    ]


def test_promotion_run_evaluation_reevaluates_every_experiment_without_latest_reads() -> None:
    repos = FakeRunEvaluationRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_MET.value,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_MET.value,
            ),
        ],
    )
    service = make_service(repos)

    response = service.evaluate(
        promotion_run_id="prun_banner_001_loop_1",
        request=PromotionRunEvaluateRequest(),
    )

    assert repos.ad_evaluation_service.calls == [
        "adexp_family_trip_001",
        "adexp_luxury_001",
    ]
    assert len(set(repos.ad_evaluation_service.cutoffs)) == 1
    assert repos.evaluations.latest_calls == []
    assert response.status == PromotionRunStatus.GOAL_MET
    assert repos.evaluations.inserted[0].status == PromotionRunStatus.GOAL_MET.value


def test_promotion_run_evaluation_marks_only_insufficient_all_segments() -> None:
    repos = FakeRunEvaluationRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.INSUFFICIENT_DATA.value,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.INSUFFICIENT_DATA.value,
            ),
        ],
    )
    service = make_service(repos)

    response = service.evaluate(
        promotion_run_id="prun_banner_001_loop_1",
        request=PromotionRunEvaluateRequest(),
    )

    assert response.status == PromotionRunStatus.INSUFFICIENT_DATA
    assert response.next_loop_required is False
    assert response.failed_segment_ids == []
    assert response.failed_ad_experiment_ids == []


def test_promotion_run_evaluation_sums_promotion_average_counts() -> None:
    repos = FakeRunEvaluationRepos(
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
                "goal_target_value": "0.300000",
                "goal_basis": GoalBasis.PROMOTION_AVERAGE.value,
                "min_sample_size": 10,
            }
        ),
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_MET.value,
                numerator_count=6,
                denominator_count=10,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_NOT_MET.value,
                numerator_count=1,
                denominator_count=10,
            ),
        ],
    )
    service = make_service(repos)

    response = service.evaluate(
        promotion_run_id="prun_banner_001_loop_1",
        request=PromotionRunEvaluateRequest(),
    )

    assert response.status == PromotionRunStatus.GOAL_MET
    assert response.failed_segment_ids == ["seg_luxury"]
    inserted = repos.evaluations.inserted[0]
    assert inserted.basis == GoalBasis.PROMOTION_AVERAGE.value
    assert inserted.actual_value == Decimal("0.350000")
    assert inserted.numerator_count == 7
    assert inserted.denominator_count == 20
    assert inserted.sample_size == 20
    assert inserted.result_json["target_gap"] == "0.050000"
    assert response.target_gap == Decimal("0.050000")


def test_promotion_run_evaluation_rejects_promotion_average_min_sample() -> None:
    repos = FakeRunEvaluationRepos(
        run=promotion_run_record(
            goal_snapshot_json={
                "goal_metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
                "goal_target_value": "0.100000",
                "goal_basis": GoalBasis.PROMOTION_AVERAGE.value,
                "min_sample_size": 50,
            }
        ),
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status=PromotionEvaluationStatus.GOAL_MET.value,
                numerator_count=10,
                denominator_count=20,
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_MET.value,
                numerator_count=10,
                denominator_count=20,
            ),
        ],
    )
    service = make_service(repos)

    response = service.evaluate(
        promotion_run_id="prun_banner_001_loop_1",
        request=PromotionRunEvaluateRequest(),
    )

    assert response.status == PromotionRunStatus.INSUFFICIENT_DATA
    assert response.next_loop_required is False
    assert repos.evaluations.inserted[0].result_json["status_reason"] == (
        "sample_size_below_minimum"
    )


def test_promotion_run_evaluation_rejects_missing_run() -> None:
    repos = FakeRunEvaluationRepos(run=None)
    service = make_service(repos)

    with pytest.raises(PromotionRunEvaluationNotFoundError):
        service.evaluate(
            promotion_run_id="missing",
            request=PromotionRunEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []


def test_promotion_run_evaluation_rejects_run_without_ad_experiments() -> None:
    repos = FakeRunEvaluationRepos(experiments=[])
    service = make_service(repos)

    with pytest.raises(PromotionRunEvaluationValidationError, match="ad experiments"):
        service.evaluate(
            promotion_run_id="prun_banner_001_loop_1",
            request=PromotionRunEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []


def test_promotion_run_evaluation_rejects_goal_near_individual_status() -> None:
    repos = FakeRunEvaluationRepos(
        evaluations=[
            evaluation_record(
                ad_experiment_id="adexp_family_trip_001",
                segment_id="seg_family_trip",
                status="goal_near",
            ),
            evaluation_record(
                ad_experiment_id="adexp_luxury_001",
                segment_id="seg_luxury",
                status=PromotionEvaluationStatus.GOAL_MET.value,
            ),
        ],
    )
    service = make_service(repos)

    with pytest.raises(PromotionRunEvaluationValidationError, match="MVP scope"):
        service.evaluate(
            promotion_run_id="prun_banner_001_loop_1",
            request=PromotionRunEvaluateRequest(),
        )

    assert repos.evaluations.inserted == []


def make_service(repos: "FakeRunEvaluationRepos") -> PromotionRunEvaluationService:
    return PromotionRunEvaluationService(
        promotion_run_repository=repos.runs,
        ad_experiment_repository=repos.experiments,
        promotion_evaluation_repository=repos.evaluations,
        ad_experiment_evaluation_service=repos.ad_evaluation_service,
    )


class FakeRunEvaluationRepos:
    def __init__(
        self,
        *,
        run: PromotionRunRecord | None | object = object(),
        experiments: Sequence[AdExperimentRecord] | None = None,
        evaluations: list[PromotionEvaluationWrite] | None = None,
    ) -> None:
        self.runs = FakePromotionRunRepository(
            promotion_run_record() if not isinstance(run, PromotionRunRecord) and run is not None else run
        )
        self.experiments = FakeAdExperimentRepository(
            list(experiments) if experiments is not None else default_experiments()
        )
        self.evaluations = FakePromotionEvaluationRepository(evaluations or [])
        self.ad_evaluation_service = FakeAdExperimentEvaluationService(
            self.evaluations
        )


class FakePromotionRunRepository:
    def __init__(self, run: PromotionRunRecord | None) -> None:
        self.run = run
        self.status_updates: list[tuple[str, str]] = []

    def get_by_id(self, promotion_run_id: str) -> PromotionRunRecord | None:
        if self.run is None:
            return None
        if self.run.promotion_run_id != promotion_run_id:
            return None
        return self.run

    def update_status(self, *, promotion_run_id: str, status: str) -> None:
        self.status_updates.append((promotion_run_id, status))


class FakeAdExperimentRepository:
    def __init__(self, experiments: list[AdExperimentRecord]) -> None:
        self.experiments = experiments

    def list_by_run(self, promotion_run_id: str) -> list[AdExperimentRecord]:
        return [
            experiment
            for experiment in self.experiments
            if experiment.promotion_run_id == promotion_run_id
        ]


class FakePromotionEvaluationRepository:
    def __init__(self, evaluations: list[PromotionEvaluationWrite]) -> None:
        self.evaluations = evaluations
        self.inserted: list[PromotionEvaluationWrite] = []
        self.latest_calls: list[str] = []

    def insert(self, evaluation: PromotionEvaluationWrite) -> None:
        self.inserted.append(evaluation)

    def list_latest_by_run_ad_experiments(
        self,
        promotion_run_id: str,
    ) -> list[PromotionEvaluationWrite]:
        self.latest_calls.append(promotion_run_id)
        return [
            evaluation
            for evaluation in self.evaluations
            if evaluation.promotion_run_id == promotion_run_id
            and evaluation.ad_experiment_id is not None
        ]


class FakeAdExperimentEvaluationService:
    def __init__(self, evaluations: FakePromotionEvaluationRepository) -> None:
        self._evaluations = evaluations
        self.calls: list[str] = []
        self.cutoffs: list[datetime] = []
        self.generated: dict[str, PromotionEvaluationWrite] = {
            str(evaluation.ad_experiment_id): evaluation
            for evaluation in evaluations.evaluations
            if evaluation.ad_experiment_id is not None
        }

    def evaluate_with_context(
        self,
        *,
        ad_experiment_id: str,
        request: AdExperimentEvaluateRequest,
        context: EvaluationContext,
    ) -> PromotionEvaluationWrite:
        _ = request
        self.calls.append(ad_experiment_id)
        self.cutoffs.append(context.evaluation_cutoff_at)
        return self.generated[ad_experiment_id]


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
        or {
            "goal_metric": GoalMetric.BOOKING_CONVERSION_RATE.value,
            "goal_target_value": "0.300000",
            "goal_basis": GoalBasis.ALL_SEGMENTS.value,
            "min_sample_size": 10,
        },
    )


def default_experiments() -> list[AdExperimentRecord]:
    return [
        ad_experiment_record(
            ad_experiment_id="adexp_family_trip_001",
            segment_id="seg_family_trip",
        ),
        ad_experiment_record(
            ad_experiment_id="adexp_luxury_001",
            segment_id="seg_luxury",
        ),
    ]


def ad_experiment_record(
    *,
    ad_experiment_id: str,
    segment_id: str,
) -> AdExperimentRecord:
    return AdExperimentRecord(
        ad_experiment_id=ad_experiment_id,
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        analysis_id="analysis_banner_001",
        generation_id="generation_banner_001",
        segment_id=segment_id,
        segment_name=segment_id,
        content_id=f"content_{segment_id}",
        content_option_id="option_a",
        channel=Channel.ONSITE_BANNER.value,
        loop_count=1,
        status=AdExperimentStatus.RUNNING.value,
        goal_metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        goal_target_value=Decimal("0.300000"),
        goal_basis=GoalBasis.ALL_SEGMENTS.value,
    )


def evaluation_record(
    *,
    ad_experiment_id: str,
    segment_id: str,
    status: str,
    actual_value: Decimal = Decimal("0.200000"),
    numerator_count: int = 2,
    denominator_count: int = 10,
) -> PromotionEvaluationWrite:
    return PromotionEvaluationWrite(
        evaluation_id=f"eval_{ad_experiment_id}",
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        promotion_run_id="prun_banner_001_loop_1",
        ad_experiment_id=ad_experiment_id,
        segment_id=segment_id,
        content_id=f"content_{segment_id}",
        content_option_id="option_a",
        metric=GoalMetric.BOOKING_CONVERSION_RATE.value,
        target_value=Decimal("0.300000"),
        actual_value=actual_value,
        numerator_count=numerator_count,
        denominator_count=denominator_count,
        sample_size=denominator_count,
        basis=GoalBasis.ALL_SEGMENTS.value,
        status=status,
        feedback=None,
        next_loop_required=status == PromotionEvaluationStatus.GOAL_NOT_MET.value,
        result_json={
            "status": status,
            "status_reason": "target_met"
            if status == PromotionEvaluationStatus.GOAL_MET.value
            else "target_not_met",
            "event_names": {
                "numerator": "booking_complete",
                "denominator": "promotion_click",
            },
            "target_gap": str(
                (actual_value - Decimal("0.300000")).quantize(
                    Decimal("0.000001")
                )
            ),
            "strategy_snapshot": {
                "strategy_key": None,
                "strategy_plan": None,
                "evidence_refs": None,
                "brief_fingerprint": None,
                "prompt_builder_version": None,
                "fallback_guidance_used": None,
            },
        },
    )
