from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWriter,
    EvaluationMetricReader,
    MetricCountRecord,
    PromotionEvaluationRecord,
    PromotionEvaluationWrite,
    PromotionEvaluationWriter,
    PromotionRunRecord,
    PromotionRunWriter,
)
from app.decision.schemas import (
    AdExperimentEvaluateRequest,
    AdExperimentEvaluateResponse,
    AdExperimentStatus,
    GoalBasis,
    GoalMetric,
    PromotionRunAdExperimentResult,
    PromotionRunEvaluateRequest,
    PromotionRunEvaluateResponse,
    PromotionRunStatus,
    PromotionEvaluationStatus,
)
from app.decision.service import build_bounded_decision_id


DECIMAL_SCALE = Decimal("0.000001")


@dataclass(frozen=True)
class _RunAggregate:
    status: str
    actual_value: Decimal
    numerator_count: int
    denominator_count: int
    sample_size: int
    next_loop_required: bool
    failed_segment_ids: list[str]
    failed_ad_experiment_ids: list[str]
    result_json: dict[str, Any]


class AdExperimentEvaluationNotFoundError(Exception):
    pass


class AdExperimentEvaluationValidationError(Exception):
    pass


class PromotionRunEvaluationNotFoundError(Exception):
    pass


class PromotionRunEvaluationValidationError(Exception):
    pass


class AdExperimentEvaluationService:
    def __init__(
        self,
        *,
        ad_experiment_repository: AdExperimentWriter,
        promotion_run_repository: PromotionRunWriter,
        promotion_evaluation_repository: PromotionEvaluationWriter,
        evaluation_metric_repository: EvaluationMetricReader,
    ) -> None:
        self._ad_experiment_repository = ad_experiment_repository
        self._promotion_run_repository = promotion_run_repository
        self._promotion_evaluation_repository = promotion_evaluation_repository
        self._evaluation_metric_repository = evaluation_metric_repository

    def evaluate(
        self,
        *,
        ad_experiment_id: str,
        request: AdExperimentEvaluateRequest,
    ) -> AdExperimentEvaluateResponse:
        _ = request
        experiment = self._ad_experiment_repository.get_by_id(ad_experiment_id)
        if experiment is None:
            raise AdExperimentEvaluationNotFoundError(
                f"ad experiment not found: {ad_experiment_id}"
            )

        run = self._promotion_run_repository.get_by_id(experiment.promotion_run_id)
        if run is None:
            raise AdExperimentEvaluationValidationError(
                f"promotion run not found: {experiment.promotion_run_id}"
            )

        metric = experiment.goal_metric
        if metric == GoalMetric.FUNNEL_STEP_RATE.value:
            raise AdExperimentEvaluationValidationError(
                "funnel_step_rate is out of Owner 2 MVP evaluation scope"
            )

        target_value = _parse_target_value(run.goal_snapshot_json)
        min_sample_size = _parse_min_sample_size(run.goal_snapshot_json)
        counts = self._load_counts(experiment)
        sample_size = counts.denominator_count
        actual_value = _calculate_actual_value(counts)
        status = _decide_status(
            actual_value=actual_value,
            target_value=target_value,
            denominator_count=counts.denominator_count,
            sample_size=sample_size,
            min_sample_size=min_sample_size,
        )
        next_loop_required = status == PromotionEvaluationStatus.GOAL_NOT_MET.value
        evaluation_id = build_bounded_decision_id(
            "eval",
            experiment.ad_experiment_id,
            metric,
            datetime.now(UTC).isoformat(timespec="microseconds"),
        )
        evaluation = PromotionEvaluationWrite(
            evaluation_id=evaluation_id,
            project_id=experiment.project_id,
            campaign_id=experiment.campaign_id,
            promotion_id=experiment.promotion_id,
            promotion_run_id=experiment.promotion_run_id,
            ad_experiment_id=experiment.ad_experiment_id,
            segment_id=experiment.segment_id,
            content_id=experiment.content_id,
            content_option_id=experiment.content_option_id,
            metric=metric,
            target_value=target_value,
            actual_value=actual_value,
            numerator_count=counts.numerator_count,
            denominator_count=counts.denominator_count,
            sample_size=sample_size,
            basis=GoalBasis.ALL_SEGMENTS.value,
            status=status,
            feedback=None,
            next_loop_required=next_loop_required,
            result_json=_build_result_json(
                experiment=experiment,
                counts=counts,
                target_value=target_value,
                actual_value=actual_value,
                min_sample_size=min_sample_size,
                status=status,
            ),
        )
        self._promotion_evaluation_repository.insert(evaluation)
        self._ad_experiment_repository.update_status(
            ad_experiment_id=experiment.ad_experiment_id,
            status=status,
        )

        return AdExperimentEvaluateResponse(
            evaluation_id=evaluation.evaluation_id,
            ad_experiment_id=experiment.ad_experiment_id,
            promotion_run_id=experiment.promotion_run_id,
            promotion_id=experiment.promotion_id,
            segment_id=experiment.segment_id,
            metric=GoalMetric(metric),
            target_value=evaluation.target_value,
            actual_value=evaluation.actual_value,
            numerator_count=evaluation.numerator_count,
            denominator_count=evaluation.denominator_count,
            sample_size=evaluation.sample_size,
            basis=GoalBasis.ALL_SEGMENTS,
            status=PromotionEvaluationStatus(status),
            next_loop_required=evaluation.next_loop_required,
            feedback=evaluation.feedback,
        )

    def _load_counts(self, experiment: AdExperimentRecord) -> MetricCountRecord:
        if experiment.goal_metric == GoalMetric.INFLOW_RATE.value:
            return self._evaluation_metric_repository.count_inflow_rate(experiment)
        if experiment.goal_metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
            return self._evaluation_metric_repository.count_booking_conversion_rate(
                experiment
            )
        raise AdExperimentEvaluationValidationError(
            f"unsupported goal metric: {experiment.goal_metric}"
        )


class PromotionRunEvaluationService:
    def __init__(
        self,
        *,
        promotion_run_repository: PromotionRunWriter,
        ad_experiment_repository: AdExperimentWriter,
        promotion_evaluation_repository: PromotionEvaluationWriter,
        ad_experiment_evaluation_service: AdExperimentEvaluationService,
    ) -> None:
        self._promotion_run_repository = promotion_run_repository
        self._ad_experiment_repository = ad_experiment_repository
        self._promotion_evaluation_repository = promotion_evaluation_repository
        self._ad_experiment_evaluation_service = ad_experiment_evaluation_service

    def evaluate(
        self,
        *,
        promotion_run_id: str,
        request: PromotionRunEvaluateRequest,
    ) -> PromotionRunEvaluateResponse:
        _ = request
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            raise PromotionRunEvaluationNotFoundError(
                f"promotion run not found: {promotion_run_id}"
            )

        experiments = self._ad_experiment_repository.list_by_run(promotion_run_id)
        if not experiments:
            raise PromotionRunEvaluationValidationError(
                f"ad experiments not found for promotion run: {promotion_run_id}"
            )

        metric = _parse_goal_metric(run.goal_snapshot_json)
        if metric == GoalMetric.FUNNEL_STEP_RATE.value:
            raise PromotionRunEvaluationValidationError(
                "funnel_step_rate is out of Owner 2 MVP evaluation scope"
            )
        basis = _parse_goal_basis(run.goal_snapshot_json)
        target_value = _parse_run_target_value(run.goal_snapshot_json)
        min_sample_size = _parse_run_min_sample_size(run.goal_snapshot_json)

        latest_by_experiment = self._latest_by_experiment(promotion_run_id)
        missing_experiments = [
            experiment
            for experiment in experiments
            if experiment.ad_experiment_id not in latest_by_experiment
        ]
        for experiment in missing_experiments:
            self._ad_experiment_evaluation_service.evaluate(
                ad_experiment_id=experiment.ad_experiment_id,
                request=AdExperimentEvaluateRequest(),
            )

        if missing_experiments:
            latest_by_experiment = self._latest_by_experiment(promotion_run_id)

        latest_evaluations = []
        for experiment in experiments:
            evaluation = latest_by_experiment.get(experiment.ad_experiment_id)
            if evaluation is None:
                raise PromotionRunEvaluationValidationError(
                    "latest ad experiment evaluation is required before aggregate"
                )
            _validate_individual_evaluation(evaluation, metric)
            latest_evaluations.append(evaluation)

        aggregate = _aggregate_run_evaluations(
            run=run,
            evaluations=latest_evaluations,
            metric=metric,
            basis=basis,
            target_value=target_value,
            min_sample_size=min_sample_size,
        )

        evaluation_id = build_bounded_decision_id(
            "eval",
            run.promotion_run_id,
            "aggregate",
            metric,
            datetime.now(UTC).isoformat(timespec="microseconds"),
        )
        evaluation = PromotionEvaluationWrite(
            evaluation_id=evaluation_id,
            project_id=run.project_id,
            campaign_id=run.campaign_id,
            promotion_id=run.promotion_id,
            promotion_run_id=run.promotion_run_id,
            ad_experiment_id=None,
            segment_id=None,
            content_id=None,
            content_option_id=None,
            metric=metric,
            target_value=target_value,
            actual_value=aggregate.actual_value,
            numerator_count=aggregate.numerator_count,
            denominator_count=aggregate.denominator_count,
            sample_size=aggregate.sample_size,
            basis=basis,
            status=aggregate.status,
            feedback=None,
            next_loop_required=aggregate.next_loop_required,
            result_json=aggregate.result_json,
        )
        self._promotion_evaluation_repository.insert(evaluation)
        self._promotion_run_repository.update_status(
            promotion_run_id=run.promotion_run_id,
            status=aggregate.status,
        )

        return PromotionRunEvaluateResponse(
            promotion_run_id=run.promotion_run_id,
            promotion_id=run.promotion_id,
            status=PromotionRunStatus(aggregate.status),
            ad_experiment_results=[
                PromotionRunAdExperimentResult(
                    ad_experiment_id=item["ad_experiment_id"],
                    segment_id=item["segment_id"],
                    actual_value=Decimal(item["actual_value"]),
                    status=PromotionEvaluationStatus(item["status"]),
                )
                for item in aggregate.result_json["ad_experiment_results"]
            ],
            next_loop_required=aggregate.next_loop_required,
            failed_segment_ids=aggregate.failed_segment_ids,
            failed_ad_experiment_ids=aggregate.failed_ad_experiment_ids,
        )

    def _latest_by_experiment(
        self,
        promotion_run_id: str,
    ) -> dict[str, PromotionEvaluationRecord]:
        return {
            str(evaluation.ad_experiment_id): evaluation
            for evaluation in self._promotion_evaluation_repository.list_latest_by_run_ad_experiments(
                promotion_run_id
            )
            if evaluation.ad_experiment_id is not None
        }


def _parse_target_value(snapshot: Mapping[str, Any]) -> Decimal:
    if "goal_target_value" not in snapshot:
        raise AdExperimentEvaluationValidationError(
            "goal_snapshot_json.goal_target_value is required"
        )
    try:
        value = Decimal(str(snapshot["goal_target_value"]))
    except Exception as exc:
        raise AdExperimentEvaluationValidationError(
            "goal_snapshot_json.goal_target_value must be decimal"
        ) from exc
    if value < 0:
        raise AdExperimentEvaluationValidationError(
            "goal_snapshot_json.goal_target_value must not be negative"
        )
    return value.quantize(DECIMAL_SCALE, rounding=ROUND_HALF_UP)


def _parse_min_sample_size(snapshot: Mapping[str, Any]) -> int:
    if "min_sample_size" not in snapshot:
        raise AdExperimentEvaluationValidationError(
            "goal_snapshot_json.min_sample_size is required"
        )
    value = snapshot["min_sample_size"]
    if isinstance(value, bool):
        raise AdExperimentEvaluationValidationError(
            "goal_snapshot_json.min_sample_size must be an integer"
        )
    try:
        min_sample_size = int(value)
    except (TypeError, ValueError) as exc:
        raise AdExperimentEvaluationValidationError(
            "goal_snapshot_json.min_sample_size must be an integer"
        ) from exc
    if min_sample_size < 0:
        raise AdExperimentEvaluationValidationError(
            "goal_snapshot_json.min_sample_size must not be negative"
        )
    return min_sample_size


def _parse_run_target_value(snapshot: Mapping[str, Any]) -> Decimal:
    try:
        return _parse_target_value(snapshot)
    except AdExperimentEvaluationValidationError as exc:
        raise PromotionRunEvaluationValidationError(str(exc)) from exc


def _parse_run_min_sample_size(snapshot: Mapping[str, Any]) -> int:
    try:
        return _parse_min_sample_size(snapshot)
    except AdExperimentEvaluationValidationError as exc:
        raise PromotionRunEvaluationValidationError(str(exc)) from exc


def _parse_goal_metric(snapshot: Mapping[str, Any]) -> str:
    if "goal_metric" not in snapshot:
        raise PromotionRunEvaluationValidationError(
            "goal_snapshot_json.goal_metric is required"
        )
    value = str(snapshot["goal_metric"])
    try:
        return GoalMetric(value).value
    except ValueError as exc:
        raise PromotionRunEvaluationValidationError(
            "goal_snapshot_json.goal_metric is invalid"
        ) from exc


def _parse_goal_basis(snapshot: Mapping[str, Any]) -> str:
    if "goal_basis" not in snapshot:
        raise PromotionRunEvaluationValidationError(
            "goal_snapshot_json.goal_basis is required"
        )
    value = str(snapshot["goal_basis"])
    try:
        return GoalBasis(value).value
    except ValueError as exc:
        raise PromotionRunEvaluationValidationError(
            "goal_snapshot_json.goal_basis is invalid"
        ) from exc


def _aggregate_run_evaluations(
    *,
    run: PromotionRunRecord,
    evaluations: list[PromotionEvaluationRecord],
    metric: str,
    basis: str,
    target_value: Decimal,
    min_sample_size: int,
) -> _RunAggregate:
    failed_evaluations = [
        evaluation
        for evaluation in evaluations
        if evaluation.status == PromotionEvaluationStatus.GOAL_NOT_MET.value
    ]
    failed_segment_ids = [
        str(evaluation.segment_id)
        for evaluation in failed_evaluations
        if evaluation.segment_id is not None
    ]
    failed_ad_experiment_ids = [
        str(evaluation.ad_experiment_id)
        for evaluation in failed_evaluations
        if evaluation.ad_experiment_id is not None
    ]
    next_loop_required = bool(failed_ad_experiment_ids)
    experiment_results = [
        _evaluation_summary(evaluation)
        for evaluation in evaluations
    ]

    if basis == GoalBasis.ALL_SEGMENTS.value:
        status = _aggregate_all_segments_status(evaluations)
        return _RunAggregate(
            status=status,
            actual_value=Decimal("0.000000"),
            numerator_count=0,
            denominator_count=0,
            sample_size=0,
            next_loop_required=next_loop_required,
            failed_segment_ids=failed_segment_ids,
            failed_ad_experiment_ids=failed_ad_experiment_ids,
            result_json={
                "basis": basis,
                "metric": metric,
                "target_value": str(target_value),
                "status_reason": _aggregate_status_reason(status, basis),
                "ad_experiment_results": experiment_results,
                "failed_segment_ids": failed_segment_ids,
                "failed_ad_experiment_ids": failed_ad_experiment_ids,
                "min_sample_size": min_sample_size,
                "numeric_fields": {
                    "actual_value": "0.000000",
                    "numerator_count": 0,
                    "denominator_count": 0,
                    "sample_size": 0,
                    "reason": "all_segments uses status aggregation; per-experiment values are stored in ad_experiment_results",
                },
            },
        )

    numerator_count = sum(evaluation.numerator_count for evaluation in evaluations)
    denominator_count = sum(evaluation.denominator_count for evaluation in evaluations)
    sample_size = denominator_count
    actual_value = _calculate_actual_value(
        MetricCountRecord(
            numerator_count=numerator_count,
            denominator_count=denominator_count,
        )
    )
    status = _decide_status(
        actual_value=actual_value,
        target_value=target_value,
        denominator_count=denominator_count,
        sample_size=sample_size,
        min_sample_size=min_sample_size,
    )
    return _RunAggregate(
        status=status,
        actual_value=actual_value,
        numerator_count=numerator_count,
        denominator_count=denominator_count,
        sample_size=sample_size,
        next_loop_required=next_loop_required,
        failed_segment_ids=failed_segment_ids,
        failed_ad_experiment_ids=failed_ad_experiment_ids,
        result_json={
            "basis": basis,
            "metric": metric,
            "target_value": str(target_value),
            "actual_value": str(actual_value),
            "numerator_count": numerator_count,
            "denominator_count": denominator_count,
            "sample_size": sample_size,
            "status_reason": _status_reason(
                status=status,
                denominator_count=denominator_count,
                sample_size=sample_size,
                min_sample_size=min_sample_size,
            ),
            "ad_experiment_results": experiment_results,
            "failed_segment_ids": failed_segment_ids,
            "failed_ad_experiment_ids": failed_ad_experiment_ids,
            "min_sample_size": min_sample_size,
            "promotion_run_id": run.promotion_run_id,
        },
    )


def _validate_individual_evaluation(
    evaluation: PromotionEvaluationRecord,
    metric: str,
) -> None:
    if evaluation.metric != metric:
        raise PromotionRunEvaluationValidationError(
            "ad experiment evaluation metric must match promotion run goal metric"
        )
    if evaluation.ad_experiment_id is None or evaluation.segment_id is None:
        raise PromotionRunEvaluationValidationError(
            "individual ad experiment evaluation must include experiment and segment ids"
        )
    try:
        PromotionEvaluationStatus(evaluation.status)
    except ValueError as exc:
        raise PromotionRunEvaluationValidationError(
            "ad experiment evaluation status is out of Owner 2 MVP scope"
        ) from exc


def _aggregate_all_segments_status(
    evaluations: list[PromotionEvaluationRecord],
) -> str:
    statuses = [evaluation.status for evaluation in evaluations]
    if all(status == PromotionEvaluationStatus.GOAL_MET.value for status in statuses):
        return PromotionEvaluationStatus.GOAL_MET.value
    if all(
        status == PromotionEvaluationStatus.INSUFFICIENT_DATA.value
        for status in statuses
    ):
        return PromotionEvaluationStatus.INSUFFICIENT_DATA.value
    has_met = PromotionEvaluationStatus.GOAL_MET.value in statuses
    has_failed = PromotionEvaluationStatus.GOAL_NOT_MET.value in statuses
    if not has_met and has_failed:
        return PromotionEvaluationStatus.GOAL_NOT_MET.value
    return PromotionEvaluationStatus.PARTIAL_GOAL_MET.value


def _evaluation_summary(
    evaluation: PromotionEvaluationRecord,
) -> dict[str, Any]:
    return {
        "evaluation_id": evaluation.evaluation_id,
        "ad_experiment_id": evaluation.ad_experiment_id,
        "segment_id": evaluation.segment_id,
        "actual_value": str(evaluation.actual_value),
        "target_value": str(evaluation.target_value),
        "status": evaluation.status,
        "numerator_count": evaluation.numerator_count,
        "denominator_count": evaluation.denominator_count,
        "sample_size": evaluation.sample_size,
    }


def _aggregate_status_reason(status: str, basis: str) -> str:
    if status == PromotionEvaluationStatus.GOAL_MET.value:
        return f"{basis}_all_goal_met"
    if status == PromotionEvaluationStatus.GOAL_NOT_MET.value:
        return f"{basis}_all_failed_or_no_success"
    if status == PromotionEvaluationStatus.INSUFFICIENT_DATA.value:
        return f"{basis}_only_insufficient_data"
    return f"{basis}_mixed_status"


def _calculate_actual_value(counts: MetricCountRecord) -> Decimal:
    if counts.denominator_count == 0:
        return Decimal("0.000000")
    return (Decimal(counts.numerator_count) / Decimal(counts.denominator_count)).quantize(
        DECIMAL_SCALE,
        rounding=ROUND_HALF_UP,
    )


def _decide_status(
    *,
    actual_value: Decimal,
    target_value: Decimal,
    denominator_count: int,
    sample_size: int,
    min_sample_size: int,
) -> str:
    if denominator_count == 0 or sample_size < min_sample_size:
        return PromotionEvaluationStatus.INSUFFICIENT_DATA.value
    if actual_value >= target_value:
        return PromotionEvaluationStatus.GOAL_MET.value
    return PromotionEvaluationStatus.GOAL_NOT_MET.value


def _build_result_json(
    *,
    experiment: AdExperimentRecord,
    counts: MetricCountRecord,
    target_value: Decimal,
    actual_value: Decimal,
    min_sample_size: int,
    status: str,
) -> dict[str, Any]:
    return {
        "metric_source": _metric_source(experiment.goal_metric),
        "event_names": _event_names(experiment.goal_metric),
        "status_reason": _status_reason(
            status=status,
            denominator_count=counts.denominator_count,
            sample_size=counts.denominator_count,
            min_sample_size=min_sample_size,
        ),
        "min_sample_size": min_sample_size,
        "target_value": str(target_value),
        "actual_value": str(actual_value),
        "numerator_count": counts.numerator_count,
        "denominator_count": counts.denominator_count,
    }


def _metric_source(metric: str) -> str:
    if metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
        return "promotion_touch_events + booking_outcome_events"
    return "promotion_touch_events"


def _event_names(metric: str) -> dict[str, str]:
    if metric == GoalMetric.INFLOW_RATE.value:
        return {
            "numerator": "campaign_landing",
            "denominator": "campaign_redirect_click",
        }
    if metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
        return {
            "numerator": "booking_complete",
            "denominator": "promotion_click",
        }
    return {}


def _status_reason(
    *,
    status: str,
    denominator_count: int,
    sample_size: int,
    min_sample_size: int,
) -> str:
    if denominator_count == 0:
        return "denominator_zero"
    if sample_size < min_sample_size:
        return "sample_size_below_minimum"
    if status == PromotionEvaluationStatus.GOAL_MET.value:
        return "target_met"
    return "target_not_met"
