from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWriter,
    EvaluationMetricReader,
    MetricCountRecord,
    PromotionEvaluationWrite,
    PromotionEvaluationWriter,
    PromotionRunWriter,
)
from app.decision.schemas import (
    AdExperimentEvaluateRequest,
    AdExperimentEvaluateResponse,
    AdExperimentStatus,
    GoalBasis,
    GoalMetric,
    PromotionEvaluationStatus,
)
from app.decision.service import build_bounded_decision_id


DECIMAL_SCALE = Decimal("0.000001")


class AdExperimentEvaluationNotFoundError(Exception):
    pass


class AdExperimentEvaluationValidationError(Exception):
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
