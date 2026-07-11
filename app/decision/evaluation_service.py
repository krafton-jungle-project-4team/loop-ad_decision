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
    Channel,
    GoalBasis,
    GoalMetric,
    PromotionRunAdExperimentResult,
    PromotionRunEvaluateRequest,
    PromotionRunEvaluateResponse,
    PromotionRunStatus,
    PromotionEvaluationStatus,
)
from app.decision.service import build_bounded_decision_id
from app.logging import log, log_context_scope, now_ms, duration_ms


DECIMAL_SCALE = Decimal("0.000001")
EVALUATOR_VERSION = "dec.target-threshold-evaluator.v1"
METRIC_SQL_VERSION = "dec.evaluation-metric-sql.v1"
EVALUATION_MODE = "target_threshold"


def _normalize_utc_milliseconds(value: datetime, *, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    normalized = value.astimezone(UTC)
    return normalized.replace(microsecond=(normalized.microsecond // 1000) * 1000)


@dataclass(frozen=True)
class EvaluationContext:
    evaluation_cutoff_at: datetime
    window_start: datetime | None = None
    evaluator_version: str = EVALUATOR_VERSION
    metric_sql_version: str = METRIC_SQL_VERSION

    def __post_init__(self) -> None:
        cutoff = _normalize_utc_milliseconds(
            self.evaluation_cutoff_at,
            field_name="evaluation_cutoff_at",
        )
        object.__setattr__(self, "evaluation_cutoff_at", cutoff)
        if self.window_start is not None:
            window_start = _normalize_utc_milliseconds(
                self.window_start,
                field_name="window_start",
            )
            if window_start > cutoff:
                raise ValueError("window_start must not be after evaluation_cutoff_at")
            object.__setattr__(self, "window_start", window_start)
        if not self.evaluator_version:
            raise ValueError("evaluator_version must not be empty")
        if not self.metric_sql_version:
            raise ValueError("metric_sql_version must not be empty")


def _new_evaluation_context() -> EvaluationContext:
    return EvaluationContext(evaluation_cutoff_at=datetime.now(UTC))


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

    @log_context_scope
    def evaluate(
        self,
        *,
        ad_experiment_id: str,
        request: AdExperimentEvaluateRequest,
    ) -> AdExperimentEvaluateResponse:
        started_at = now_ms()
        log.assign_context({"adExperimentId": ad_experiment_id})
        log.info("started", {"adExperimentId": ad_experiment_id, "request": request})
        evaluation = self.evaluate_with_context(
            ad_experiment_id=ad_experiment_id,
            request=request,
            context=_new_evaluation_context(),
        )
        response = _to_ad_experiment_response(evaluation)
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response

    def evaluate_with_context(
        self,
        *,
        ad_experiment_id: str,
        request: AdExperimentEvaluateRequest,
        context: EvaluationContext,
    ) -> PromotionEvaluationWrite:
        _ = request
        experiment = self._ad_experiment_repository.get_by_id(ad_experiment_id)
        if experiment is None:
            log.warn("ad_experiment_not_found", {"adExperimentId": ad_experiment_id})
            raise AdExperimentEvaluationNotFoundError(
                f"ad experiment not found: {ad_experiment_id}"
            )
        log.assign_context(
            {
                "projectId": experiment.project_id,
                "campaignId": experiment.campaign_id,
                "promotionId": experiment.promotion_id,
                "promotionRunId": experiment.promotion_run_id,
                "segmentId": experiment.segment_id,
                "contentId": experiment.content_id,
                "contentOptionId": experiment.content_option_id,
            }
        )
        log.info("ad_experiment_loaded", {"adExperiment": experiment})

        run = self._promotion_run_repository.get_by_id(experiment.promotion_run_id)
        if run is None:
            log.warn("promotion_run_not_found", {"promotionRunId": experiment.promotion_run_id})
            raise AdExperimentEvaluationValidationError(
                f"promotion run not found: {experiment.promotion_run_id}"
            )
        log.info("promotion_run_loaded", {"promotionRun": run})

        metric = experiment.goal_metric
        if metric == GoalMetric.FUNNEL_STEP_RATE.value:
            log.warn("goal_metric_unsupported", {"metric": metric})
            raise AdExperimentEvaluationValidationError(
                "funnel_step_rate is out of Owner 2 MVP evaluation scope"
            )

        target_value = _parse_target_value(run.goal_snapshot_json)
        min_sample_size = _parse_min_sample_size(run.goal_snapshot_json)
        counts = self._load_counts(experiment, context=context)
        log.info("metric_counts_loaded", {"counts": counts})
        _validate_metric_counts(counts)
        sample_size = counts.denominator_count
        actual_value = _calculate_actual_value(counts)
        _validate_metric_rate(actual_value)
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
                context=context,
            ),
        )
        self._promotion_evaluation_repository.insert(evaluation)
        self._ad_experiment_repository.update_status(
            ad_experiment_id=experiment.ad_experiment_id,
            status=status,
        )
        log.info("promotion_evaluation_created", {"evaluation": evaluation, "status": status})
        return evaluation

    def _load_counts(
        self,
        experiment: AdExperimentRecord,
        *,
        context: EvaluationContext,
    ) -> MetricCountRecord:
        if experiment.goal_metric == GoalMetric.INFLOW_RATE.value:
            return self._evaluation_metric_repository.count_inflow_rate(
                experiment,
                evaluation_cutoff_at=context.evaluation_cutoff_at,
            )
        if experiment.goal_metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
            return self._evaluation_metric_repository.count_booking_conversion_rate(
                experiment,
                evaluation_cutoff_at=context.evaluation_cutoff_at,
            )
        log.warn("goal_metric_unsupported", {"metric": experiment.goal_metric})
        raise AdExperimentEvaluationValidationError(
            f"unsupported goal metric: {experiment.goal_metric}"
        )


def _to_ad_experiment_response(
    evaluation: PromotionEvaluationWrite,
) -> AdExperimentEvaluateResponse:
    if evaluation.ad_experiment_id is None or evaluation.segment_id is None:
        raise AdExperimentEvaluationValidationError(
            "individual evaluation must include experiment and segment ids"
        )
    return AdExperimentEvaluateResponse(
        evaluation_id=evaluation.evaluation_id,
        ad_experiment_id=evaluation.ad_experiment_id,
        promotion_run_id=evaluation.promotion_run_id,
        promotion_id=evaluation.promotion_id,
        segment_id=evaluation.segment_id,
        metric=GoalMetric(evaluation.metric),
        target_value=evaluation.target_value,
        actual_value=evaluation.actual_value,
        numerator_count=evaluation.numerator_count,
        denominator_count=evaluation.denominator_count,
        sample_size=evaluation.sample_size,
        basis=GoalBasis.ALL_SEGMENTS,
        status=PromotionEvaluationStatus(evaluation.status),
        next_loop_required=evaluation.next_loop_required,
        feedback=evaluation.feedback,
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

    @log_context_scope
    def evaluate(
        self,
        *,
        promotion_run_id: str,
        request: PromotionRunEvaluateRequest,
    ) -> PromotionRunEvaluateResponse:
        started_at = now_ms()
        log.assign_context({"promotionRunId": promotion_run_id})
        log.info("started", {"promotionRunId": promotion_run_id, "request": request})
        _ = request
        context = _new_evaluation_context()
        run = self._promotion_run_repository.get_by_id(promotion_run_id)
        if run is None:
            log.warn("promotion_run_not_found", {"promotionRunId": promotion_run_id})
            raise PromotionRunEvaluationNotFoundError(
                f"promotion run not found: {promotion_run_id}"
            )
        log.assign_context(
            {
                "projectId": run.project_id,
                "campaignId": run.campaign_id,
                "promotionId": run.promotion_id,
                "analysisId": run.analysis_id,
                "generationId": run.generation_id,
            }
        )
        log.info("promotion_run_loaded", {"promotionRun": run})

        experiments = self._ad_experiment_repository.list_by_run(promotion_run_id)
        if not experiments:
            log.warn("ad_experiments_empty", {"promotionRunId": promotion_run_id})
            raise PromotionRunEvaluationValidationError(
                f"ad experiments not found for promotion run: {promotion_run_id}"
            )
        log.info("ad_experiments_loaded", {"adExperimentCount": len(experiments)})

        metric = _parse_goal_metric(run.goal_snapshot_json)
        if metric == GoalMetric.FUNNEL_STEP_RATE.value:
            log.warn("goal_metric_unsupported", {"metric": metric})
            raise PromotionRunEvaluationValidationError(
                "funnel_step_rate is out of Owner 2 MVP evaluation scope"
            )
        basis = _parse_goal_basis(run.goal_snapshot_json)
        target_value = _parse_run_target_value(run.goal_snapshot_json)
        min_sample_size = _parse_run_min_sample_size(run.goal_snapshot_json)

        request_evaluations: list[PromotionEvaluationWrite] = []
        for experiment in experiments:
            log.info(
                "ad_experiment_evaluation_started",
                {"adExperimentId": experiment.ad_experiment_id},
            )
            evaluation = self._ad_experiment_evaluation_service.evaluate_with_context(
                ad_experiment_id=experiment.ad_experiment_id,
                request=AdExperimentEvaluateRequest(),
                context=context,
            )
            _validate_individual_evaluation(evaluation, metric)
            request_evaluations.append(evaluation)

        aggregate = _aggregate_run_evaluations(
            run=run,
            evaluations=request_evaluations,
            metric=metric,
            basis=basis,
            target_value=target_value,
            min_sample_size=min_sample_size,
            context=context,
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
        log.info("promotion_run_evaluation_created", {"evaluation": evaluation, "status": aggregate.status})

        response = PromotionRunEvaluateResponse(
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
        log.info("completed", {"response": response, "durationMs": duration_ms(started_at)})
        return response


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
    evaluations: list[PromotionEvaluationRecord | PromotionEvaluationWrite],
    metric: str,
    basis: str,
    target_value: Decimal,
    min_sample_size: int,
    context: EvaluationContext,
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
                **_evaluation_context_json(
                    context,
                    evaluation_scope="promotion_run_aggregate",
                ),
                "basis": basis,
                "metric": metric,
                "event_names": _aggregate_event_names(evaluations),
                "target_value": str(target_value),
                "numerator_count": 0,
                "denominator_count": 0,
                "sample_size": 0,
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
    aggregate_counts = MetricCountRecord(
        numerator_count=numerator_count,
        denominator_count=denominator_count,
    )
    _validate_metric_counts(aggregate_counts)
    actual_value = _calculate_actual_value(aggregate_counts)
    _validate_metric_rate(actual_value)
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
            **_evaluation_context_json(
                context,
                evaluation_scope="promotion_run_aggregate",
            ),
            "basis": basis,
            "metric": metric,
            "event_names": _aggregate_event_names(evaluations),
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
    evaluation: PromotionEvaluationRecord | PromotionEvaluationWrite,
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
    evaluations: list[PromotionEvaluationRecord | PromotionEvaluationWrite],
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
    evaluation: PromotionEvaluationRecord | PromotionEvaluationWrite,
) -> dict[str, Any]:
    return {
        "evaluation_id": evaluation.evaluation_id,
        "ad_experiment_id": evaluation.ad_experiment_id,
        "segment_id": evaluation.segment_id,
        "actual_value": str(evaluation.actual_value),
        "target_value": str(evaluation.target_value),
        "status": evaluation.status,
        "status_reason": evaluation.result_json.get("status_reason"),
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


def _validate_metric_counts(counts: MetricCountRecord) -> None:
    if counts.numerator_count < 0 or counts.denominator_count < 0:
        raise AdExperimentEvaluationValidationError(
            "metric counts must not be negative"
        )
    if counts.numerator_count > counts.denominator_count:
        raise AdExperimentEvaluationValidationError(
            "metric numerator_count must not exceed denominator_count"
        )


def _validate_metric_rate(actual_value: Decimal) -> None:
    if actual_value < 0 or actual_value > 1:
        raise AdExperimentEvaluationValidationError(
            "metric actual_value must be between 0 and 1"
        )


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


def _evaluation_context_json(
    context: EvaluationContext,
    *,
    evaluation_scope: str,
) -> dict[str, Any]:
    return {
        "evaluation_cutoff_at": _rfc3339_milliseconds(
            context.evaluation_cutoff_at
        ),
        "window_start": (
            _rfc3339_milliseconds(context.window_start)
            if context.window_start is not None
            else None
        ),
        "evaluation_mode": EVALUATION_MODE,
        "evaluation_scope": evaluation_scope,
        "evaluator_version": context.evaluator_version,
        "metric_sql_version": context.metric_sql_version,
    }


def _rfc3339_milliseconds(value: datetime) -> str:
    normalized = _normalize_utc_milliseconds(value, field_name="datetime")
    return normalized.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _aggregate_event_names(
    evaluations: list[PromotionEvaluationRecord | PromotionEvaluationWrite],
) -> dict[str, str | list[str] | None]:
    event_names_by_role: dict[str, list[str]] = {
        "numerator": [],
        "denominator": [],
    }
    for evaluation in evaluations:
        raw_event_names = evaluation.result_json.get("event_names")
        if not isinstance(raw_event_names, Mapping):
            continue
        for role in event_names_by_role:
            value = raw_event_names.get(role)
            if isinstance(value, str) and value not in event_names_by_role[role]:
                event_names_by_role[role].append(value)
    result: dict[str, str | list[str] | None] = {}
    for role, values in event_names_by_role.items():
        if not values:
            result[role] = None
        elif len(values) == 1:
            result[role] = values[0]
        else:
            result[role] = values
    return result


def _build_result_json(
    *,
    experiment: AdExperimentRecord,
    counts: MetricCountRecord,
    target_value: Decimal,
    actual_value: Decimal,
    min_sample_size: int,
    status: str,
    context: EvaluationContext,
) -> dict[str, Any]:
    return {
        **_evaluation_context_json(context, evaluation_scope="ad_experiment"),
        "metric_source": _metric_source(experiment.goal_metric),
        "event_names": _event_names(experiment.goal_metric, experiment.channel),
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


def _event_names(metric: str, channel: str | None = None) -> dict[str, str]:
    if metric == GoalMetric.INFLOW_RATE.value:
        return {
            "numerator": "campaign_landing",
            "denominator": "campaign_redirect_click",
        }
    if metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
        return {
            "numerator": "booking_complete",
            "denominator": _booking_conversion_denominator_event(channel),
        }
    return {}


def _booking_conversion_denominator_event(channel: str | None) -> str:
    if channel == Channel.EMAIL.value:
        return "campaign_landing"
    return "promotion_click"


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
