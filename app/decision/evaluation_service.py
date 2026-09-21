from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Mapping

from app.decision.matcher import FALLBACK_SEGMENT_ID
from app.decision.repositories import (
    AdExperimentRecord,
    AdExperimentWriter,
    BookingIntentCohortRecord,
    EvaluationFunnelRecord,
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
EVALUATOR_VERSION = "dec.target-threshold-evaluator.v2"
METRIC_SQL_VERSION = "dec.evaluation-metric-sql.v2"
EVALUATION_MODE = "target_threshold"
DETAILED_DIAGNOSIS_MIN_SAMPLE_SIZE = 30


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
        booking_intent_cohorts = self._load_booking_intent_cohorts(
            experiment=experiment,
            run=run,
            status=status,
            context=context,
        )
        diagnosis = _build_evaluation_diagnosis(
            metric=metric,
            channel=experiment.channel,
            status=status,
            target_value=target_value,
            actual_value=actual_value,
            counts=counts,
            min_sample_size=min_sample_size,
            booking_intent_cohorts=booking_intent_cohorts,
            destination_ids=_outcome_destination_ids(run.goal_snapshot_json),
            age_groups=_audience_age_groups(run.goal_snapshot_json),
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
            feedback=(
                None
                if status == PromotionEvaluationStatus.GOAL_MET.value
                else diagnosis["summary"]
            ),
            next_loop_required=next_loop_required,
            result_json=_build_result_json(
                experiment=experiment,
                counts=counts,
                target_value=target_value,
                actual_value=actual_value,
                min_sample_size=min_sample_size,
                status=status,
                context=context,
                diagnosis=diagnosis,
            ),
        )
        self._promotion_evaluation_repository.insert(evaluation)
        log.info("promotion_evaluation_created", {"evaluation": evaluation, "status": status})
        return evaluation

    def _load_booking_intent_cohorts(
        self,
        *,
        experiment: AdExperimentRecord,
        run: PromotionRunRecord,
        status: str,
        context: EvaluationContext,
    ) -> BookingIntentCohortRecord | None:
        if (
            experiment.goal_metric != GoalMetric.BOOKING_CONVERSION_RATE.value
            or status != PromotionEvaluationStatus.GOAL_NOT_MET.value
        ):
            return None
        destination_ids = _outcome_destination_ids(run.goal_snapshot_json)
        try:
            cohorts = self._evaluation_metric_repository.analyze_booking_intent_cohorts(
                experiment,
                destination_ids=destination_ids,
                evaluation_cutoff_at=context.evaluation_cutoff_at,
                lookback_days=30,
            )
            _validate_booking_intent_cohorts(cohorts)
        except Exception as exc:
            log.warn(
                "booking_intent_cohort_analysis_unavailable",
                {
                    "destinationCount": len(destination_ids),
                    "err": exc,
                },
            )
            return None
        log.info(
            "booking_intent_cohort_analysis_completed",
            {
                "comparisonUsers": cohorts.comparison_user_count,
                "repeatViewUsers": cohorts.repeat_view_user_count,
            },
        )
        return cohorts

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

        experiments = [
            experiment
            for experiment in self._ad_experiment_repository.list_by_run(
                promotion_run_id
            )
            if experiment.segment_id != FALLBACK_SEGMENT_ID
        ]
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
                    target_value=Decimal(item["target_value"]),
                    actual_value=Decimal(item["actual_value"]),
                    numerator_count=item["numerator_count"],
                    denominator_count=item["denominator_count"],
                    sample_size=item["sample_size"],
                    status=PromotionEvaluationStatus(item["status"]),
                    feedback=item["feedback"],
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
        "feedback": evaluation.feedback,
        "diagnosis": evaluation.result_json.get("diagnosis"),
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
    if counts.funnel is None:
        return

    funnel_counts = (
        counts.funnel.response_count,
        counts.funnel.hotel_search_count,
        counts.funnel.hotel_detail_view_count,
        counts.funnel.booking_start_count,
        counts.funnel.booking_complete_count,
    )
    if any(value < 0 for value in funnel_counts):
        raise AdExperimentEvaluationValidationError(
            "evaluation funnel counts must not be negative"
        )
    if any(
        current < following
        for current, following in zip(funnel_counts, funnel_counts[1:])
    ):
        raise AdExperimentEvaluationValidationError(
            "evaluation funnel counts must be monotonically non-increasing"
        )
    if counts.funnel.response_count != counts.denominator_count:
        raise AdExperimentEvaluationValidationError(
            "evaluation funnel response_count must match denominator_count"
        )
    if counts.funnel.booking_complete_count != counts.numerator_count:
        raise AdExperimentEvaluationValidationError(
            "evaluation funnel booking_complete_count must match numerator_count"
        )
    if not 0 <= counts.funnel.fixture_response_count <= counts.funnel.response_count:
        raise AdExperimentEvaluationValidationError(
            "evaluation funnel fixture_response_count is invalid"
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
    diagnosis: Mapping[str, Any],
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
        "diagnosis": dict(diagnosis),
    }


def _build_evaluation_diagnosis(
    *,
    metric: str,
    channel: str,
    status: str,
    target_value: Decimal,
    actual_value: Decimal,
    counts: MetricCountRecord,
    min_sample_size: int,
    booking_intent_cohorts: BookingIntentCohortRecord | None = None,
    destination_ids: tuple[str, ...] = (),
    age_groups: tuple[str, ...] = (),
) -> dict[str, Any]:
    gap_percentage_points = max(
        (target_value - actual_value) * Decimal("100"),
        Decimal("0"),
    ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    actual_percent = _format_percent(actual_value)
    target_percent = _format_percent(target_value)
    funnel = _build_evaluation_funnel(
        metric=metric,
        channel=channel,
        counts=counts,
    )
    largest_dropoff = funnel["largest_dropoff"]
    evidence_strength = _evaluation_evidence_strength(
        sample_size=counts.denominator_count,
        min_sample_size=min_sample_size,
    )
    limitations = [
        "동일 광고 실험에 귀속된 유효 이벤트의 고유 사용자만 집계했습니다."
    ]
    if evidence_strength["level"] == "limited":
        limitations.append(
            f"관측 표본이 {counts.denominator_count}명으로 적어 이탈 구간은 참고 수준입니다."
        )
    if counts.funnel is None and metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
        limitations.append(
            "중간 숙박 행동 이벤트가 없어 광고 반응과 예약 완료만 비교했습니다."
        )

    if counts.denominator_count == 0:
        summary = (
            "평가 기준 행동이 아직 0건이라 성과 원인을 판단할 수 없습니다. "
            "광고 발송 상태와 링크·SDK 이벤트 수집을 확인한 뒤 다시 평가하세요."
        )
        bottleneck = "measurement_unavailable"
        evidence = [
            "평가 기준 행동 0건",
            f"최소 평가 표본 {min_sample_size}명",
        ]
        directions = [
            "광고 발송 및 노출 상태 확인",
            "링크 이동과 성과 이벤트 수집 경로 확인",
        ]
    elif counts.denominator_count < min_sample_size:
        summary = (
            f"현재 평가 표본은 {counts.denominator_count}명으로 최소 기준 "
            f"{min_sample_size}명보다 적어 성과 원인을 확정할 수 없습니다. "
            "표본이 쌓인 뒤 다시 평가하세요."
        )
        bottleneck = "sample_size_below_minimum"
        evidence = [
            f"현재 평가 표본 {counts.denominator_count}명",
            f"최소 평가 표본 {min_sample_size}명",
        ]
        directions = [
            "현재 실험을 유지해 평가 표본 확보",
            "발송 및 성과 이벤트 누락 여부 확인",
        ]
    elif status == PromotionEvaluationStatus.GOAL_MET.value:
        summary = (
            f"실제 성과 {actual_percent}로 목표 {target_percent}를 달성했습니다."
        )
        bottleneck = "none"
        evidence = [
            f"성공 {counts.numerator_count}건 / 기준 행동 {counts.denominator_count}건",
            f"실제 성과 {actual_percent} / 목표 {target_percent}",
        ]
        directions = ["현재 메시지와 고객군 전략 유지"]
    elif metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
        if largest_dropoff is None:
            bottleneck_summary = "중간 단계의 이탈 구간은 아직 확인되지 않았습니다."
            bottleneck = "measurement_unavailable"
        else:
            bottleneck_summary = (
                f"가장 큰 관측 이탈은 {largest_dropoff['from_stage_label']}에서 "
                f"{largest_dropoff['to_stage_label']} 단계로 넘어가는 구간으로, "
                f"{largest_dropoff['from_count']}명 중 "
                f"{largest_dropoff['dropoff_count']}명"
                f"({_format_ratio_percent(largest_dropoff['dropoff_rate'])})이 "
                "다음 단계에 도달하지 않았습니다."
            )
            bottleneck = (
                f"{largest_dropoff['from_stage_key']}_to_"
                f"{largest_dropoff['to_stage_key']}"
            )
        summary = (
            f"예약 완료율 {actual_percent}로 목표 {target_percent}보다 "
            f"{gap_percentage_points}%p 낮습니다. {bottleneck_summary}"
        )
        evidence = [
            f"광고 반응 고객 {counts.denominator_count}명 중 예약 완료 {counts.numerator_count}명",
            f"목표 대비 {gap_percentage_points}%p 부족",
        ]
        if largest_dropoff is not None:
            evidence.insert(
                1,
                f"{largest_dropoff['from_stage_label']} {largest_dropoff['from_count']}명 중 "
                f"{largest_dropoff['to_stage_label']} {largest_dropoff['to_count']}명",
            )
        directions = _bottleneck_improvement_directions(bottleneck)
        if bottleneck == "booking_start_to_booking_complete":
            limitations.append(
                "결제 오류, 가격 변경, 객실 소진 같은 상세 실패 이벤트가 없어 직접 원인은 확정할 수 없습니다."
            )
    else:
        summary = (
            f"랜딩 도달률 {actual_percent}로 목표 {target_percent}보다 "
            f"{gap_percentage_points}%p 낮습니다. 측정상 병목은 광고 클릭 이후 "
            "랜딩 도달 단계입니다. 링크 동작을 확인하고 CTA와 랜딩 내용을 "
            "더 일관되게 구성하세요."
        )
        bottleneck = "redirect_click_to_campaign_landing"
        evidence = [
            f"광고 클릭 {counts.denominator_count}건 중 랜딩 도달 {counts.numerator_count}건",
            f"목표 대비 {gap_percentage_points}%p 부족",
        ]
        directions = [
            "리다이렉트 링크와 랜딩 페이지 정상 동작 확인",
            "CTA 문구와 연결되는 랜딩 내용의 일치도 강화",
        ]

    price_abandonment_analysis = _build_price_abandonment_analysis(
        status=status,
        target_value=target_value,
        actual_value=actual_value,
        counts=counts,
        cohorts=booking_intent_cohorts,
        destination_ids=destination_ids,
        age_groups=age_groups,
    )

    return {
        "version": "dec.evaluation-diagnosis.v4",
        "status": status,
        "summary": summary,
        "observed_bottleneck": bottleneck,
        "largest_dropoff": largest_dropoff,
        "evidence": evidence,
        "improvement_directions": directions,
        "gap_percentage_points": str(gap_percentage_points),
        "evidence_strength": evidence_strength,
        "limitations": limitations,
        "data_origin": _evaluation_data_origin(counts.funnel),
        "audience_intent_analysis": None,
        "price_abandonment_analysis": price_abandonment_analysis,
        "funnel": funnel,
    }


def _build_price_abandonment_analysis(
    *,
    status: str,
    target_value: Decimal,
    actual_value: Decimal,
    counts: MetricCountRecord,
    cohorts: BookingIntentCohortRecord | None,
    destination_ids: tuple[str, ...],
    age_groups: tuple[str, ...],
) -> dict[str, Any] | None:
    funnel = counts.funnel
    if (
        status != PromotionEvaluationStatus.GOAL_NOT_MET.value
        or funnel is None
        or cohorts is None
        or cohorts.high_price_booking_start_user_count == 0
    ):
        return None

    destination_label = _destination_display_label(destination_ids)
    destination_phrase = (
        f"{destination_label} 숙소"
        if destination_label
        else "프로모션 대상 숙소"
    )

    paragraphs = [
        (
            f"목표 예약 전환율은 {_format_percent(target_value)}였지만 실제 전환율은 "
            f"{_format_percent(actual_value)}였습니다. 광고 반응 고객 "
            f"{counts.denominator_count}명 중 {funnel.booking_start_count}명이 예약을 "
            f"시작했고 {funnel.booking_complete_count}명만 완료했습니다."
        ),
        (
            f"이 가운데 1박 가격이 20만 원을 초과한 {destination_phrase}를 선택하고 "
            f"예약을 시작한 고객은 {cohorts.high_price_booking_start_user_count}명이었고, "
            f"{cohorts.high_price_booking_abandon_user_count}명이 예약을 완료하지 "
            "않았습니다. 높은 1박 가격이 결제 직전 결정에 부담이 되었을 가능성이 "
            "있습니다."
        ),
        (
            "가격과 예약 이탈의 연관성은 관측됐지만 가격이 직접 원인이라고 단정할 수는 "
            "없습니다. 다음 실험에서 해당 고객에게 추가 할인을 제시해 가설을 검증해 "
            "보세요."
        ),
    ]

    condition_labels: list[str] = []
    if age_groups:
        condition_labels.append(_age_group_display_label(age_groups))
    condition_labels.extend(
        [
            f"최근 7일 {destination_phrase} 1박 가격 20만 원 초과",
            "예약 시작 후 미완료",
        ]
    )
    return {
        "version": "dec.price-abandonment-analysis.v1",
        "title": "높은 1박 가격이 예약 완료에 부담이 되었을 가능성이 있습니다",
        "paragraphs": paragraphs,
        "price_abandonment": {
            "currency": "KRW",
            "nightly_price_threshold": "200000",
            "booking_start_user_count": cohorts.high_price_booking_start_user_count,
            "booking_abandon_user_count": (
                cohorts.high_price_booking_abandon_user_count
            ),
            "booking_complete_user_count": (
                cohorts.high_price_booking_complete_user_count
            ),
            "booking_abandon_median_nightly_price": (
                None
                if cohorts.booking_abandon_median_nightly_price is None
                else str(cohorts.booking_abandon_median_nightly_price)
            ),
            "booking_complete_median_nightly_price": (
                None
                if cohorts.booking_complete_median_nightly_price is None
                else str(cohorts.booking_complete_median_nightly_price)
            ),
        },
        "next_segment_hypothesis": {
            "lookback_days": 7,
            "condition_labels": condition_labels,
            "validation_note": (
                "관측된 가격과 이탈의 연관성을 바탕으로 만든 다음 실험 가설이며 "
                "성공을 보장하지 않습니다."
            ),
        },
    }


def _validate_booking_intent_cohorts(cohorts: BookingIntentCohortRecord) -> None:
    count_fields = (
        cohorts.ad_click_count,
        cohorts.repeat_view_user_count,
        cohorts.repeat_view_booking_count,
        cohorts.comparison_user_count,
        cohorts.comparison_booking_count,
        cohorts.booking_abandon_user_count,
        cohorts.booking_complete_user_count,
        cohorts.high_price_booking_start_user_count,
        cohorts.high_price_booking_abandon_user_count,
        cohorts.high_price_booking_complete_user_count,
    )
    if any(value < 0 for value in count_fields):
        raise ValueError("booking intent cohort counts must not be negative")
    if cohorts.repeat_view_booking_count > cohorts.repeat_view_user_count:
        raise ValueError("repeat-view bookings exceed cohort users")
    if cohorts.comparison_booking_count > cohorts.comparison_user_count:
        raise ValueError("comparison bookings exceed cohort users")
    if (
        cohorts.high_price_booking_abandon_user_count
        + cohorts.high_price_booking_complete_user_count
        > cohorts.high_price_booking_start_user_count
    ):
        raise ValueError("high-price outcomes exceed booking-start users")
    for value in (
        cohorts.booking_abandon_median_revenue,
        cohorts.booking_complete_median_revenue,
        cohorts.booking_abandon_median_nightly_price,
        cohorts.booking_complete_median_nightly_price,
    ):
        if value is not None and value < 0:
            raise ValueError("booking intent cohort revenue must not be negative")


def _outcome_destination_ids(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    outcome_spec = snapshot.get("outcome_spec")
    if not isinstance(outcome_spec, Mapping):
        return ()
    outcome_filter = outcome_spec.get("outcome_filter")
    if not isinstance(outcome_filter, Mapping):
        return ()
    values = outcome_filter.get("destination_ids")
    if not isinstance(values, list):
        return ()
    return tuple(
        sorted(
            {
                str(value).strip().casefold()
                for value in values
                if isinstance(value, str) and value.strip()
            }
        )
    )


def _audience_age_groups(snapshot: Mapping[str, Any]) -> tuple[str, ...]:
    context = snapshot.get("audience_context")
    if not isinstance(context, Mapping):
        return ()
    values = context.get("age_groups")
    if not isinstance(values, list):
        return ()
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if isinstance(value, str) and value.strip()
        )
    )


def _destination_display_label(destination_ids: tuple[str, ...]) -> str:
    labels = {
        "jeju": "제주",
        "okinawa": "오키나와",
        "busan": "부산",
        "gangneung": "강릉",
        "seoul": "서울",
    }
    return "·".join(labels.get(value, value) for value in destination_ids)


def _age_group_display_label(values: tuple[str, ...]) -> str:
    normalized = [
        value.strip()[:-1] if value.strip().endswith("대") else value.strip()
        for value in values
        if value.strip()
    ]
    if len(normalized) == 2:
        return f"{normalized[0]}~{normalized[1]}대"
    return f"{'·'.join(normalized)}대"


def _ratio_decimal(numerator: int, denominator: int) -> Decimal:
    if denominator <= 0:
        return Decimal("0")
    return (Decimal(numerator) / Decimal(denominator)).quantize(
        DECIMAL_SCALE,
        rounding=ROUND_HALF_UP,
    )


def _format_rate_one_decimal(value: Decimal) -> str:
    return f"{(value * Decimal('100')).quantize(Decimal('0.1'), rounding=ROUND_HALF_UP)}%"


def _format_won(value: Decimal) -> str:
    return f"{int(value.quantize(Decimal('1'), rounding=ROUND_HALF_UP)):,}원"


def _build_evaluation_funnel(
    *,
    metric: str,
    channel: str,
    counts: MetricCountRecord,
) -> dict[str, Any]:
    if metric == GoalMetric.INFLOW_RATE.value:
        raw_stages = [
            ("campaign_redirect_click", "광고 링크 클릭", counts.denominator_count),
            ("campaign_landing", "랜딩 도달", counts.numerator_count),
        ]
        counting_method = "attribution_key_reach"
    elif counts.funnel is not None:
        raw_stages = _booking_funnel_stages(channel, counts.funnel)
        counting_method = "cumulative_user_reach_after_ad_response"
    else:
        raw_stages = [
            (
                _booking_conversion_denominator_event(channel),
                _booking_response_label(channel),
                counts.denominator_count,
            ),
            ("booking_complete", "예약 완료", counts.numerator_count),
        ]
        counting_method = "metric_endpoints_only"

    stages: list[dict[str, Any]] = []
    largest_dropoff: dict[str, Any] | None = None
    largest_dropoff_rank: tuple[Decimal, int] | None = None
    for index, (key, label, user_count) in enumerate(raw_stages):
        previous = stages[index - 1] if index > 0 else None
        if previous is None:
            conversion_rate = None
            dropoff_count = None
            dropoff_rate = None
        else:
            previous_count = int(previous["user_count"])
            dropoff_count = max(previous_count - user_count, 0)
            conversion_rate = _ratio_string(user_count, previous_count)
            dropoff_rate = _ratio_string(dropoff_count, previous_count)
            if previous_count > 0 and dropoff_count > 0 and dropoff_rate is not None:
                rank = (Decimal(dropoff_rate), dropoff_count)
                if largest_dropoff_rank is None or rank > largest_dropoff_rank:
                    largest_dropoff_rank = rank
                    largest_dropoff = {
                        "from_stage_key": previous["key"],
                        "from_stage_label": previous["label"],
                        "to_stage_key": key,
                        "to_stage_label": label,
                        "from_count": previous_count,
                        "to_count": user_count,
                        "dropoff_count": dropoff_count,
                        "dropoff_rate": dropoff_rate,
                    }
        stages.append(
            {
                "key": key,
                "label": label,
                "user_count": user_count,
                "conversion_rate_from_previous": conversion_rate,
                "dropoff_count_from_previous": dropoff_count,
                "dropoff_rate_from_previous": dropoff_rate,
            }
        )

    return {
        "counting_method": counting_method,
        "stages": stages,
        "largest_dropoff": largest_dropoff,
    }


def _booking_funnel_stages(
    channel: str,
    funnel: EvaluationFunnelRecord,
) -> list[tuple[str, str, int]]:
    return [
        (
            _booking_conversion_denominator_event(channel),
            _booking_response_label(channel),
            funnel.response_count,
        ),
        ("hotel_search", "숙소 탐색", funnel.hotel_search_count),
        ("hotel_detail_view", "숙소 상세 조회", funnel.hotel_detail_view_count),
        ("booking_start", "예약 시작", funnel.booking_start_count),
        ("booking_complete", "예약 완료", funnel.booking_complete_count),
    ]


def _booking_response_label(channel: str) -> str:
    if channel == Channel.EMAIL.value:
        return "광고 랜딩 도달"
    return "광고 클릭"


def _ratio_string(numerator: int, denominator: int) -> str | None:
    if denominator <= 0:
        return None
    return str(
        (Decimal(numerator) / Decimal(denominator)).quantize(
            DECIMAL_SCALE,
            rounding=ROUND_HALF_UP,
        )
    )


def _format_ratio_percent(value: str) -> str:
    return _format_percent(Decimal(value))


def _evaluation_evidence_strength(
    *,
    sample_size: int,
    min_sample_size: int,
) -> dict[str, Any]:
    if sample_size == 0:
        return {
            "level": "unavailable",
            "sample_size": sample_size,
            "reason": "평가 기준 행동이 없습니다.",
        }
    if sample_size < min_sample_size:
        return {
            "level": "insufficient",
            "sample_size": sample_size,
            "reason": f"최소 평가 표본 {min_sample_size}명보다 적습니다.",
        }
    if sample_size < DETAILED_DIAGNOSIS_MIN_SAMPLE_SIZE:
        return {
            "level": "limited",
            "sample_size": sample_size,
            "reason": (
                f"원인 진단 참고 기준 {DETAILED_DIAGNOSIS_MIN_SAMPLE_SIZE}명보다 적습니다."
            ),
        }
    return {
        "level": "sufficient",
        "sample_size": sample_size,
        "reason": "단계별 이탈을 비교할 수 있는 관측 표본이 확보되었습니다.",
    }


def _evaluation_data_origin(
    funnel: EvaluationFunnelRecord | None,
) -> dict[str, str]:
    if funnel is None or funnel.response_count == 0:
        return {"kind": "observed", "label": "수집 이벤트"}
    if funnel.fixture_response_count == funnel.response_count:
        return {"kind": "demo_fixture", "label": "시연 데이터"}
    if funnel.fixture_response_count > 0:
        return {"kind": "mixed", "label": "수집·시연 혼합 데이터"}
    return {"kind": "observed", "label": "수집 이벤트"}


def _bottleneck_improvement_directions(bottleneck: str) -> list[str]:
    directions_by_bottleneck = {
        "campaign_landing_to_hotel_search": [
            "랜딩 첫 화면에서 목적지·할인·예약 조건이 바로 보이는지 확인",
            "숙소 검색으로 이어지는 CTA와 랜딩 동선을 점검",
        ],
        "promotion_click_to_hotel_search": [
            "광고 메시지와 랜딩의 목적지·혜택 정보가 일치하는지 확인",
            "숙소 검색으로 이어지는 CTA와 랜딩 동선을 점검",
        ],
        "hotel_search_to_hotel_detail_view": [
            "검색 결과에서 가격·혜택·객실 가능 여부가 충분히 구분되는지 확인",
            "추천 숙소 정렬과 상세 진입 동선을 점검",
        ],
        "hotel_detail_view_to_booking_start": [
            "상세 화면의 총 결제금액·취소 조건·객실 가능 여부를 명확히 제시",
            "예약 시작 CTA의 위치와 메시지를 점검",
        ],
        "booking_start_to_booking_complete": [
            "결제 실패·가격 변경·객실 소진 이벤트를 추가 수집해 직접 원인을 확인",
            "예약 단계 입력 항목과 결제 완료 동선을 점검",
        ],
    }
    return directions_by_bottleneck.get(
        bottleneck,
        [
            "광고 반응부터 예약 완료까지 이벤트 수집 상태 확인",
            "가장 이탈이 큰 화면의 메시지와 다음 행동 동선 점검",
        ],
    )


def _format_percent(value: Decimal) -> str:
    percent = (value * Decimal("100")).quantize(
        Decimal("0.01"),
        rounding=ROUND_HALF_UP,
    )
    return f"{percent}%"


def _metric_source(metric: str) -> str:
    if metric == GoalMetric.BOOKING_CONVERSION_RATE.value:
        return "promotion_touch_events + raw_events + booking_outcome_events"
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
