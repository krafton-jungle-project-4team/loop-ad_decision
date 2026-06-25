from dataclasses import dataclass
from datetime import datetime

from app.anomalies.schemas import (
    FunnelAnomalyEvaluation,
    FunnelAnomalyRequest,
    FunnelAnomalyResponse,
    FunnelAnomalyStatus,
    VolumeAnomalyEvaluation,
)
from app.metrics.repository import FunnelMetricsRepository
from app.metrics.schemas import FunnelMetricRequest, FunnelMetrics
from app.metrics.service import calculate_funnel_metrics


@dataclass(frozen=True)
class FunnelAnomalyMetricConfig:
    metric: str
    funnel_step: str
    denominator: str


@dataclass(frozen=True)
class VolumeAnomalyMetricConfig:
    metric: str


FUNNEL_ANOMALY_METRICS = (
    FunnelAnomalyMetricConfig(
        metric="view_to_cart_rate",
        funnel_step="product_view_to_add_to_cart",
        denominator="product_view_sessions",
    ),
    FunnelAnomalyMetricConfig(
        metric="cart_to_checkout_rate",
        funnel_step="add_to_cart_to_checkout_start",
        denominator="add_to_cart_sessions",
    ),
    FunnelAnomalyMetricConfig(
        metric="checkout_to_purchase_rate",
        funnel_step="checkout_start_to_purchase",
        denominator="checkout_start_sessions",
    ),
    FunnelAnomalyMetricConfig(
        metric="view_to_purchase_rate",
        funnel_step="product_view_to_purchase",
        denominator="product_view_sessions",
    ),
)

VOLUME_ANOMALY_METRICS = (
    VolumeAnomalyMetricConfig(metric="product_view_sessions"),
    VolumeAnomalyMetricConfig(metric="add_to_cart_sessions"),
    VolumeAnomalyMetricConfig(metric="checkout_start_sessions"),
    VolumeAnomalyMetricConfig(metric="purchase_sessions"),
)


def resolve_baseline_window(request: FunnelAnomalyRequest) -> tuple[datetime, datetime]:
    if request.baseline_start is not None and request.baseline_end is not None:
        return request.baseline_start, request.baseline_end

    window_duration = request.window_end - request.window_start
    return request.window_start - window_duration, request.window_start


def calculate_funnel_anomalies(
    request: FunnelAnomalyRequest,
    repository: FunnelMetricsRepository,
) -> FunnelAnomalyResponse:
    baseline_start, baseline_end = resolve_baseline_window(request)
    current_response = calculate_funnel_metrics(
        FunnelMetricRequest(
            project_id=request.project_id,
            window_start=request.window_start,
            window_end=request.window_end,
            filters=request.filters,
        ),
        repository,
    )
    baseline_response = calculate_funnel_metrics(
        FunnelMetricRequest(
            project_id=request.project_id,
            window_start=baseline_start,
            window_end=baseline_end,
            filters=request.filters,
        ),
        repository,
    )

    evaluations = [
        evaluate_funnel_metric(
            config=config,
            current_metrics=current_response.metrics,
            baseline_metrics=baseline_response.metrics,
            request=request,
        )
        for config in FUNNEL_ANOMALY_METRICS
    ]
    anomalies = [
        evaluation
        for evaluation in evaluations
        if evaluation.severity in {"warning", "critical"}
    ]
    volume_evaluations = (
        [
            evaluate_volume_metric(
                config=config,
                current_metrics=current_response.metrics,
                baseline_metrics=baseline_response.metrics,
                request=request,
            )
            for config in VOLUME_ANOMALY_METRICS
        ]
        if request.include_volume_anomalies
        else []
    )
    volume_anomalies = [
        evaluation
        for evaluation in volume_evaluations
        if evaluation.severity in {"warning", "critical"}
    ]
    primary_anomaly = resolve_primary_anomaly(anomalies, volume_anomalies)
    status = resolve_response_status(evaluations, volume_evaluations)

    return FunnelAnomalyResponse(
        project_id=request.project_id,
        window_start=current_response.window_start,
        window_end=current_response.window_end,
        baseline_start=baseline_response.window_start,
        baseline_end=baseline_response.window_end,
        segment=current_response.segment,
        status=status,
        current_metrics=current_response.metrics,
        baseline_metrics=baseline_response.metrics,
        evaluations=evaluations,
        anomalies=anomalies,
        volume_evaluations=volume_evaluations,
        volume_anomalies=volume_anomalies,
        primary_anomaly=primary_anomaly,
        summary_message=build_summary_message(status, primary_anomaly),
    )


def evaluate_funnel_metric(
    config: FunnelAnomalyMetricConfig,
    current_metrics: FunnelMetrics,
    baseline_metrics: FunnelMetrics,
    request: FunnelAnomalyRequest,
) -> FunnelAnomalyEvaluation:
    current_value = getattr(current_metrics, config.metric)
    baseline_value = getattr(baseline_metrics, config.metric)
    current_denominator = getattr(current_metrics, config.denominator)
    baseline_denominator = getattr(baseline_metrics, config.denominator)

    delta_point = calculate_delta(current_value, baseline_value)
    drop_point = calculate_drop(current_value, baseline_value)
    relative_change = calculate_relative_change(current_value, baseline_value)
    relative_drop = calculate_relative_drop(current_value, baseline_value)
    severity = resolve_metric_severity(
        current_value=current_value,
        baseline_value=baseline_value,
        drop_point=drop_point,
        relative_drop=relative_drop,
        current_denominator=current_denominator,
        baseline_denominator=baseline_denominator,
        request=request,
    )

    return FunnelAnomalyEvaluation(
        metric=config.metric,
        funnel_step=config.funnel_step,
        severity=severity,
        current_value=current_value,
        baseline_value=baseline_value,
        delta_point=delta_point,
        relative_change=relative_change,
        drop_point=drop_point,
        relative_drop=relative_drop,
        current_denominator=current_denominator,
        baseline_denominator=baseline_denominator,
        min_sample_size=request.min_sample_size,
        message=build_evaluation_message(
            funnel_step=config.funnel_step,
            severity=severity,
            current_value=current_value,
            baseline_value=baseline_value,
            current_denominator=current_denominator,
            baseline_denominator=baseline_denominator,
            min_sample_size=request.min_sample_size,
        ),
    )


def evaluate_volume_metric(
    config: VolumeAnomalyMetricConfig,
    current_metrics: FunnelMetrics,
    baseline_metrics: FunnelMetrics,
    request: FunnelAnomalyRequest,
) -> VolumeAnomalyEvaluation:
    current_value = getattr(current_metrics, config.metric)
    baseline_value = getattr(baseline_metrics, config.metric)
    delta = current_value - baseline_value
    drop = baseline_value - current_value
    relative_change = calculate_volume_relative_change(current_value, baseline_value)
    relative_drop = calculate_volume_relative_drop(current_value, baseline_value)
    severity = resolve_volume_severity(
        baseline_value=baseline_value,
        relative_drop=relative_drop,
        request=request,
    )

    return VolumeAnomalyEvaluation(
        metric=config.metric,
        severity=severity,
        current_value=current_value,
        baseline_value=baseline_value,
        delta=delta,
        relative_change=relative_change,
        drop=drop,
        relative_drop=relative_drop,
        min_volume_count=request.min_volume_count,
        message=build_volume_message(
            metric=config.metric,
            severity=severity,
            current_value=current_value,
            baseline_value=baseline_value,
            drop=drop,
            relative_drop=relative_drop,
            min_volume_count=request.min_volume_count,
        ),
    )


def calculate_delta(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    return current_value - baseline_value


def calculate_drop(current_value: float | None, baseline_value: float | None) -> float | None:
    if current_value is None or baseline_value is None:
        return None
    return baseline_value - current_value


def calculate_relative_change(
    current_value: float | None,
    baseline_value: float | None,
) -> float | None:
    if current_value is None or baseline_value is None or baseline_value <= 0:
        return None
    return (current_value - baseline_value) / baseline_value


def calculate_relative_drop(
    current_value: float | None,
    baseline_value: float | None,
) -> float | None:
    if current_value is None or baseline_value is None or baseline_value <= 0:
        return None
    return (baseline_value - current_value) / baseline_value


def calculate_volume_relative_change(current_value: int, baseline_value: int) -> float | None:
    if baseline_value <= 0:
        return None
    return (current_value - baseline_value) / baseline_value


def calculate_volume_relative_drop(current_value: int, baseline_value: int) -> float | None:
    if baseline_value <= 0:
        return None
    return (baseline_value - current_value) / baseline_value


def resolve_metric_severity(
    *,
    current_value: float | None,
    baseline_value: float | None,
    drop_point: float | None,
    relative_drop: float | None,
    current_denominator: int,
    baseline_denominator: int,
    request: FunnelAnomalyRequest,
) -> FunnelAnomalyStatus:
    if (
        current_denominator < request.min_sample_size
        or baseline_denominator < request.min_sample_size
    ):
        return "insufficient_data"

    if current_value is None or baseline_value is None:
        return "insufficient_data"

    if baseline_value <= 0:
        return "insufficient_data"

    if drop_point is None or relative_drop is None:
        return "insufficient_data"

    if (
        drop_point >= request.critical_abs_drop
        or relative_drop >= request.critical_relative_drop
    ):
        return "critical"

    if (
        drop_point >= request.warning_abs_drop
        or relative_drop >= request.warning_relative_drop
    ):
        return "warning"

    return "normal"


def resolve_volume_severity(
    *,
    baseline_value: int,
    relative_drop: float | None,
    request: FunnelAnomalyRequest,
) -> FunnelAnomalyStatus:
    if baseline_value <= 0:
        return "insufficient_data"

    if baseline_value < request.min_volume_count:
        return "insufficient_data"

    if relative_drop is None:
        return "insufficient_data"

    if relative_drop >= request.critical_volume_relative_drop:
        return "critical"

    if relative_drop >= request.warning_volume_relative_drop:
        return "warning"

    return "normal"


def resolve_response_status(
    evaluations: list[FunnelAnomalyEvaluation],
    volume_evaluations: list[VolumeAnomalyEvaluation],
) -> FunnelAnomalyStatus:
    all_evaluations = [*evaluations, *volume_evaluations]
    severities = [evaluation.severity for evaluation in all_evaluations]
    if "critical" in severities:
        return "critical"
    if "warning" in severities:
        return "warning"
    if severities and all(severity == "insufficient_data" for severity in severities):
        return "insufficient_data"
    return "normal"


def resolve_primary_anomaly(
    anomalies: list[FunnelAnomalyEvaluation],
    volume_anomalies: list[VolumeAnomalyEvaluation],
) -> FunnelAnomalyEvaluation | VolumeAnomalyEvaluation | None:
    combined_anomalies = [*anomalies, *volume_anomalies]
    if not combined_anomalies:
        return None

    return max(combined_anomalies, key=build_primary_anomaly_sort_key)


def build_primary_anomaly_sort_key(
    anomaly: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation,
) -> tuple[int, float]:
    severity_rank = 2 if anomaly.severity == "critical" else 1
    relative_drop = anomaly.relative_drop
    if relative_drop is not None:
        return severity_rank, relative_drop

    if isinstance(anomaly, VolumeAnomalyEvaluation):
        return severity_rank, float(anomaly.drop)

    return severity_rank, float(anomaly.drop_point or 0)


def build_evaluation_message(
    *,
    funnel_step: str,
    severity: FunnelAnomalyStatus,
    current_value: float | None,
    baseline_value: float | None,
    current_denominator: int,
    baseline_denominator: int,
    min_sample_size: int,
) -> str:
    if severity == "insufficient_data":
        if current_denominator < min_sample_size or baseline_denominator < min_sample_size:
            return (
                f"{funnel_step} has insufficient sample size: "
                f"current denominator {current_denominator}, "
                f"baseline denominator {baseline_denominator}, "
                f"minimum {min_sample_size}."
            )
        if current_value is None or baseline_value is None:
            return f"{funnel_step} conversion rate could not be calculated."
        return f"{funnel_step} baseline conversion rate must be greater than 0."

    if severity in {"warning", "critical"}:
        return (
            f"{funnel_step} conversion rate dropped from "
            f"{baseline_value:.4f} to {current_value:.4f}."
        )

    return (
        f"{funnel_step} conversion rate changed from "
        f"{baseline_value:.4f} to {current_value:.4f}."
    )


def build_volume_message(
    *,
    metric: str,
    severity: FunnelAnomalyStatus,
    current_value: int,
    baseline_value: int,
    drop: int,
    relative_drop: float | None,
    min_volume_count: int,
) -> str:
    if severity == "insufficient_data":
        if baseline_value <= 0:
            return f"{metric} baseline volume must be greater than 0."
        if baseline_value < min_volume_count:
            return (
                f"{metric} has insufficient baseline volume: "
                f"baseline {baseline_value}, minimum {min_volume_count}."
            )
        return f"{metric} volume could not be evaluated."

    if severity in {"warning", "critical"}:
        if relative_drop is not None:
            return f"{metric} dropped by {relative_drop * 100:.1f}% compared with baseline."
        return f"{metric} dropped by {drop} compared with baseline."

    return f"{metric} changed from {baseline_value} to {current_value}."


def build_summary_message(
    status: FunnelAnomalyStatus,
    primary_anomaly: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation | None,
) -> str:
    if primary_anomaly is None:
        if status == "insufficient_data":
            return "Not enough data to determine funnel anomaly."
        return "No funnel anomaly detected."

    if isinstance(primary_anomaly, VolumeAnomalyEvaluation):
        return primary_anomaly.message

    return primary_anomaly.message
