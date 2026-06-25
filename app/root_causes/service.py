from dataclasses import dataclass

from app.anomalies.schemas import (
    FunnelAnomalyEvaluation,
    FunnelAnomalyStatus,
    VolumeAnomalyEvaluation,
)
from app.anomalies.service import calculate_funnel_anomalies
from app.metrics.repository import FILTER_COLUMNS
from app.root_causes.repository import GroupedFunnelCountsRow, RootCauseRepository
from app.root_causes.schemas import (
    RootCauseAnalysisRequest,
    RootCauseAnalysisResponse,
    RootCauseCandidate,
    RootCauseSeverity,
    RootCauseTargetAnomaly,
)


@dataclass(frozen=True)
class RateMetricConfig:
    metric: str
    funnel_step: str
    denominator: str
    numerator: str


RATE_METRIC_CONFIGS = {
    "view_to_cart_rate": RateMetricConfig(
        metric="view_to_cart_rate",
        funnel_step="product_view_to_add_to_cart",
        denominator="product_view_sessions",
        numerator="add_to_cart_sessions",
    ),
    "cart_to_checkout_rate": RateMetricConfig(
        metric="cart_to_checkout_rate",
        funnel_step="add_to_cart_to_checkout_start",
        denominator="add_to_cart_sessions",
        numerator="checkout_start_sessions",
    ),
    "checkout_to_purchase_rate": RateMetricConfig(
        metric="checkout_to_purchase_rate",
        funnel_step="checkout_start_to_purchase",
        denominator="checkout_start_sessions",
        numerator="purchase_sessions",
    ),
    "view_to_purchase_rate": RateMetricConfig(
        metric="view_to_purchase_rate",
        funnel_step="product_view_to_purchase",
        denominator="product_view_sessions",
        numerator="purchase_sessions",
    ),
}

VOLUME_METRICS = {
    "product_view_sessions",
    "add_to_cart_sessions",
    "checkout_start_sessions",
    "purchase_sessions",
}

CAUSE_TYPES = {
    "inventory_status": "inventory_issue",
    "product_id": "product_specific_drop",
    "category": "category_specific_drop",
    "campaign_id": "campaign_specific_drop",
    "channel": "channel_specific_drop",
    "device": "device_specific_drop",
    "age_group": "customer_segment_drop",
    "gender": "customer_segment_drop",
}


def calculate_root_causes(
    request: RootCauseAnalysisRequest,
    repository: RootCauseRepository,
) -> RootCauseAnalysisResponse:
    anomaly_request = build_anomaly_request(request)
    anomaly_response = calculate_funnel_anomalies(anomaly_request, repository)
    target_anomaly = build_target_anomaly(anomaly_response.primary_anomaly)

    if anomaly_response.primary_anomaly is None or anomaly_response.status == "normal":
        return RootCauseAnalysisResponse(
            project_id=request.project_id,
            window_start=anomaly_response.window_start,
            window_end=anomaly_response.window_end,
            baseline_start=anomaly_response.baseline_start,
            baseline_end=anomaly_response.baseline_end,
            segment=anomaly_response.segment,
            status=anomaly_response.status,
            target_anomaly=target_anomaly,
            total_candidates_evaluated=0,
            candidates=[],
            summary_message="No root cause candidates because no funnel anomaly was detected.",
        )

    candidate_dimensions = resolve_candidate_dimensions(request)
    all_candidates: list[RootCauseCandidate] = []
    total_candidates_evaluated = 0

    for dimension in candidate_dimensions:
        rows = repository.fetch_grouped_funnel_counts(
            project_id=request.project_id,
            window_start=anomaly_response.window_start,
            window_end=anomaly_response.window_end,
            baseline_start=anomaly_response.baseline_start,
            baseline_end=anomaly_response.baseline_end,
            filters=request.filters,
            dimension=dimension,
            limit=request.candidate_limit,
        )
        total_candidates_evaluated += len(rows)
        for row in rows:
            candidate = build_candidate_for_row(
                request=request,
                row=row,
                dimension=dimension,
                target=anomaly_response.primary_anomaly,
                total_current_denominator=get_total_current_denominator(
                    anomaly_response.primary_anomaly,
                    anomaly_response.current_metrics,
                ),
                total_baseline_value=get_total_baseline_value(
                    anomaly_response.primary_anomaly,
                    anomaly_response.baseline_metrics,
                ),
            )
            if candidate is not None and candidate.severity in {"warning", "critical"}:
                all_candidates.append(candidate)

    sorted_candidates = sorted(
        all_candidates,
        key=build_candidate_sort_key,
        reverse=True,
    )[: request.limit]
    ranked_candidates = [
        candidate.model_copy(update={"rank": index + 1})
        for index, candidate in enumerate(sorted_candidates)
    ]

    return RootCauseAnalysisResponse(
        project_id=request.project_id,
        window_start=anomaly_response.window_start,
        window_end=anomaly_response.window_end,
        baseline_start=anomaly_response.baseline_start,
        baseline_end=anomaly_response.baseline_end,
        segment=anomaly_response.segment,
        status=anomaly_response.status,
        target_anomaly=target_anomaly,
        total_candidates_evaluated=total_candidates_evaluated,
        candidates=ranked_candidates,
        summary_message=build_root_cause_summary_message(ranked_candidates),
    )


def build_anomaly_request(request: RootCauseAnalysisRequest):
    from app.anomalies.schemas import FunnelAnomalyRequest

    return FunnelAnomalyRequest(
        project_id=request.project_id,
        window_start=request.window_start,
        window_end=request.window_end,
        baseline_start=request.baseline_start,
        baseline_end=request.baseline_end,
        filters=request.filters,
        min_sample_size=request.min_sample_size,
        warning_abs_drop=request.warning_abs_drop,
        critical_abs_drop=request.critical_abs_drop,
        warning_relative_drop=request.warning_relative_drop,
        critical_relative_drop=request.critical_relative_drop,
        include_volume_anomalies=request.include_volume_anomalies,
        min_volume_count=request.min_volume_count,
        warning_volume_relative_drop=request.warning_volume_relative_drop,
        critical_volume_relative_drop=request.critical_volume_relative_drop,
    )


def resolve_candidate_dimensions(request: RootCauseAnalysisRequest) -> list[str]:
    if request.candidate_dimensions is not None:
        return list(dict.fromkeys(request.candidate_dimensions))

    filtered_dimensions = (
        set(request.filters.model_dump(exclude_none=True)) if request.filters else set()
    )
    return [
        dimension
        for dimension in FILTER_COLUMNS
        if dimension not in filtered_dimensions
    ]


def build_target_anomaly(
    anomaly: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation | None,
) -> RootCauseTargetAnomaly | None:
    if anomaly is None:
        return None

    if isinstance(anomaly, VolumeAnomalyEvaluation):
        return RootCauseTargetAnomaly(
            metric=anomaly.metric,
            funnel_step=None,
            severity=anomaly.severity,
            current_value=anomaly.current_value,
            baseline_value=anomaly.baseline_value,
            drop_point=float(anomaly.drop),
            relative_drop=anomaly.relative_drop,
            message=anomaly.message,
        )

    return RootCauseTargetAnomaly(
        metric=anomaly.metric,
        funnel_step=anomaly.funnel_step,
        severity=anomaly.severity,
        current_value=anomaly.current_value,
        baseline_value=anomaly.baseline_value,
        drop_point=anomaly.drop_point,
        relative_drop=anomaly.relative_drop,
        message=anomaly.message,
    )


def build_candidate_for_row(
    *,
    request: RootCauseAnalysisRequest,
    row: GroupedFunnelCountsRow,
    dimension: str,
    target: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation,
    total_current_denominator: int,
    total_baseline_value: int,
) -> RootCauseCandidate | None:
    if isinstance(target, VolumeAnomalyEvaluation):
        return build_volume_candidate(
            request=request,
            row=row,
            dimension=dimension,
            metric=target.metric,
            total_baseline_value=total_baseline_value,
        )

    config = RATE_METRIC_CONFIGS.get(target.metric)
    if config is None:
        return None

    return build_rate_candidate(
        request=request,
        row=row,
        dimension=dimension,
        config=config,
        total_current_denominator=total_current_denominator,
    )


def build_rate_candidate(
    *,
    request: RootCauseAnalysisRequest,
    row: GroupedFunnelCountsRow,
    dimension: str,
    config: RateMetricConfig,
    total_current_denominator: int,
) -> RootCauseCandidate | None:
    current_denominator = get_count(row, "current", config.denominator)
    baseline_denominator = get_count(row, "baseline", config.denominator)
    current_numerator = get_count(row, "current", config.numerator)
    baseline_numerator = get_count(row, "baseline", config.numerator)
    current_value = calculate_rate(current_numerator, current_denominator)
    baseline_value = calculate_rate(baseline_numerator, baseline_denominator)

    if current_value is None or baseline_value is None:
        severity: RootCauseSeverity = "insufficient_data"
        drop_point = None
        relative_drop = None
    else:
        drop_point = baseline_value - current_value
        relative_drop = drop_point / baseline_value if baseline_value > 0 else None
        severity = resolve_rate_candidate_severity(
            request=request,
            current_denominator=current_denominator,
            baseline_denominator=baseline_denominator,
            baseline_value=baseline_value,
            drop_point=drop_point,
            relative_drop=relative_drop,
        )

    if severity not in {"warning", "critical"} or drop_point is None or relative_drop is None:
        return None

    support_share = (
        current_denominator / total_current_denominator
        if total_current_denominator > 0
        else 0.0
    )
    excess_lost_sessions = max(drop_point * current_denominator, 0.0)
    score = calculate_candidate_score(severity, relative_drop, support_share)

    return RootCauseCandidate(
        rank=0,
        cause_type=resolve_cause_type(dimension, is_volume=False),
        dimension=dimension,
        value=row.dimension_value,
        metric=config.metric,
        funnel_step=config.funnel_step,
        severity=severity,
        current_value=current_value,
        baseline_value=baseline_value,
        drop_point=drop_point,
        relative_drop=relative_drop,
        current_denominator=current_denominator,
        baseline_denominator=baseline_denominator,
        support_share=support_share,
        excess_lost_sessions=excess_lost_sessions,
        score=score,
        message=(
            f"{dimension}={row.dimension_value}에서 {config.funnel_step} 전환율이 "
            f"{baseline_value:.4f}에서 {current_value:.4f}으로 하락했습니다."
        ),
    )


def build_volume_candidate(
    *,
    request: RootCauseAnalysisRequest,
    row: GroupedFunnelCountsRow,
    dimension: str,
    metric: str,
    total_baseline_value: int,
) -> RootCauseCandidate | None:
    if metric not in VOLUME_METRICS:
        return None

    current_value = get_count(row, "current", metric)
    baseline_value = get_count(row, "baseline", metric)
    drop_point = float(baseline_value - current_value)
    relative_drop = drop_point / baseline_value if baseline_value > 0 else None
    severity = resolve_volume_candidate_severity(
        request=request,
        baseline_value=baseline_value,
        relative_drop=relative_drop,
    )

    if severity not in {"warning", "critical"} or relative_drop is None:
        return None

    support_share = (
        baseline_value / total_baseline_value
        if total_baseline_value > 0
        else 0.0
    )
    excess_lost_sessions = max(drop_point, 0.0)
    score = calculate_candidate_score(severity, relative_drop, support_share)

    return RootCauseCandidate(
        rank=0,
        cause_type=resolve_cause_type(dimension, is_volume=True),
        dimension=dimension,
        value=row.dimension_value,
        metric=metric,
        funnel_step=None,
        severity=severity,
        current_value=current_value,
        baseline_value=baseline_value,
        drop_point=drop_point,
        relative_drop=relative_drop,
        current_denominator=current_value,
        baseline_denominator=baseline_value,
        support_share=support_share,
        excess_lost_sessions=excess_lost_sessions,
        score=score,
        message=(
            f"{dimension}={row.dimension_value}에서 {metric} 볼륨이 "
            f"{baseline_value}에서 {current_value}으로 하락했습니다."
        ),
    )


def calculate_rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return numerator / denominator


def resolve_rate_candidate_severity(
    *,
    request: RootCauseAnalysisRequest,
    current_denominator: int,
    baseline_denominator: int,
    baseline_value: float,
    drop_point: float,
    relative_drop: float | None,
) -> RootCauseSeverity:
    if (
        current_denominator < request.min_sample_size
        or baseline_denominator < request.min_sample_size
    ):
        return "insufficient_data"
    if baseline_value <= 0 or relative_drop is None:
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


def resolve_volume_candidate_severity(
    *,
    request: RootCauseAnalysisRequest,
    baseline_value: int,
    relative_drop: float | None,
) -> RootCauseSeverity:
    if baseline_value < request.min_volume_count:
        return "insufficient_data"
    if relative_drop is None:
        return "insufficient_data"
    if relative_drop >= request.critical_volume_relative_drop:
        return "critical"
    if relative_drop >= request.warning_volume_relative_drop:
        return "warning"
    return "normal"


def calculate_candidate_score(
    severity: str,
    relative_drop: float,
    support_share: float,
) -> float:
    severity_rank = 2 if severity == "critical" else 1
    return severity_rank + relative_drop + support_share


def build_candidate_sort_key(candidate: RootCauseCandidate) -> tuple[int, float, float, float]:
    return (
        2 if candidate.severity == "critical" else 1,
        candidate.score,
        candidate.excess_lost_sessions,
        candidate.relative_drop or 0.0,
    )


def resolve_cause_type(dimension: str, *, is_volume: bool) -> str:
    if is_volume:
        return "traffic_volume_drop"
    return CAUSE_TYPES[dimension]


def get_count(row: GroupedFunnelCountsRow, period: str, metric: str) -> int:
    return getattr(row, f"{period}_{metric}")


def get_total_current_denominator(
    target: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation,
    current_metrics,
) -> int:
    if isinstance(target, VolumeAnomalyEvaluation):
        return 0
    config = RATE_METRIC_CONFIGS.get(target.metric)
    if config is None:
        return 0
    return getattr(current_metrics, config.denominator)


def get_total_baseline_value(
    target: FunnelAnomalyEvaluation | VolumeAnomalyEvaluation,
    baseline_metrics,
) -> int:
    if isinstance(target, VolumeAnomalyEvaluation):
        return getattr(baseline_metrics, target.metric)
    return 0


def build_root_cause_summary_message(candidates: list[RootCauseCandidate]) -> str:
    if not candidates:
        return "No root cause candidates found for the target anomaly."
    top_candidate = candidates[0]
    return (
        f"Top root cause candidate is "
        f"{top_candidate.dimension}={top_candidate.value}."
    )
