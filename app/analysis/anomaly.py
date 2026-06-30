from __future__ import annotations

from collections.abc import Mapping
from decimal import Decimal

from app.analysis.models import (
    BaselineMetrics,
    RootCauseCandidate,
    SegmentAggregate,
    SegmentAnomalyCandidate,
    StoredAnomaly,
    StoredSegment,
)

BASELINE_DROP_THRESHOLD = Decimal("0.20")

FunnelStepSelection = tuple[str, str, str, str, Decimal]

ROOT_CAUSE_STEPS = (
    (
        "view_to_cart_rate",
        "view_to_cart",
        "상품 조회 후 장바구니 전환 낮음",
        "product_view_to_add_to_cart",
    ),
    (
        "cart_to_checkout_rate",
        "cart_to_checkout",
        "장바구니 후 결제 시작 전환 낮음",
        "add_to_cart_to_checkout_start",
    ),
    (
        "checkout_to_purchase_rate",
        "checkout_to_purchase",
        "결제 시작 후 구매 전환 낮음",
        "checkout_start_to_purchase",
    ),
)

FUNNEL_STEP_LABELS = {
    "product_view_to_add_to_cart": "상품 조회 후 장바구니 추가",
    "add_to_cart_to_checkout_start": "장바구니 추가 후 결제 시작",
    "checkout_start_to_purchase": "결제 시작 후 구매",
}


def detect_segment_anomalies(
    aggregates: list[SegmentAggregate],
    stored_segments: Mapping[str, StoredSegment],
    baselines: Mapping[int, BaselineMetrics],
) -> list[SegmentAnomalyCandidate]:
    candidates: list[SegmentAnomalyCandidate] = []
    for aggregate in aggregates:
        stored_segment = stored_segments.get(aggregate.segment_key)
        if stored_segment is None or aggregate.view_to_purchase_rate is None:
            continue
        baseline = baselines.get(stored_segment.id)
        baseline_rate = baseline.view_to_purchase_rate if baseline is not None else None
        target_value = aggregate.target_view_to_purchase_rate
        actual_value = aggregate.view_to_purchase_rate
        target_triggered = actual_value < target_value
        baseline_drop_rate = calculate_baseline_drop_rate(actual_value, baseline_rate)
        baseline_triggered = (
            baseline_drop_rate is not None
            and baseline_drop_rate >= BASELINE_DROP_THRESHOLD
        )
        if not target_triggered and not baseline_triggered:
            continue

        expected_value = baseline_rate if baseline_triggered and baseline_rate is not None else target_value
        difference_value = expected_value - actual_value
        difference_rate = difference_value / expected_value if expected_value > 0 else None
        impact_score = difference_value * Decimal(aggregate.product_view_count)
        selected_step = select_lowest_non_null_funnel_step(aggregate)
        primary_drop_off_metric = selected_step[0] if selected_step is not None else None
        primary_drop_off_step = selected_step[3] if selected_step is not None else None
        primary_drop_off_rate = str(selected_step[4]) if selected_step is not None else None
        candidates.append(
            SegmentAnomalyCandidate(
                segment_id=stored_segment.id,
                metric_name="view_to_purchase_rate",
                actual_value=actual_value,
                expected_value=expected_value,
                target_value=target_value,
                difference_value=difference_value,
                difference_rate=difference_rate,
                severity=resolve_severity(impact_score),
                impact_score=impact_score,
                evidence_json={
                    "segment_key": aggregate.segment_key,
                    "target_triggered": target_triggered,
                    "baseline_triggered": baseline_triggered,
                    "baseline_view_to_purchase_rate": str(baseline_rate)
                    if baseline_rate is not None
                    else None,
                    "baseline_drop_rate": str(baseline_drop_rate)
                    if baseline_drop_rate is not None
                    else None,
                    "product_view_count": aggregate.product_view_count,
                    "purchase_count": aggregate.purchase_count,
                    "hypothesis": build_anomaly_hypothesis(
                        target_triggered=target_triggered,
                        baseline_triggered=baseline_triggered,
                        selected_step=selected_step,
                    ),
                    "primary_drop_off_step": primary_drop_off_step,
                    "primary_drop_off_metric": primary_drop_off_metric,
                    "primary_drop_off_rate": primary_drop_off_rate,
                },
            )
        )
    return candidates


def build_root_cause_candidates(
    aggregates: list[SegmentAggregate],
    stored_segments: Mapping[str, StoredSegment],
    stored_anomalies: list[StoredAnomaly],
) -> list[RootCauseCandidate]:
    anomaly_by_segment_id = {
        anomaly.segment_id: anomaly
        for anomaly in stored_anomalies
    }
    root_causes: list[RootCauseCandidate] = []
    for aggregate in aggregates:
        stored_segment = stored_segments.get(aggregate.segment_key)
        if stored_segment is None:
            continue
        anomaly = anomaly_by_segment_id.get(stored_segment.id)
        if anomaly is None:
            continue
        selected = select_lowest_non_null_funnel_step(aggregate)
        if selected is None:
            continue
        metric_name, cause_key, title, funnel_step, rate = selected
        root_causes.append(
            RootCauseCandidate(
                anomaly_id=anomaly.id,
                cause_type="funnel_step_drop",
                cause_key=cause_key,
                title=title,
                description=f"{funnel_step} 구간의 전환율이 가장 낮습니다.",
                confidence_score=Decimal("0.7"),
                impact_score=Decimal("1") - rate,
                rank_no=1,
                evidence_json={
                    "metric_name": metric_name,
                    "funnel_step": funnel_step,
                    "rate": str(rate),
                    "segment_key": aggregate.segment_key,
                },
            )
        )
    return root_causes


def calculate_baseline_drop_rate(
    actual_value: Decimal,
    baseline_value: Decimal | None,
) -> Decimal | None:
    if baseline_value is None or baseline_value <= 0:
        return None
    return (baseline_value - actual_value) / baseline_value


def resolve_severity(impact_score: Decimal) -> str:
    if impact_score >= Decimal("100"):
        return "critical"
    if impact_score >= Decimal("50"):
        return "high"
    if impact_score >= Decimal("10"):
        return "medium"
    return "low"


def build_anomaly_hypothesis(
    target_triggered: bool,
    baseline_triggered: bool,
    selected_step: FunnelStepSelection | None,
) -> str:
    sentences: list[str] = []
    if target_triggered and baseline_triggered:
        sentences.append("실제 구매 전환율이 목표 전환율보다 낮고, 기준 전환율보다 낮아졌습니다.")
    elif target_triggered:
        sentences.append("실제 구매 전환율이 목표 전환율보다 낮습니다.")
    elif baseline_triggered:
        sentences.append("실제 구매 전환율이 기준 전환율보다 낮아졌습니다.")
    if selected_step is not None:
        funnel_step = selected_step[3]
        label = FUNNEL_STEP_LABELS.get(funnel_step, "주요 퍼널 전환")
        sentences.append(f"{label} 구간에서 가장 큰 이탈이 발생했습니다.")
    if not sentences:
        return "구매 전환율 이상 징후가 감지되었습니다."
    return " ".join(sentences)


def select_lowest_non_null_funnel_step(
    aggregate: SegmentAggregate,
) -> FunnelStepSelection | None:
    rates: list[FunnelStepSelection] = []
    for metric_name, cause_key, title, funnel_step in ROOT_CAUSE_STEPS:
        rate = getattr(aggregate, metric_name)
        if rate is not None:
            rates.append((metric_name, cause_key, title, funnel_step, rate))
    if not rates:
        return None
    return min(rates, key=lambda item: item[4])
