from __future__ import annotations

from decimal import Decimal

from app.analysis.anomaly import (
    build_root_cause_candidates,
    detect_segment_anomalies,
    select_lowest_non_null_funnel_step,
)
from app.analysis.models import BaselineMetrics, SegmentAggregate, StoredAnomaly, StoredSegment
from tests.test_analysis_service import segment_aggregate

SEVERITY_WORDS = {"low", "medium", "high", "critical"}


def test_detect_segment_anomalies_uses_target_when_baseline_missing() -> None:
    aggregate = SegmentAggregate(
        **{
            **segment_aggregate().__dict__,
            "purchase_count": 1,
            "view_to_purchase_rate": Decimal("0.01"),
            "cvr": Decimal("0.01"),
        }
    )

    anomalies = detect_segment_anomalies(
        aggregates=[aggregate],
        stored_segments={aggregate.segment_key: StoredSegment(id=10, segment_key=aggregate.segment_key)},
        baselines={},
    )

    assert len(anomalies) == 1
    assert anomalies[0].expected_value == Decimal("0.05")
    assert anomalies[0].evidence_json["baseline_triggered"] is False
    evidence = anomalies[0].evidence_json
    hypothesis = evidence["hypothesis"]
    assert isinstance(hypothesis, str)
    assert hypothesis
    assert "목표 전환율" in hypothesis
    assert "상품 조회 후 장바구니 추가 구간" in hypothesis
    assert "product_view_to_add_to_cart" not in hypothesis
    assert not any(severity in hypothesis for severity in SEVERITY_WORDS)
    assert evidence["primary_drop_off_step"] == "product_view_to_add_to_cart"
    assert evidence["primary_drop_off_metric"] == "view_to_cart_rate"
    assert evidence["primary_drop_off_rate"] == "0.2"


def test_detect_segment_anomalies_uses_seven_day_baseline_when_available() -> None:
    aggregate = SegmentAggregate(
        **{
            **segment_aggregate().__dict__,
            "purchase_count": 4,
            "view_to_purchase_rate": Decimal("0.04"),
            "cvr": Decimal("0.04"),
        }
    )

    anomalies = detect_segment_anomalies(
        aggregates=[aggregate],
        stored_segments={aggregate.segment_key: StoredSegment(id=10, segment_key=aggregate.segment_key)},
        baselines={10: BaselineMetrics(segment_id=10, view_to_purchase_rate=Decimal("0.06"))},
    )

    assert len(anomalies) == 1
    assert anomalies[0].expected_value == Decimal("0.06")
    assert anomalies[0].evidence_json["baseline_triggered"] is True
    hypothesis = anomalies[0].evidence_json["hypothesis"]
    assert isinstance(hypothesis, str)
    assert "기준 전환율" in hypothesis
    assert not any(severity in hypothesis for severity in SEVERITY_WORDS)


def test_detect_segment_anomalies_keeps_hypothesis_when_no_funnel_step_exists() -> None:
    aggregate = SegmentAggregate(
        **{
            **segment_aggregate().__dict__,
            "purchase_count": 1,
            "view_to_cart_rate": None,
            "cart_to_checkout_rate": None,
            "checkout_to_purchase_rate": None,
            "view_to_purchase_rate": Decimal("0.01"),
            "cvr": Decimal("0.01"),
        }
    )

    anomalies = detect_segment_anomalies(
        aggregates=[aggregate],
        stored_segments={aggregate.segment_key: StoredSegment(id=10, segment_key=aggregate.segment_key)},
        baselines={},
    )

    assert len(anomalies) == 1
    evidence = anomalies[0].evidence_json
    hypothesis = evidence["hypothesis"]
    assert isinstance(hypothesis, str)
    assert hypothesis
    assert not any(severity in hypothesis for severity in SEVERITY_WORDS)
    assert evidence["primary_drop_off_step"] is None
    assert evidence["primary_drop_off_metric"] is None
    assert evidence["primary_drop_off_rate"] is None


def test_select_lowest_non_null_funnel_step_excludes_null_rates() -> None:
    aggregate = SegmentAggregate(
        **{
            **segment_aggregate().__dict__,
            "view_to_cart_rate": None,
            "cart_to_checkout_rate": Decimal("0.3"),
            "checkout_to_purchase_rate": Decimal("0.2"),
        }
    )

    selected = select_lowest_non_null_funnel_step(aggregate)

    assert selected is not None
    assert selected[1] == "checkout_to_purchase"


def test_build_root_cause_candidates_uses_stored_anomaly_id() -> None:
    aggregate = SegmentAggregate(
        **{
            **segment_aggregate().__dict__,
            "view_to_cart_rate": Decimal("0.1"),
            "cart_to_checkout_rate": Decimal("0.3"),
            "checkout_to_purchase_rate": Decimal("0.2"),
        }
    )

    root_causes = build_root_cause_candidates(
        aggregates=[aggregate],
        stored_segments={aggregate.segment_key: StoredSegment(id=10, segment_key=aggregate.segment_key)},
        stored_anomalies=[StoredAnomaly(id=99, segment_id=10)],
    )

    assert len(root_causes) == 1
    assert root_causes[0].anomaly_id == 99
    assert root_causes[0].cause_key == "view_to_cart"
