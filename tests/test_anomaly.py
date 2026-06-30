from __future__ import annotations

from decimal import Decimal

from app.analysis.anomaly import (
    build_root_cause_candidates,
    derive_matching_weights,
    detect_segment_anomalies,
    select_lowest_non_null_funnel_step,
)
from app.analysis.models import BaselineMetrics, SegmentAggregate, StoredAnomaly, StoredSegment
from tests.test_analysis_service import segment_aggregate


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


def test_derive_matching_weights_boosts_defined_dimensions_from_impact() -> None:
    result = derive_matching_weights(
        {
            "primary_category": "fresh",
            "acquisition_channel": "kakao",
            "age_group": "30s",
            "gender": "male",
        },
        0.875,
    )

    matching = result["matching"]
    assert matching["dimension_weights"] == {
        "primary_category": 6,
        "acquisition_channel": 4,
        "device_type": 1,
        "age_group": 2,
        "gender": 2,
    }
    assert matching["min_score"] == 2
    assert matching["source"] == "anomaly_impact"


def test_derive_matching_weights_uses_base_weights_without_impact() -> None:
    expected_weights = {
        "primary_category": 3,
        "acquisition_channel": 2,
        "device_type": 1,
        "age_group": 1,
        "gender": 1,
    }

    none_result = derive_matching_weights({"primary_category": "fresh"}, None)
    zero_result = derive_matching_weights({"primary_category": "fresh"}, 0)

    assert none_result["matching"]["dimension_weights"] == expected_weights
    assert none_result["matching"]["min_score"] == 3
    assert zero_result["matching"]["dimension_weights"] == expected_weights
    assert zero_result["matching"]["min_score"] == 3


def test_derive_matching_weights_never_boosts_absent_dimensions() -> None:
    result = derive_matching_weights({"primary_category": "fresh"}, 0.875)

    weights = result["matching"]["dimension_weights"]
    assert weights["primary_category"] == 6
    assert weights["acquisition_channel"] == 2
    assert weights["device_type"] == 1
    assert weights["age_group"] == 1
    assert weights["gender"] == 1


def test_derive_matching_weights_uses_base_weights_without_rule_json() -> None:
    result = derive_matching_weights(None, 0.875)

    assert result["matching"]["dimension_weights"] == {
        "primary_category": 3,
        "acquisition_channel": 2,
        "device_type": 1,
        "age_group": 1,
        "gender": 1,
    }
