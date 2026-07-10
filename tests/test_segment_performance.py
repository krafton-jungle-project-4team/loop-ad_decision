from __future__ import annotations

import pytest

from app.analysis.segment_performance import (
    CalibrationTrainingExample,
    ContextualBookingHeuristicPredictor,
    LogisticSegmentPerformanceModel,
    SegmentPerformanceFeatures,
    build_segment_performance_predictor,
    fit_logistic_segment_performance_model,
    write_segment_performance_model,
)


def features(
    *,
    candidate_type: str = "intent_matched",
    destination_match_user_rate: float,
    destination_match_event_rate: float,
    promotion_condition_match: float = 0.8,
) -> SegmentPerformanceFeatures:
    return SegmentPerformanceFeatures(
        candidate_type=candidate_type,
        promotion_condition_match=promotion_condition_match,
        destination_context_required=True,
        destination_match_user_rate=destination_match_user_rate,
        destination_match_event_rate=destination_match_event_rate,
        eligible_destination_match_user_rate=0.1,
        hotel_detail_view_user_rate=0.7,
        booking_start_user_rate=0.2,
        booking_complete_user_rate=0.1,
        funnel_recovery_user_rate=0.1,
        benefit_user_rate=0.2,
        promotion_response_user_rate=0.0,
        sample_reliability=1.0,
    )


def test_contextual_heuristic_penalizes_missing_destination_context() -> None:
    predictor = ContextualBookingHeuristicPredictor()

    matched = predictor.predict(
        features(
            destination_match_user_rate=1.0,
            destination_match_event_rate=0.8,
        )
    )
    unmatched = predictor.predict(
        features(
            destination_match_user_rate=0.0,
            destination_match_event_rate=0.0,
        )
    )

    assert matched > unmatched
    assert unmatched < 0.02


def test_logistic_calibration_learns_future_contextual_outcomes() -> None:
    low = features(
        candidate_type="general_destination_explorer",
        destination_match_user_rate=0.0,
        destination_match_event_rate=0.0,
        promotion_condition_match=0.1,
    )
    high = features(
        destination_match_user_rate=1.0,
        destination_match_event_rate=0.9,
    )
    model = fit_logistic_segment_performance_model(
        [
            CalibrationTrainingExample(low, success_count=1, sample_size=100),
            CalibrationTrainingExample(low, success_count=2, sample_size=100),
            CalibrationTrainingExample(high, success_count=20, sample_size=100),
            CalibrationTrainingExample(high, success_count=25, sample_size=100),
        ],
        training_metadata={"training_period": "2013"},
    )

    assert model.predict(high) > model.predict(low)
    assert model.metadata()["training_period"] == "2013"
    assert model.metadata()["optimizer"]["selection_basis"] == (
        "caller_configured"
    )


def test_logistic_calibration_json_round_trip_preserves_prediction() -> None:
    example_features = features(
        destination_match_user_rate=1.0,
        destination_match_event_rate=0.7,
    )
    model = fit_logistic_segment_performance_model(
        [
            CalibrationTrainingExample(
                example_features,
                success_count=20,
                sample_size=100,
            ),
            CalibrationTrainingExample(
                features(
                    destination_match_user_rate=0.0,
                    destination_match_event_rate=0.0,
                ),
                success_count=1,
                sample_size=100,
            ),
        ]
    )

    restored = LogisticSegmentPerformanceModel.from_json(model.to_json())

    assert restored.predict(example_features) == pytest.approx(
        model.predict(example_features)
    )


def test_configured_calibration_model_is_loaded_from_json(tmp_path) -> None:
    example_features = features(
        destination_match_user_rate=1.0,
        destination_match_event_rate=0.7,
    )
    model = fit_logistic_segment_performance_model(
        [
            CalibrationTrainingExample(example_features, 20, 100),
            CalibrationTrainingExample(
                features(
                    destination_match_user_rate=0.0,
                    destination_match_event_rate=0.0,
                ),
                1,
                100,
            ),
        ]
    )
    model_path = tmp_path / "model.json"
    write_segment_performance_model(model, model_path)

    loaded = build_segment_performance_predictor(model_path)

    assert loaded.predict(example_features) == pytest.approx(
        model.predict(example_features)
    )


def test_configured_missing_calibration_model_fails_fast(tmp_path) -> None:
    with pytest.raises(ValueError, match="model not found"):
        build_segment_performance_predictor(tmp_path / "missing.json")


def test_bundled_model_uses_only_2013_training_outcomes() -> None:
    model = build_segment_performance_predictor()

    assert model.calibration_status == "calibrated"
    assert model.version == "dec.contextual-booking-calibration.v2"
    assert model.metadata()["training_end_cutoff"].startswith("2013-")
    assert model.metadata()["candidate_training_scope"] == (
        "all_eligible_candidate_types"
    )
    assert model.metadata()["optimizer"]["selection_basis"] == (
        "2014_development_validation"
    )
