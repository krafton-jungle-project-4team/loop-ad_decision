from __future__ import annotations

from dataclasses import replace

import pytest

from app.analysis.segment_performance import (
    CANDIDATE_TYPE_SUPPORT_CONTRACT_VERSION,
    CalibrationTrainingExample,
    ContextualBookingHeuristicPredictor,
    LogisticSegmentPerformanceModel,
    PREDICTION_PRIOR_USER_COUNT,
    PREDICTION_POLICY_VERSION,
    SegmentPerformanceFeatures,
    UnsupportedCandidateTypeError,
    build_segment_performance_predictor,
    candidate_type_prediction_support,
    fit_logistic_segment_performance_model,
    predict_segment_performance,
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
    assert model.metadata()["candidate_type_support_contract_version"] == (
        CANDIDATE_TYPE_SUPPORT_CONTRACT_VERSION
    )
    assert model.metadata()["training_candidate_type_example_counts"] == {
        "intent_matched": 2,
        "target_destination_affinity": 0,
        "funnel_recovery": 0,
        "benefit_value_seeker": 0,
        "promotion_responsive": 0,
        "general_destination_explorer": 2,
    }
    assert model.metadata()[
        "training_candidate_type_user_observation_counts"
    ] == {
        "intent_matched": 200,
        "target_destination_affinity": 0,
        "funnel_recovery": 0,
        "benefit_value_seeker": 0,
        "promotion_responsive": 0,
        "general_destination_explorer": 200,
    }


def test_calibrated_model_rejects_candidate_type_without_training_examples() -> None:
    supported_features = features(
        destination_match_user_rate=1.0,
        destination_match_event_rate=0.7,
    )
    model = fit_logistic_segment_performance_model(
        [
            CalibrationTrainingExample(supported_features, 20, 100),
            CalibrationTrainingExample(supported_features, 10, 100),
        ]
    )
    unsupported_features = replace(
        supported_features,
        candidate_type="promotion_responsive",
    )

    support = candidate_type_prediction_support(
        model,
        goal_metric="booking_conversion_rate",
        candidate_type="promotion_responsive",
    )

    assert support.supported is False
    assert support.training_example_count == 0
    assert support.reason == "candidate_type_not_observed_in_model_training"
    with pytest.raises(UnsupportedCandidateTypeError, match="no training examples"):
        model.predict(unsupported_features)


def test_non_booking_metric_does_not_use_booking_candidate_support_contract() -> None:
    example_features = features(
        destination_match_user_rate=1.0,
        destination_match_event_rate=0.7,
    )
    model = fit_logistic_segment_performance_model(
        [
            CalibrationTrainingExample(example_features, 20, 100),
            CalibrationTrainingExample(example_features, 10, 100),
        ]
    )

    support = candidate_type_prediction_support(
        model,
        goal_metric="inflow_rate",
        candidate_type="promotion_responsive",
    )

    assert support.supported is True
    assert support.training_example_count is None


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
    assert model.metadata()["training_candidate_type_example_counts"] == {
        "intent_matched": 24,
        "target_destination_affinity": 24,
        "funnel_recovery": 24,
        "benefit_value_seeker": 24,
        "promotion_responsive": 0,
        "general_destination_explorer": 0,
    }
    assert model.supports_candidate_type("intent_matched") is True
    assert model.supports_candidate_type("promotion_responsive") is False


def test_serving_prediction_limits_extreme_features_and_shrinks_small_sample() -> None:
    model = build_segment_performance_predictor()
    assert isinstance(model, LogisticSegmentPerformanceModel)
    extreme_features = replace(
        features(
            candidate_type="target_destination_affinity",
            destination_match_user_rate=1.0,
            destination_match_event_rate=0.679,
        ),
        eligible_destination_match_user_rate=4 / 312,
        hotel_detail_view_user_rate=1.0,
        booking_start_user_rate=1.0,
        booking_complete_user_rate=1.0,
        funnel_recovery_user_rate=0.75,
        benefit_user_rate=1.0,
        promotion_response_user_rate=1.0,
    )

    prediction = predict_segment_performance(
        model,
        extreme_features,
        sample_size=4,
    )

    assert prediction.raw_model_value > 0.5
    assert prediction.distribution_guarded_value < prediction.raw_model_value
    assert prediction.value < prediction.distribution_guarded_value
    assert prediction.training_baseline_rate is not None
    assert prediction.sample_weight == pytest.approx(
        4 / (4 + PREDICTION_PRIOR_USER_COUNT)
    )
    assert prediction.value == pytest.approx(
        prediction.sample_weight * prediction.distribution_guarded_value
        + (1.0 - prediction.sample_weight)
        * prediction.training_baseline_rate
    )
    assert "destination_match_event_rate" in (
        prediction.influential_out_of_distribution_features
    )
    metadata = prediction.metadata()["prediction_adjustment"]
    assert metadata["policy_version"] == PREDICTION_POLICY_VERSION
    assert metadata["candidate_sample_size"] == 4
    assert metadata["out_of_distribution_feature_count"] > 0


def test_serving_prediction_preserves_non_logistic_predictor_result() -> None:
    predictor = ContextualBookingHeuristicPredictor()
    example_features = features(
        destination_match_user_rate=1.0,
        destination_match_event_rate=0.7,
    )

    prediction = predict_segment_performance(
        predictor,
        example_features,
        sample_size=4,
    )

    assert prediction.value == pytest.approx(predictor.predict(example_features))
    assert prediction.raw_model_value == prediction.value
    assert prediction.policy_version is None
    assert prediction.metadata()["prediction_adjustment"]["applied"] is False
