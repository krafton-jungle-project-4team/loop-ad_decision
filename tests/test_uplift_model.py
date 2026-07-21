from app.uplift.model import (
    UpliftModelMetadata,
    serving_cate_scores,
    signed_cate_summary,
)
from app.uplift.synthetic_validation import run_synthetic_validation
from app.uplift.validation import load_validation_policy


def test_synthetic_validation_recovers_negative_zero_positive_effect_order() -> None:
    report = run_synthetic_validation(sample_size=6000, seed=41)
    recovered = report["recovered_signed_cate"]

    assert recovered["negative"] < -0.04
    assert abs(recovered["zero"]) < 0.04
    assert recovered["positive"] > 0.04
    assert recovered["negative"] < recovered["zero"] < recovered["positive"]
    assert report["model_metadata"]["model_lifecycle_status"] == "collecting_data"
    assert report["model_metadata"]["serving_eligible"] is False


def test_signed_cate_summary_keeps_negative_effects() -> None:
    summary = signed_cate_summary([0.05, 0.03, -0.08])

    assert abs(summary["mean_cate"]) < 1e-12
    assert abs(summary["expected_incremental_bookings"]) < 1e-12
    assert summary["negative_cate_user_ratio"] == 1 / 3


def test_external_or_collecting_model_cannot_score_serving_candidates() -> None:
    class _Model:
        def predict_many(self, _examples):
            return [0.5]

    metadata = UpliftModelMetadata(
        model_lifecycle_status="candidate",
        validation_scope="external_pipeline_validation",
        dataset="criteo_uplift",
        serving_eligible=False,
        model_version="test",
    )

    assert serving_cate_scores(
        model=_Model(),
        metadata=metadata,
        examples=[],
    ) is None


def test_validation_policy_is_explicitly_provisional() -> None:
    policy = load_validation_policy()

    assert policy["minimum_completed_experiments"] == 20
    assert policy["policy_status"] == "provisional_safety_guard"
    assert policy["statistical_power_derived"] is False
    assert policy["requires_manual_approval"] is True
