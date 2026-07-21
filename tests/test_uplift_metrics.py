from __future__ import annotations

import math

from app.uplift.contracts import UpliftTrainingExample
from app.uplift.metrics import evaluate_uplift_predictions


def test_propensity_metrics_match_legacy_results_for_fifty_fifty_design() -> None:
    examples = [
        example("u1", treatment=1, outcome=1, probability=0.5),
        example("u2", treatment=0, outcome=0, probability=0.5),
        example("u3", treatment=1, outcome=0, probability=0.5),
        example("u4", treatment=0, outcome=1, probability=0.5),
        example("u5", treatment=1, outcome=1, probability=0.5),
        example("u6", treatment=0, outcome=0, probability=0.5),
        example("u7", treatment=1, outcome=0, probability=0.5),
        example("u8", treatment=0, outcome=0, probability=0.5),
    ]
    scores = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    metrics = evaluate_uplift_predictions(
        examples,
        scores,
        top_fractions=(0.5, 1.0),
    )
    legacy = legacy_fifty_fifty_metrics(examples, scores)

    assert metrics["ate"] == legacy["ate"]
    assert metrics["auuc"] == legacy["auuc"]
    assert metrics["qini"] == legacy["qini"]
    assert metrics["uplift_at_top_k"]["top_50_percent"] == (
        legacy["top_50_percent"]
    )


def test_mixed_assignment_propensities_recover_known_average_effect() -> None:
    examples: list[UpliftTrainingExample] = []
    experiment_specs = (
        ("exp_20", 0.2, 0.1, 0.2),
        ("exp_50", 0.5, 0.3, 0.4),
        ("exp_80", 0.8, 0.5, 0.6),
    )
    for experiment_id, propensity, control_rate, treatment_rate in experiment_specs:
        treatment_count = int(1000 * propensity)
        control_count = 1000 - treatment_count
        treatment_outcomes = int(treatment_count * treatment_rate)
        control_outcomes = int(control_count * control_rate)
        for index in range(treatment_count):
            examples.append(
                example(
                    f"{experiment_id}_t_{index:04d}",
                    treatment=1,
                    outcome=int(index < treatment_outcomes),
                    probability=propensity,
                    experiment_id=experiment_id,
                )
            )
        for index in range(control_count):
            examples.append(
                example(
                    f"{experiment_id}_c_{index:04d}",
                    treatment=0,
                    outcome=int(index < control_outcomes),
                    probability=propensity,
                    experiment_id=experiment_id,
                )
            )
    scores = [0.1] * len(examples)

    metrics = evaluate_uplift_predictions(
        examples,
        scores,
        top_fractions=(1.0,),
    )
    raw_treatment_rate = sum(
        item.outcome for item in examples if item.treatment
    ) / sum(item.treatment for item in examples)
    raw_control_rate = sum(
        item.outcome for item in examples if not item.treatment
    ) / sum(not item.treatment for item in examples)

    assert raw_treatment_rate - raw_control_rate > 0.2
    assert math.isclose(metrics["ate"], 0.1, abs_tol=1e-12)
    assert math.isclose(
        metrics["uplift_at_top_k"]["top_100_percent"],
        0.1,
        abs_tol=1e-12,
    )
    assert math.isfinite(metrics["qini"])
    assert math.isfinite(metrics["auuc"])
    assert metrics["evaluation_method"] == "individual_propensity_ipw.v1"


def example(
    user_id: str,
    *,
    treatment: int,
    outcome: int,
    probability: float,
    experiment_id: str = "experiment",
) -> UpliftTrainingExample:
    return UpliftTrainingExample(
        experiment_unit_id=user_id,
        project_id="project",
        promotion_run_id="run",
        ad_experiment_id=experiment_id,
        segment_id="segment",
        user_id=user_id,
        audience_snapshot_id="snapshot",
        vector_generation_id="generation",
        features=(0.0,),
        treatment=treatment,
        outcome=outcome,
        treatment_probability=probability,
    )


def legacy_fifty_fifty_metrics(examples, scores):
    treatment = [item for item in examples if item.treatment]
    control = [item for item in examples if not item.treatment]
    ate = sum(item.outcome for item in treatment) / len(treatment) - sum(
        item.outcome for item in control
    ) / len(control)
    ordered = sorted(
        zip(examples, scores, strict=True),
        key=lambda item: (-item[1], item[0].experiment_unit_id),
    )
    treatment_outcomes = 0
    control_outcomes = 0
    curve = []
    for item, _score in ordered:
        if item.treatment:
            treatment_outcomes += item.outcome
        else:
            control_outcomes += item.outcome
        curve.append(treatment_outcomes - control_outcomes)
    final_gain = curve[-1]
    qini = sum(
        gain - final_gain * ((index + 1) / len(curve))
        for index, gain in enumerate(curve)
    ) / len(curve)
    top_half = [item for item, _score in ordered[: len(ordered) // 2]]
    top_treatment = [item.outcome for item in top_half if item.treatment]
    top_control = [item.outcome for item in top_half if not item.treatment]
    return {
        "ate": ate,
        "auuc": sum(curve) / len(curve),
        "qini": qini,
        "top_50_percent": (
            sum(top_treatment) / len(top_treatment)
            - sum(top_control) / len(top_control)
        ),
    }
