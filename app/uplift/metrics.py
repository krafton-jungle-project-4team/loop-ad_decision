from __future__ import annotations

import math
from typing import Any, Sequence

from app.uplift.contracts import UpliftTrainingExample


def evaluate_uplift_predictions(
    examples: Sequence[UpliftTrainingExample],
    cate_scores: Sequence[float],
    *,
    top_fractions: Sequence[float] = (0.1, 0.2, 0.3),
) -> dict[str, Any]:
    if len(examples) != len(cate_scores):
        raise ValueError("examples and CATE scores must have the same length")
    if not examples:
        raise ValueError("uplift evaluation examples are required")
    treatment = [example for example in examples if example.treatment == 1]
    control = [example for example in examples if example.treatment == 0]
    if not treatment or not control:
        raise ValueError("uplift evaluation requires non-empty treatment and control")

    _validate_randomized_examples(examples)
    treatment_rate, control_rate = _ipw_arm_outcome_rates(examples)
    ordered = sorted(
        zip(examples, cate_scores, strict=True),
        key=lambda item: (-item[1], item[0].experiment_unit_id),
    )
    cumulative_gain = 0.0
    qini_curve: list[float] = []
    for example, _score in ordered:
        probability = float(example.treatment_probability)
        if example.treatment:
            cumulative_gain += 0.5 * example.outcome / probability
        else:
            cumulative_gain -= 0.5 * example.outcome / (1.0 - probability)
        qini_curve.append(cumulative_gain)
    final_gain = qini_curve[-1]
    auuc_baseline = sum(
        final_gain * ((index + 1) / len(qini_curve))
        for index in range(len(qini_curve))
    ) / len(qini_curve)
    qini = sum(
        gain - final_gain * ((index + 1) / len(qini_curve))
        for index, gain in enumerate(qini_curve)
    ) / len(qini_curve)
    auuc = sum(qini_curve) / len(qini_curve)
    predicted_ate = sum(float(score) for score in cate_scores) / len(cate_scores)
    uplift_at_top_k = {
        _fraction_label(fraction): _ipw_observed_uplift(
            [example for example, _score in ordered[: max(1, math.ceil(len(ordered) * fraction))]]
        )
        for fraction in top_fractions
    }
    return {
        "observation_count": len(examples),
        "treatment_count": len(treatment),
        "control_count": len(control),
        "treatment_outcome_rate": treatment_rate,
        "control_outcome_rate": control_rate,
        "ate": treatment_rate - control_rate,
        "predicted_ate": predicted_ate,
        "predicted_observed_ate_absolute_error": abs(
            predicted_ate - (treatment_rate - control_rate)
        ),
        "auuc": auuc,
        "auuc_baseline": auuc_baseline,
        "qini": qini,
        "negative_cate_ratio": sum(float(score) < 0 for score in cate_scores)
        / len(cate_scores),
        "uplift_at_top_k": uplift_at_top_k,
        "evaluation_method": "individual_propensity_ipw.v1",
    }


def _ipw_observed_uplift(
    examples: Sequence[UpliftTrainingExample],
) -> float | None:
    if not any(example.treatment == 1 for example in examples) or not any(
        example.treatment == 0 for example in examples
    ):
        return None
    treatment_rate, control_rate = _ipw_arm_outcome_rates(examples)
    return treatment_rate - control_rate


def _ipw_arm_outcome_rates(
    examples: Sequence[UpliftTrainingExample],
) -> tuple[float, float]:
    treatment_outcome = 0.0
    treatment_weight = 0.0
    control_outcome = 0.0
    control_weight = 0.0
    for example in examples:
        probability = float(example.treatment_probability)
        if example.treatment:
            weight = 1.0 / probability
            treatment_outcome += weight * example.outcome
            treatment_weight += weight
        else:
            weight = 1.0 / (1.0 - probability)
            control_outcome += weight * example.outcome
            control_weight += weight
    if treatment_weight == 0.0 or control_weight == 0.0:
        raise ValueError("uplift evaluation requires both randomized arms")
    return (
        treatment_outcome / treatment_weight,
        control_outcome / control_weight,
    )


def _validate_randomized_examples(
    examples: Sequence[UpliftTrainingExample],
) -> None:
    for example in examples:
        probability = float(example.treatment_probability)
        if example.treatment not in (0, 1):
            raise ValueError("uplift treatment must be zero or one")
        if not 0.0 < probability < 1.0:
            raise ValueError(
                "uplift evaluation requires individual randomized propensities"
            )


def _fraction_label(fraction: float) -> str:
    if not 0 < fraction <= 1:
        raise ValueError("top fraction must be between zero and one")
    return f"top_{int(round(fraction * 100))}_percent"
