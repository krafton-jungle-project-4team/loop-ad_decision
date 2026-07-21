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

    treatment_rate = sum(example.outcome for example in treatment) / len(treatment)
    control_rate = sum(example.outcome for example in control) / len(control)
    ordered = sorted(
        zip(examples, cate_scores, strict=True),
        key=lambda item: (-item[1], item[0].experiment_unit_id),
    )
    treatment_control_ratio = len(treatment) / len(control)
    cumulative_treatment_outcomes = 0
    cumulative_control_outcomes = 0
    qini_curve: list[float] = []
    for example, _score in ordered:
        if example.treatment:
            cumulative_treatment_outcomes += example.outcome
        else:
            cumulative_control_outcomes += example.outcome
        qini_curve.append(
            cumulative_treatment_outcomes
            - cumulative_control_outcomes * treatment_control_ratio
        )
    final_gain = qini_curve[-1]
    qini = sum(
        gain - final_gain * ((index + 1) / len(qini_curve))
        for index, gain in enumerate(qini_curve)
    ) / len(qini_curve)
    auuc = sum(qini_curve) / len(qini_curve)
    uplift_at_top_k = {
        _fraction_label(fraction): _observed_uplift(
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
        "auuc": auuc,
        "qini": qini,
        "uplift_at_top_k": uplift_at_top_k,
    }


def _observed_uplift(examples: Sequence[UpliftTrainingExample]) -> float | None:
    treatment = [example.outcome for example in examples if example.treatment == 1]
    control = [example.outcome for example in examples if example.treatment == 0]
    if not treatment or not control:
        return None
    return sum(treatment) / len(treatment) - sum(control) / len(control)


def _fraction_label(fraction: float) -> str:
    if not 0 < fraction <= 1:
        raise ValueError("top fraction must be between zero and one")
    return f"top_{int(round(fraction * 100))}_percent"

