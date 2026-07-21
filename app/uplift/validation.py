from __future__ import annotations

import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.uplift.contracts import UpliftTrainingExample
from app.uplift.model import (
    UpliftModelLifecycleStatus,
    UpliftModelMetadata,
)


VALIDATION_POLICY_PATH = Path(__file__).with_name("validation_policy.v1.json")


@dataclass(frozen=True, slots=True)
class PredictedCateClusterVariabilityInterval:
    lower: float | None
    upper: float | None
    method: str
    reference_only: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "method": self.method,
            "reference_only": self.reference_only,
        }


def load_validation_policy() -> Mapping[str, Any]:
    return json.loads(VALIDATION_POLICY_PATH.read_text(encoding="utf-8"))


def evaluate_validation_policy(
    *,
    metrics: Mapping[str, Any],
    completed_experiment_count: int,
    treatment_observation_count: int,
    control_observation_count: int,
    positive_treatment_outcome_count: int,
    positive_control_outcome_count: int,
    policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved = dict(policy or load_validation_policy())
    failed_rules: list[str] = []
    count_rules = {
        "minimum_completed_experiments": completed_experiment_count,
        "minimum_treatment_observations": treatment_observation_count,
        "minimum_control_observations": control_observation_count,
    }
    for rule, actual in count_rules.items():
        if actual < int(resolved[rule]):
            failed_rules.append(rule)
    minimum_positive = int(resolved["minimum_positive_outcomes_per_arm"])
    if positive_treatment_outcome_count < minimum_positive:
        failed_rules.append("minimum_positive_treatment_outcomes")
    if positive_control_outcome_count < minimum_positive:
        failed_rules.append("minimum_positive_control_outcomes")

    required_metrics = resolved.get("required_metrics", {})
    if required_metrics.get("qini_above_zero") and not _finite_above(
        metrics.get("qini"),
        0.0,
    ):
        failed_rules.append("qini_above_zero")
    if required_metrics.get("auuc_above_baseline") and not _finite_above(
        metrics.get("auuc"),
        metrics.get("auuc_baseline"),
    ):
        failed_rules.append("auuc_above_baseline")
    return {
        "passed": not failed_rules,
        "failed_rules": failed_rules,
        "policy_version": str(resolved["validation_policy_version"]),
        "policy_status": str(resolved["policy_status"]),
        "statistical_power_derived": bool(
            resolved["statistical_power_derived"]
        ),
        "requires_manual_approval": bool(resolved["requires_manual_approval"]),
    }


def predicted_cate_cluster_variability_interval(
    examples: Sequence[UpliftTrainingExample],
    cate_scores: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int = 20260721,
) -> PredictedCateClusterVariabilityInterval:
    if len(examples) != len(cate_scores):
        raise ValueError("examples and CATE scores must have the same length")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    by_experiment: dict[str, list[float]] = defaultdict(list)
    for example, score in zip(examples, cate_scores, strict=True):
        by_experiment[example.ad_experiment_id].append(float(score))
    experiment_ids = sorted(by_experiment)
    if len(experiment_ids) < 2:
        return PredictedCateClusterVariabilityInterval(
            lower=None,
            upper=None,
            method="predicted_cate_cluster_variability_interval.v1",
            reference_only=True,
        )

    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(iterations):
        sampled_scores: list[float] = []
        for _cluster in experiment_ids:
            sampled_experiment = rng.choice(experiment_ids)
            sampled_scores.extend(by_experiment[sampled_experiment])
        estimates.append(sum(sampled_scores) / len(sampled_scores))
    estimates.sort()
    return PredictedCateClusterVariabilityInterval(
        lower=_percentile(estimates, 0.025),
        upper=_percentile(estimates, 0.975),
        method="predicted_cate_cluster_variability_interval.v1",
        reference_only=True,
    )


def collecting_data_metadata(
    *,
    model_version: str,
    dataset: str = "loopad_randomized_experiments",
) -> UpliftModelMetadata:
    policy = load_validation_policy()
    return UpliftModelMetadata(
        model_lifecycle_status=UpliftModelLifecycleStatus.COLLECTING_DATA,
        validation_scope="loopad_randomized_experiments",
        dataset=dataset,
        serving_eligible=False,
        model_version=model_version,
        validation_policy_version=str(policy["validation_policy_version"]),
    )


def external_validation_metadata(
    *,
    model_version: str,
    dataset: str,
) -> UpliftModelMetadata:
    return UpliftModelMetadata(
        model_lifecycle_status=UpliftModelLifecycleStatus.CANDIDATE,
        validation_scope="external_pipeline_validation",
        dataset=dataset,
        serving_eligible=False,
        model_version=model_version,
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    position = (len(values) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(values) - 1)
    fraction = position - lower_index
    return values[lower_index] * (1 - fraction) + values[upper_index] * fraction


def _finite_above(value: Any, baseline: Any) -> bool:
    try:
        numeric = float(value)
        numeric_baseline = float(baseline)
    except (TypeError, ValueError):
        return False
    return (
        math.isfinite(numeric)
        and math.isfinite(numeric_baseline)
        and numeric > numeric_baseline
    )
