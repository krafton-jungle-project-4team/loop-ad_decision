from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from app.uplift.contracts import UpliftTrainingExample
from app.uplift.model import UpliftModelMetadata


VALIDATION_POLICY_PATH = Path(__file__).with_name("validation_policy.v1.json")


@dataclass(frozen=True, slots=True)
class CateConfidenceInterval:
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


def experiment_cluster_bootstrap_cate_ci(
    examples: Sequence[UpliftTrainingExample],
    cate_scores: Sequence[float],
    *,
    iterations: int = 1000,
    seed: int = 20260721,
) -> CateConfidenceInterval:
    if len(examples) != len(cate_scores):
        raise ValueError("examples and CATE scores must have the same length")
    if iterations < 1:
        raise ValueError("bootstrap iterations must be positive")
    by_experiment: dict[str, list[float]] = defaultdict(list)
    for example, score in zip(examples, cate_scores, strict=True):
        by_experiment[example.ad_experiment_id].append(float(score))
    experiment_ids = sorted(by_experiment)
    if len(experiment_ids) < 2:
        return CateConfidenceInterval(
            lower=None,
            upper=None,
            method="experiment_cluster_bootstrap.v1",
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
    return CateConfidenceInterval(
        lower=_percentile(estimates, 0.025),
        upper=_percentile(estimates, 0.975),
        method="experiment_cluster_bootstrap.v1",
        reference_only=False,
    )


def collecting_data_metadata(
    *,
    model_version: str,
    dataset: str = "loopad_randomized_experiments",
) -> UpliftModelMetadata:
    policy = load_validation_policy()
    return UpliftModelMetadata(
        model_lifecycle_status="collecting_data",
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
        model_lifecycle_status="candidate",
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

