from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
import math
from typing import Sequence

from app.uplift.contracts import UpliftTrainingExample
from app.uplift.dataset import SPLIT_POLICY_VERSION


class UpliftDatasetSplitUnavailable(ValueError):
    code = "uplift_train_test_split_unavailable"


@dataclass(frozen=True, slots=True)
class UpliftExperimentSplit:
    train_examples: tuple[UpliftTrainingExample, ...]
    test_examples: tuple[UpliftTrainingExample, ...]
    train_experiment_ids: tuple[str, ...]
    test_experiment_ids: tuple[str, ...]
    split_policy_version: str = SPLIT_POLICY_VERSION


def split_by_experiment_end_time(
    examples: Sequence[UpliftTrainingExample],
    *,
    test_fraction: float = 0.2,
) -> UpliftExperimentSplit:
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between zero and one")
    by_experiment: dict[str, list[UpliftTrainingExample]] = defaultdict(list)
    for example in examples:
        by_experiment[example.ad_experiment_id].append(example)
    if len(by_experiment) < 2:
        raise UpliftDatasetSplitUnavailable(
            "at least two completed randomized experiments are required"
        )

    experiment_end_times: dict[str, datetime] = {}
    for experiment_id, experiment_examples in by_experiment.items():
        arms = {example.treatment for example in experiment_examples}
        if arms != {0, 1}:
            raise UpliftDatasetSplitUnavailable(
                f"experiment has an incomplete randomized arm: {experiment_id}"
            )
        end_times = [
            example.outcome_window_end
            for example in experiment_examples
            if example.outcome_window_end is not None
        ]
        if len(end_times) != len(experiment_examples):
            raise UpliftDatasetSplitUnavailable(
                f"experiment outcome end time is missing: {experiment_id}"
            )
        experiment_end_times[experiment_id] = max(end_times)

    ordered_experiments = sorted(
        by_experiment,
        key=lambda experiment_id: (
            experiment_end_times[experiment_id],
            experiment_id,
        ),
    )
    test_count = min(
        max(math.ceil(len(ordered_experiments) * test_fraction), 1),
        len(ordered_experiments) - 1,
    )
    train_ids = tuple(ordered_experiments[:-test_count])
    test_ids = tuple(ordered_experiments[-test_count:])
    train_id_set = set(train_ids)
    test_id_set = set(test_ids)
    train = tuple(
        example
        for example in examples
        if example.ad_experiment_id in train_id_set
    )
    test = tuple(
        example
        for example in examples
        if example.ad_experiment_id in test_id_set
    )
    _require_both_arms(train, partition="train")
    _require_both_arms(test, partition="test")
    return UpliftExperimentSplit(
        train_examples=train,
        test_examples=test,
        train_experiment_ids=train_ids,
        test_experiment_ids=test_ids,
    )


def _require_both_arms(
    examples: Sequence[UpliftTrainingExample],
    *,
    partition: str,
) -> None:
    if {example.treatment for example in examples} != {0, 1}:
        raise UpliftDatasetSplitUnavailable(
            f"{partition} partition requires treatment and control"
        )
