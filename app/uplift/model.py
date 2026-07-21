from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence

from app.uplift.contracts import UpliftTrainingExample


@dataclass(frozen=True, slots=True)
class TransformedOutcomeRidgeModel:
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    coefficients: tuple[float, ...]
    intercept: float
    model_version: str = "transformed-outcome-ridge.v1"

    def predict_cate(self, features: Sequence[float]) -> float:
        if len(features) != len(self.coefficients):
            raise ValueError("feature dimension does not match the uplift model")
        return self.intercept + sum(
            coefficient * ((float(value) - mean) / scale)
            for value, mean, scale, coefficient in zip(
                features,
                self.feature_means,
                self.feature_scales,
                self.coefficients,
                strict=True,
            )
        )

    def predict_many(
        self,
        examples: Sequence[UpliftTrainingExample],
    ) -> list[float]:
        return [self.predict_cate(example.features) for example in examples]


class UpliftModelLifecycleStatus(StrEnum):
    COLLECTING_DATA = "collecting_data"
    CANDIDATE = "candidate"
    VALIDATED = "validated"
    ACTIVE = "active"
    REJECTED = "rejected"
    RETIRED = "retired"


@dataclass(frozen=True, slots=True)
class UpliftModelMetadata:
    model_lifecycle_status: UpliftModelLifecycleStatus
    validation_scope: str
    dataset: str
    serving_eligible: bool
    model_version: str
    validation_policy_version: str | None = None

    def __post_init__(self) -> None:
        status = self.model_lifecycle_status
        if not isinstance(status, UpliftModelLifecycleStatus):
            try:
                status = UpliftModelLifecycleStatus(str(status))
            except ValueError as exc:
                raise ValueError("unsupported uplift model lifecycle status") from exc
            object.__setattr__(self, "model_lifecycle_status", status)
        if status is UpliftModelLifecycleStatus.ACTIVE or self.serving_eligible:
            raise ValueError(
                "active uplift metadata requires the future persistent model registry"
            )

    def to_json(self) -> dict[str, Any]:
        return {
            "model_lifecycle_status": self.model_lifecycle_status,
            "validation_scope": self.validation_scope,
            "dataset": self.dataset,
            "serving_eligible": self.serving_eligible,
            "model_version": self.model_version,
            "validation_policy_version": self.validation_policy_version,
        }


def fit_transformed_outcome_ridge(
    examples: Sequence[UpliftTrainingExample],
    *,
    ridge_strength: float = 1.0,
) -> TransformedOutcomeRidgeModel:
    if not examples:
        raise ValueError("uplift training examples are required")
    if ridge_strength <= 0:
        raise ValueError("ridge_strength must be positive")
    feature_count = len(examples[0].features)
    if feature_count == 0 or any(
        len(example.features) != feature_count for example in examples
    ):
        raise ValueError("uplift features must have one stable non-zero dimension")

    means = tuple(
        sum(example.features[index] for example in examples) / len(examples)
        for index in range(feature_count)
    )
    scales = tuple(
        _standard_deviation(
            [example.features[index] for example in examples],
            means[index],
        )
        for index in range(feature_count)
    )
    design_rows: list[list[float]] = []
    targets: list[float] = []
    for example in examples:
        probability = float(example.treatment_probability)
        if not 0 < probability < 1:
            raise ValueError(
                "uplift training requires randomized treatment probabilities"
            )
        design_rows.append(
            [
                1.0,
                *(
                    (float(value) - means[index]) / scales[index]
                    for index, value in enumerate(example.features)
                ),
            ]
        )
        targets.append(
            float(example.outcome)
            * (float(example.treatment) - probability)
            / (probability * (1.0 - probability))
        )

    parameter_count = feature_count + 1
    normal_matrix = [
        [0.0 for _ in range(parameter_count)]
        for _ in range(parameter_count)
    ]
    normal_vector = [0.0 for _ in range(parameter_count)]
    for row, target in zip(design_rows, targets, strict=True):
        for left in range(parameter_count):
            normal_vector[left] += row[left] * target
            for right in range(parameter_count):
                normal_matrix[left][right] += row[left] * row[right]
    for index in range(1, parameter_count):
        normal_matrix[index][index] += ridge_strength
    parameters = _solve_linear_system(normal_matrix, normal_vector)
    return TransformedOutcomeRidgeModel(
        feature_means=means,
        feature_scales=scales,
        coefficients=tuple(parameters[1:]),
        intercept=parameters[0],
    )


def serving_cate_scores(
    *,
    model: TransformedOutcomeRidgeModel,
    metadata: UpliftModelMetadata,
    examples: Sequence[UpliftTrainingExample],
) -> list[float] | None:
    # No persistent registry or approval provenance exists in this version.
    # Metadata assembled in process must never activate uplift serving.
    del model, metadata, examples
    return None


def signed_cate_summary(scores: Sequence[float]) -> Mapping[str, float]:
    if not scores:
        return {
            "mean_cate": 0.0,
            "expected_incremental_bookings": 0.0,
            "negative_cate_user_ratio": 0.0,
        }
    return {
        "mean_cate": sum(scores) / len(scores),
        "expected_incremental_bookings": sum(scores),
        "negative_cate_user_ratio": sum(score < 0 for score in scores)
        / len(scores),
    }


def _standard_deviation(values: Sequence[float], mean: float) -> float:
    variance = sum((float(value) - mean) ** 2 for value in values) / len(values)
    return max(variance**0.5, 1e-12)


def _solve_linear_system(
    matrix: Sequence[Sequence[float]],
    vector: Sequence[float],
) -> list[float]:
    size = len(vector)
    augmented = [
        [float(value) for value in matrix[row]] + [float(vector[row])]
        for row in range(size)
    ]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) < 1e-12:
            augmented[pivot][column] = 1e-12
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        divisor = augmented[column][column]
        augmented[column] = [value / divisor for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            if factor == 0:
                continue
            augmented[row] = [
                value - factor * pivot_value
                for value, pivot_value in zip(
                    augmented[row],
                    augmented[column],
                    strict=True,
                )
            ]
    return [augmented[row][-1] for row in range(size)]
