from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import math
from typing import Any, Mapping

from app.uplift.contracts import UpliftDatasetBuildResult, UpliftTrainingExample
from app.uplift.dataset import (
    MODEL_TYPE,
    SPLIT_POLICY_VERSION,
    TRAINING_CODE_VERSION,
    UpliftDatasetBuilder,
    build_dataset_manifest,
)
from app.uplift.metrics import evaluate_uplift_predictions
from app.uplift.model import fit_transformed_outcome_ridge, signed_cate_summary
from app.uplift.registry import (
    LOOPAD_VALIDATION_SCOPE,
    UpliftModelLifecycleService,
    UpliftModelVersionRecord,
)
from app.uplift.split import split_by_experiment_end_time
from app.uplift.validation import (
    evaluate_validation_policy,
    load_validation_policy,
    predicted_cate_cluster_variability_interval,
)


@dataclass(frozen=True, slots=True)
class UpliftOneShotTrainingResult:
    model: UpliftModelVersionRecord
    dataset_manifest: Mapping[str, Any]
    dataset_summary: Mapping[str, Any]
    validation_result: Mapping[str, Any]
    reused_existing_model: bool

    def to_json(self) -> dict[str, Any]:
        return {
            "model_version_id": self.model.model_version_id,
            "project_id": self.model.project_id,
            "model_type": self.model.model_type,
            "lifecycle_status": self.model.lifecycle_status,
            "serving_eligible": self.model.serving_eligible,
            "dataset_fingerprint": self.model.dataset_fingerprint,
            "dataset_manifest": dict(self.dataset_manifest),
            "dataset_summary": dict(self.dataset_summary),
            "validation_result": dict(self.validation_result),
            "reused_existing_model": self.reused_existing_model,
            "active_transitioned": False,
        }


class UpliftOneShotTrainingService:
    def __init__(
        self,
        *,
        dataset_builder: UpliftDatasetBuilder,
        lifecycle_service: UpliftModelLifecycleService,
        validation_policy: Mapping[str, Any] | None = None,
    ) -> None:
        self._dataset_builder = dataset_builder
        self._lifecycle_service = lifecycle_service
        self._validation_policy = dict(
            validation_policy or load_validation_policy()
        )

    def run(
        self,
        *,
        project_id: str,
        reference_time: datetime,
    ) -> UpliftOneShotTrainingResult:
        dataset = self._dataset_builder.build(
            project_id=project_id,
            reference_time=reference_time,
        )
        manifest, dataset_fingerprint = build_dataset_manifest(
            dataset,
            project_id=project_id,
            reference_time=reference_time,
        )
        existing = self._lifecycle_service.find_by_dataset_fingerprint(
            project_id=project_id,
            model_type=MODEL_TYPE,
            dataset_fingerprint=dataset_fingerprint,
            training_code_version=TRAINING_CODE_VERSION,
        )
        if existing is not None:
            return UpliftOneShotTrainingResult(
                model=existing,
                dataset_manifest=existing.dataset_manifest_json,
                dataset_summary=dataset.summary(),
                validation_result=existing.validation_result_json,
                reused_existing_model=True,
            )

        split = split_by_experiment_end_time(dataset.examples)
        model = fit_transformed_outcome_ridge(split.train_examples)
        test_scores = model.predict_many(split.test_examples)
        metrics = evaluate_uplift_predictions(split.test_examples, test_scores)
        metrics.update(
            {
                "signed_cate_summary": dict(signed_cate_summary(test_scores)),
                "predicted_cate_cluster_variability_interval": (
                    predicted_cate_cluster_variability_interval(
                        split.test_examples,
                        test_scores,
                    ).to_json()
                ),
                "feature_ood": _feature_ood_diagnostic(
                    split.test_examples,
                    feature_means=model.feature_means,
                    feature_scales=model.feature_scales,
                ),
                "split": {
                    "policy_version": split.split_policy_version,
                    "train_experiment_ids": list(split.train_experiment_ids),
                    "test_experiment_ids": list(split.test_experiment_ids),
                    "train_observation_count": len(split.train_examples),
                    "test_observation_count": len(split.test_examples),
                },
            }
        )
        counts = _dataset_counts(dataset)
        validation_result = evaluate_validation_policy(
            metrics=metrics,
            completed_experiment_count=counts["completed_experiment_count"],
            treatment_observation_count=counts["treatment_observation_count"],
            control_observation_count=counts["control_observation_count"],
            positive_treatment_outcome_count=counts[
                "positive_treatment_outcome_count"
            ],
            positive_control_outcome_count=counts[
                "positive_control_outcome_count"
            ],
            policy=self._validation_policy,
        )
        metrics["training_population"] = counts
        candidate = self._lifecycle_service.register_candidate_model(
            project_id=project_id,
            model_type=MODEL_TYPE,
            validation_scope=LOOPAD_VALIDATION_SCOPE,
            dataset_fingerprint=dataset_fingerprint,
            dataset_manifest=manifest,
            feature_contract_hash=str(manifest["feature_contract_hash"]),
            outcome_contract_hash=str(manifest["outcome_contract_hash"]),
            training_code_version=TRAINING_CODE_VERSION,
            validation_policy_version=str(
                self._validation_policy["validation_policy_version"]
            ),
            split_policy_version=SPLIT_POLICY_VERSION,
            model_payload=model.to_payload(),
            metrics=metrics,
        )
        if candidate.lifecycle_status == "candidate":
            candidate = self._lifecycle_service.validate_candidate_model(
                candidate.model_version_id,
                validation_result=validation_result,
            )
        return UpliftOneShotTrainingResult(
            model=candidate,
            dataset_manifest=manifest,
            dataset_summary=dataset.summary(),
            validation_result=validation_result,
            reused_existing_model=False,
        )


def _dataset_counts(result: UpliftDatasetBuildResult) -> dict[str, int]:
    examples = result.examples
    return {
        "completed_experiment_count": len(
            {example.ad_experiment_id for example in examples}
        ),
        "treatment_observation_count": sum(
            example.treatment == 1 for example in examples
        ),
        "control_observation_count": sum(
            example.treatment == 0 for example in examples
        ),
        "positive_treatment_outcome_count": sum(
            example.treatment == 1 and example.outcome == 1
            for example in examples
        ),
        "positive_control_outcome_count": sum(
            example.treatment == 0 and example.outcome == 1
            for example in examples
        ),
    }


def _feature_ood_diagnostic(
    examples: tuple[UpliftTrainingExample, ...],
    *,
    feature_means: tuple[float, ...],
    feature_scales: tuple[float, ...],
) -> dict[str, Any]:
    absolute_z_scores = [
        abs((float(value) - feature_means[index]) / feature_scales[index])
        for example in examples
        for index, value in enumerate(example.features)
    ]
    return {
        "method": "standardized_feature_distance.v1",
        "reference_only": True,
        "maximum_absolute_z_score": max(absolute_z_scores, default=0.0),
        "feature_value_ratio_above_4sigma": (
            sum(value > 4.0 for value in absolute_z_scores)
            / len(absolute_z_scores)
            if absolute_z_scores
            else 0.0
        ),
        "all_values_finite": all(
            math.isfinite(value) for value in absolute_z_scores
        ),
    }
