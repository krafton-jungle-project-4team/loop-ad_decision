from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from app.config import load_settings
from app.main import create_app
from app.uplift.contracts import UpliftDatasetBuildResult, UpliftTrainingExample
from app.uplift.dataset import MODEL_TYPE, TRAINING_CODE_VERSION
from app.uplift.registry import (
    EXTERNAL_VALIDATION_SCOPE,
    LOOPAD_VALIDATION_SCOPE,
    RegistryUpliftServingGuard,
    UpliftModelActivationError,
    UpliftModelLifecycleService,
    UpliftModelVersionRecord,
)
from app.uplift.router import get_uplift_model_lifecycle_service
from app.uplift.training import UpliftOneShotTrainingService
from tests.config_env import required_env_values


NOW = datetime(2026, 7, 22, tzinfo=UTC)


class _Registry:
    def __init__(self) -> None:
        self.records: dict[str, UpliftModelVersionRecord] = {}

    def insert_candidate(self, **values):
        existing = self.find_by_dataset_fingerprint(
            project_id=values["project_id"],
            model_type=values["model_type"],
            dataset_fingerprint=values["dataset_fingerprint"],
            training_code_version=values["training_code_version"],
        )
        if existing is not None:
            return existing
        record = UpliftModelVersionRecord(
            model_version_id=values["model_version_id"],
            project_id=values["project_id"],
            model_type=values["model_type"],
            lifecycle_status="candidate",
            validation_scope=values["validation_scope"],
            serving_eligible=False,
            dataset_fingerprint=values["dataset_fingerprint"],
            dataset_manifest_json=values["dataset_manifest_json"],
            feature_contract_hash=values["feature_contract_hash"],
            outcome_contract_hash=values["outcome_contract_hash"],
            training_code_version=values["training_code_version"],
            validation_policy_version=values["validation_policy_version"],
            split_policy_version=values["split_policy_version"],
            model_payload_json=values["model_payload_json"],
            metrics_json=values["metrics_json"],
            validation_result_json={},
            approved_by=None,
            approved_at=None,
            created_at=NOW,
            validated_at=None,
            activated_at=None,
            retired_at=None,
        )
        self.records[record.model_version_id] = record
        return record

    def get_by_id(self, model_version_id):
        return self.records.get(model_version_id)

    def find_by_dataset_fingerprint(self, **values):
        return next(
            (
                record
                for record in self.records.values()
                if record.project_id == values["project_id"]
                and record.model_type == values["model_type"]
                and record.dataset_fingerprint == values["dataset_fingerprint"]
                and record.training_code_version
                == values["training_code_version"]
            ),
            None,
        )

    def mark_validated(self, model_version_id, *, validation_result):
        record = replace(
            self.records[model_version_id],
            lifecycle_status="validated",
            validation_result_json=validation_result,
            validated_at=NOW,
        )
        self.records[model_version_id] = record
        return record

    def record_validation_result(self, model_version_id, *, validation_result):
        record = replace(
            self.records[model_version_id],
            validation_result_json=validation_result,
        )
        self.records[model_version_id] = record
        return record

    def mark_rejected(self, model_version_id, *, validation_result):
        record = replace(
            self.records[model_version_id],
            lifecycle_status="rejected",
            validation_result_json=validation_result,
        )
        self.records[model_version_id] = record
        return record

    def activate_validated(self, model_version_id, *, approved_by):
        target = self.records[model_version_id]
        for identifier, record in tuple(self.records.items()):
            if (
                record.lifecycle_status == "active"
                and record.project_id == target.project_id
                and record.feature_contract_hash == target.feature_contract_hash
                and record.outcome_contract_hash == target.outcome_contract_hash
            ):
                self.records[identifier] = replace(
                    record,
                    lifecycle_status="retired",
                    serving_eligible=False,
                    retired_at=NOW,
                )
        activated = replace(
            target,
            lifecycle_status="active",
            serving_eligible=True,
            approved_by=approved_by,
            approved_at=NOW,
            activated_at=NOW,
        )
        self.records[model_version_id] = activated
        return activated

    def retire_active(self, model_version_id):
        record = replace(
            self.records[model_version_id],
            lifecycle_status="retired",
            serving_eligible=False,
            retired_at=NOW,
        )
        self.records[model_version_id] = record
        return record

    def load_active_compatible(self, **values):
        return next(
            (
                record
                for record in self.records.values()
                if record.lifecycle_status == "active"
                and record.serving_eligible
                and record.project_id == values["project_id"]
                and record.model_type == values["model_type"]
                and record.feature_contract_hash
                == values["feature_contract_hash"]
                and record.outcome_contract_hash
                == values["outcome_contract_hash"]
            ),
            None,
        )


class _DatasetBuilder:
    def __init__(self, examples):
        self.examples = tuple(examples)
        self.calls = 0

    def build(self, *, project_id, reference_time):
        del project_id, reference_time
        self.calls += 1
        return UpliftDatasetBuildResult(
            examples=self.examples,
            excluded_reason_counts={},
            source_unit_count=len(self.examples),
        )


def test_one_shot_training_registers_validated_model_and_reuses_dataset() -> None:
    repository = _Registry()
    lifecycle = UpliftModelLifecycleService(repository)
    builder = _DatasetBuilder(training_examples())
    service = UpliftOneShotTrainingService(
        dataset_builder=builder,
        lifecycle_service=lifecycle,
        validation_policy=permissive_policy(),
    )

    first = service.run(project_id="project", reference_time=NOW)
    second = service.run(project_id="project", reference_time=NOW)

    assert first.model.lifecycle_status == "validated"
    assert first.model.serving_eligible is False
    assert first.validation_result["passed"] is True
    assert second.model.model_version_id == first.model.model_version_id
    assert second.reused_existing_model is True
    assert len(repository.records) == 1


def test_failed_policy_stays_candidate_and_external_model_cannot_activate() -> None:
    repository = _Registry()
    lifecycle = UpliftModelLifecycleService(repository)
    service = UpliftOneShotTrainingService(
        dataset_builder=_DatasetBuilder(training_examples()),
        lifecycle_service=lifecycle,
    )

    result = service.run(project_id="project", reference_time=NOW)

    assert result.model.lifecycle_status == "candidate"
    assert result.validation_result["passed"] is False
    with pytest.raises(UpliftModelActivationError):
        lifecycle.activate_validated_model(
            result.model.model_version_id,
            approved_by="reviewer",
        )

    external = lifecycle.register_candidate_model(
        project_id="project",
        model_type=MODEL_TYPE,
        validation_scope=EXTERNAL_VALIDATION_SCOPE,
        dataset_fingerprint="d" * 64,
        dataset_manifest={"dataset": "criteo"},
        feature_contract_hash="a" * 64,
        outcome_contract_hash="c" * 64,
        training_code_version=TRAINING_CODE_VERSION,
        validation_policy_version="uplift-validation.v1",
        split_policy_version="external-split.v1",
        model_payload=model_payload(),
        metrics={},
    )
    external = lifecycle.validate_candidate_model(
        external.model_version_id,
        validation_result={"passed": True},
    )
    assert external.lifecycle_status == "validated"
    with pytest.raises(UpliftModelActivationError, match="external"):
        lifecycle.activate_validated_model(
            external.model_version_id,
            approved_by="reviewer",
        )


def test_registry_serving_guard_scores_only_active_compatible_valid_payload() -> None:
    repository = _Registry()
    lifecycle = UpliftModelLifecycleService(repository)
    candidate = lifecycle.register_candidate_model(
        project_id="project",
        model_type=MODEL_TYPE,
        validation_scope=LOOPAD_VALIDATION_SCOPE,
        dataset_fingerprint="e" * 64,
        dataset_manifest={"dataset": "loopad"},
        feature_contract_hash="a" * 64,
        outcome_contract_hash="c" * 64,
        training_code_version=TRAINING_CODE_VERSION,
        validation_policy_version="uplift-validation.v1",
        split_policy_version="experiment-time-holdout.v1",
        model_payload=model_payload(),
        metrics={},
    )
    lifecycle.validate_candidate_model(
        candidate.model_version_id,
        validation_result={"passed": True},
    )
    guard = RegistryUpliftServingGuard(lifecycle)
    examples = [training_examples()[0]]

    assert guard.score(
        project_id="project",
        model_type=MODEL_TYPE,
        feature_contract_hash="a" * 64,
        outcome_contract_hash="c" * 64,
        examples=examples,
    ) is None

    lifecycle.activate_validated_model(
        candidate.model_version_id,
        approved_by="reviewer@example.com",
    )
    scores = guard.score(
        project_id="project",
        model_type=MODEL_TYPE,
        feature_contract_hash="a" * 64,
        outcome_contract_hash="c" * 64,
        examples=examples,
    )

    assert scores == [0.2]
    assert guard.score(
        project_id="project",
        model_type=MODEL_TYPE,
        feature_contract_hash="f" * 64,
        outcome_contract_hash="c" * 64,
        examples=examples,
    ) is None
    repository.records[candidate.model_version_id] = replace(
        repository.records[candidate.model_version_id],
        model_payload_json={"model_version": "corrupt"},
    )
    assert guard.score(
        project_id="project",
        model_type=MODEL_TYPE,
        feature_contract_hash="a" * 64,
        outcome_contract_hash="c" * 64,
        examples=examples,
    ) is None


def test_internal_activation_api_requires_auth_and_explicit_approver() -> None:
    env = required_env_values()
    env.update(
        {
            "LOOPAD_ENV": "test",
            "LOOPAD_SERVICE_ID": "decision-api",
            "PORT": "8080",
            "LOOPAD_AURORA_PORT": "5432",
        }
    )
    app = create_app(settings=load_settings(env))

    class _Service:
        def activate_validated_model(self, model_version_id, *, approved_by):
            assert model_version_id == "model"
            return replace(
                registry_record(),
                lifecycle_status="active",
                serving_eligible=True,
                approved_by=approved_by,
                approved_at=NOW,
                activated_at=NOW,
            )

    app.dependency_overrides[get_uplift_model_lifecycle_service] = lambda: _Service()
    client = TestClient(app)

    unauthorized = client.post(
        "/internal/decision/v1/uplift/models/model/activate",
        json={"approved_by": "reviewer"},
    )
    activated = client.post(
        "/internal/decision/v1/uplift/models/model/activate",
        headers={"X-Loop-Ad-Internal-Key": env["LOOPAD_INTERNAL_API_KEY"]},
        json={"approved_by": "reviewer"},
    )

    assert unauthorized.status_code == 401
    assert activated.status_code == 200
    assert activated.json()["lifecycle_status"] == "active"
    assert activated.json()["approved_by"] == "reviewer"


def training_examples() -> list[UpliftTrainingExample]:
    examples: list[UpliftTrainingExample] = []
    for experiment_index, days_ago in enumerate((5, 3, 1)):
        for feature, treatment, outcome in (
            (1.0, 1, 1),
            (1.0, 0, 0),
            (-1.0, 1, 0),
            (-1.0, 0, 1),
        ):
            suffix = len(examples)
            outcome_end = NOW - timedelta(days=days_ago)
            examples.append(
                UpliftTrainingExample(
                    experiment_unit_id=f"unit_{suffix}",
                    project_id="project",
                    promotion_run_id=f"run_{experiment_index}",
                    ad_experiment_id=f"experiment_{experiment_index}",
                    segment_id="segment",
                    user_id=f"user_{suffix}",
                    audience_snapshot_id=f"snapshot_{experiment_index}",
                    vector_generation_id="generation",
                    features=(feature,),
                    treatment=treatment,
                    outcome=outcome,
                    treatment_probability=0.5,
                    assigned_at=outcome_end - timedelta(days=30),
                    outcome_window_start=outcome_end - timedelta(days=30),
                    outcome_window_end=outcome_end,
                    vector_version="hotel_behavior.v2",
                    feature_contract_hash="a" * 64,
                    outcome_spec_hash="b" * 64,
                    outcome_contract_hash="c" * 64,
                )
            )
    return examples


def permissive_policy():
    return {
        "validation_policy_version": "uplift-validation.test",
        "policy_status": "provisional_safety_guard",
        "minimum_completed_experiments": 1,
        "minimum_treatment_observations": 1,
        "minimum_control_observations": 1,
        "minimum_positive_outcomes_per_arm": 1,
        "required_metrics": {
            "qini_above_zero": True,
            "auuc_above_baseline": True,
        },
        "requires_manual_approval": True,
        "statistical_power_derived": False,
    }


def model_payload():
    return {
        "model_version": MODEL_TYPE,
        "feature_means": [0.0],
        "feature_scales": [1.0],
        "coefficients": [0.2],
        "intercept": 0.0,
    }


def registry_record():
    return UpliftModelVersionRecord(
        model_version_id="model",
        project_id="project",
        model_type=MODEL_TYPE,
        lifecycle_status="validated",
        validation_scope=LOOPAD_VALIDATION_SCOPE,
        serving_eligible=False,
        dataset_fingerprint="d" * 64,
        dataset_manifest_json={},
        feature_contract_hash="a" * 64,
        outcome_contract_hash="c" * 64,
        training_code_version=TRAINING_CODE_VERSION,
        validation_policy_version="uplift-validation.v1",
        split_policy_version="experiment-time-holdout.v1",
        model_payload_json=model_payload(),
        metrics_json={},
        validation_result_json={"passed": True},
        approved_by=None,
        approved_at=None,
        created_at=NOW,
        validated_at=NOW,
        activated_at=None,
        retired_at=None,
    )
