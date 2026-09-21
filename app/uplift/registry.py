from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import hashlib
from typing import Any, Mapping, Sequence

from app.decision.repositories import PostgresExecutor
from app.uplift.contracts import UpliftTrainingExample
from app.uplift.model import (
    TransformedOutcomeRidgeModel,
    UpliftModelLifecycleStatus,
    UpliftModelMetadata,
    serving_cate_scores,
    validate_model_payload,
)


LOOPAD_VALIDATION_SCOPE = "loopad_randomized_experiments"
EXTERNAL_VALIDATION_SCOPE = "external_pipeline_validation"
MODEL_VERSION_ID_PREFIX = "uplift_model_"


class UpliftModelRegistryError(RuntimeError):
    pass


class UpliftModelNotFoundError(UpliftModelRegistryError):
    pass


class UpliftModelActivationError(UpliftModelRegistryError):
    pass


@dataclass(frozen=True, slots=True)
class UpliftModelVersionRecord:
    model_version_id: str
    project_id: str
    model_type: str
    lifecycle_status: str
    validation_scope: str
    serving_eligible: bool
    dataset_fingerprint: str
    dataset_manifest_json: Mapping[str, Any]
    feature_contract_hash: str
    outcome_contract_hash: str
    training_code_version: str
    validation_policy_version: str
    split_policy_version: str
    model_payload_json: Mapping[str, Any]
    metrics_json: Mapping[str, Any]
    validation_result_json: Mapping[str, Any]
    approved_by: str | None
    approved_at: datetime | None
    created_at: datetime
    validated_at: datetime | None
    activated_at: datetime | None
    retired_at: datetime | None


class UpliftModelRegistryRepository:
    _COLUMNS = """
        model_version_id,
        project_id,
        model_type,
        lifecycle_status,
        validation_scope,
        serving_eligible,
        dataset_fingerprint,
        dataset_manifest_json,
        feature_contract_hash,
        outcome_contract_hash,
        training_code_version,
        validation_policy_version,
        split_policy_version,
        model_payload_json,
        metrics_json,
        validation_result_json,
        approved_by,
        approved_at,
        created_at,
        validated_at,
        activated_at,
        retired_at
    """

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def insert_candidate(
        self,
        *,
        model_version_id: str,
        project_id: str,
        model_type: str,
        validation_scope: str,
        dataset_fingerprint: str,
        dataset_manifest_json: Mapping[str, Any],
        feature_contract_hash: str,
        outcome_contract_hash: str,
        training_code_version: str,
        validation_policy_version: str,
        split_policy_version: str,
        model_payload_json: Mapping[str, Any],
        metrics_json: Mapping[str, Any],
    ) -> UpliftModelVersionRecord:
        row = self._db.fetchone(
            f"""
            INSERT INTO uplift_model_versions (
                model_version_id,
                project_id,
                model_type,
                lifecycle_status,
                validation_scope,
                serving_eligible,
                dataset_fingerprint,
                dataset_manifest_json,
                feature_contract_hash,
                outcome_contract_hash,
                training_code_version,
                validation_policy_version,
                split_policy_version,
                model_payload_json,
                metrics_json,
                validation_result_json
            ) VALUES (
                %s, %s, %s, 'candidate', %s, false,
                %s, %s, %s, %s, %s, %s, %s, %s, %s, '{{}}'::jsonb
            )
            ON CONFLICT (
                project_id,
                model_type,
                dataset_fingerprint,
                training_code_version
            ) DO NOTHING
            RETURNING {self._COLUMNS}
            """,
            (
                model_version_id,
                project_id,
                model_type,
                validation_scope,
                dataset_fingerprint,
                dataset_manifest_json,
                feature_contract_hash,
                outcome_contract_hash,
                training_code_version,
                validation_policy_version,
                split_policy_version,
                model_payload_json,
                metrics_json,
            ),
        )
        if row is not None:
            return _record(row)
        existing = self.find_by_dataset_fingerprint(
            project_id=project_id,
            model_type=model_type,
            dataset_fingerprint=dataset_fingerprint,
            training_code_version=training_code_version,
        )
        if existing is None:
            raise UpliftModelRegistryError(
                "candidate insert conflicted without an existing model"
            )
        return existing

    def get_by_id(self, model_version_id: str) -> UpliftModelVersionRecord | None:
        row = self._db.fetchone(
            f"""
            SELECT {self._COLUMNS}
            FROM uplift_model_versions
            WHERE model_version_id = %s
            """,
            (model_version_id,),
        )
        return _record(row) if row is not None else None

    def find_by_dataset_fingerprint(
        self,
        *,
        project_id: str,
        model_type: str,
        dataset_fingerprint: str,
        training_code_version: str,
    ) -> UpliftModelVersionRecord | None:
        row = self._db.fetchone(
            f"""
            SELECT {self._COLUMNS}
            FROM uplift_model_versions
            WHERE project_id = %s
              AND model_type = %s
              AND dataset_fingerprint = %s
              AND training_code_version = %s
            """,
            (
                project_id,
                model_type,
                dataset_fingerprint,
                training_code_version,
            ),
        )
        return _record(row) if row is not None else None

    def mark_validated(
        self,
        model_version_id: str,
        *,
        validation_result: Mapping[str, Any],
    ) -> UpliftModelVersionRecord:
        return self._update_lifecycle(
            model_version_id,
            lifecycle_status="validated",
            validation_result=validation_result,
        )

    def record_validation_result(
        self,
        model_version_id: str,
        *,
        validation_result: Mapping[str, Any],
    ) -> UpliftModelVersionRecord:
        row = self._db.fetchone(
            f"""
            UPDATE uplift_model_versions
            SET validation_result_json = %s
            WHERE model_version_id = %s
              AND lifecycle_status = 'candidate'
            RETURNING {self._COLUMNS}
            """,
            (validation_result, model_version_id),
        )
        if row is None:
            raise UpliftModelRegistryError(
                "only candidate models can record validation results"
            )
        return _record(row)

    def mark_rejected(
        self,
        model_version_id: str,
        *,
        validation_result: Mapping[str, Any],
    ) -> UpliftModelVersionRecord:
        return self._update_lifecycle(
            model_version_id,
            lifecycle_status="rejected",
            validation_result=validation_result,
        )

    def activate_validated(
        self,
        model_version_id: str,
        *,
        approved_by: str,
    ) -> UpliftModelVersionRecord:
        try:
            row = self._db.fetchone(
                f"""
                SELECT {self._COLUMNS}
                FROM activate_uplift_model_version(%s, %s)
                """,
                (model_version_id, approved_by),
            )
        except Exception as exc:
            raise UpliftModelActivationError(
                "uplift model activation was rejected"
            ) from exc
        if row is None:
            raise UpliftModelRegistryError("model activation returned no row")
        return _record(row)

    def retire_active(self, model_version_id: str) -> UpliftModelVersionRecord:
        row = self._db.fetchone(
            f"""
            UPDATE uplift_model_versions
            SET lifecycle_status = 'retired',
                serving_eligible = false,
                retired_at = clock_timestamp()
            WHERE model_version_id = %s
              AND lifecycle_status = 'active'
            RETURNING {self._COLUMNS}
            """,
            (model_version_id,),
        )
        if row is None:
            raise UpliftModelRegistryError("only active models can be retired")
        return _record(row)

    def load_active_compatible(
        self,
        *,
        project_id: str,
        model_type: str,
        feature_contract_hash: str,
        outcome_contract_hash: str,
    ) -> UpliftModelVersionRecord | None:
        row = self._db.fetchone(
            f"""
            SELECT {self._COLUMNS}
            FROM uplift_model_versions
            WHERE project_id = %s
              AND model_type = %s
              AND feature_contract_hash = %s
              AND outcome_contract_hash = %s
              AND lifecycle_status = 'active'
              AND serving_eligible
              AND validation_scope = 'loopad_randomized_experiments'
            """,
            (
                project_id,
                model_type,
                feature_contract_hash,
                outcome_contract_hash,
            ),
        )
        return _record(row) if row is not None else None

    def _update_lifecycle(
        self,
        model_version_id: str,
        *,
        lifecycle_status: str,
        validation_result: Mapping[str, Any],
    ) -> UpliftModelVersionRecord:
        validated_at_sql = (
            "clock_timestamp()" if lifecycle_status == "validated" else "validated_at"
        )
        row = self._db.fetchone(
            f"""
            UPDATE uplift_model_versions
            SET lifecycle_status = %s,
                validation_result_json = %s,
                validated_at = {validated_at_sql}
            WHERE model_version_id = %s
              AND lifecycle_status IN ('candidate', 'validated')
            RETURNING {self._COLUMNS}
            """,
            (lifecycle_status, validation_result, model_version_id),
        )
        if row is None:
            raise UpliftModelRegistryError("model lifecycle transition was rejected")
        return _record(row)


class UpliftModelLifecycleService:
    def __init__(self, repository: UpliftModelRegistryRepository) -> None:
        self._repository = repository

    def register_candidate_model(
        self,
        *,
        project_id: str,
        model_type: str,
        validation_scope: str,
        dataset_fingerprint: str,
        dataset_manifest: Mapping[str, Any],
        feature_contract_hash: str,
        outcome_contract_hash: str,
        training_code_version: str,
        validation_policy_version: str,
        split_policy_version: str,
        model_payload: Mapping[str, Any],
        metrics: Mapping[str, Any],
    ) -> UpliftModelVersionRecord:
        _require_hash(dataset_fingerprint, "dataset fingerprint")
        _require_hash(feature_contract_hash, "feature contract hash")
        _require_hash(outcome_contract_hash, "outcome contract hash")
        validate_model_payload(model_payload)
        if validation_scope not in {
            LOOPAD_VALIDATION_SCOPE,
            EXTERNAL_VALIDATION_SCOPE,
            "synthetic_pipeline_validation",
        }:
            raise ValueError("unsupported uplift validation scope")
        model_version_id = MODEL_VERSION_ID_PREFIX + hashlib.sha256(
            (
                project_id
                + "\x1f"
                + model_type
                + "\x1f"
                + dataset_fingerprint
                + "\x1f"
                + training_code_version
            ).encode("utf-8")
        ).hexdigest()[:56]
        return self._repository.insert_candidate(
            model_version_id=model_version_id,
            project_id=project_id,
            model_type=model_type,
            validation_scope=validation_scope,
            dataset_fingerprint=dataset_fingerprint,
            dataset_manifest_json=dataset_manifest,
            feature_contract_hash=feature_contract_hash,
            outcome_contract_hash=outcome_contract_hash,
            training_code_version=training_code_version,
            validation_policy_version=validation_policy_version,
            split_policy_version=split_policy_version,
            model_payload_json=model_payload,
            metrics_json=metrics,
        )

    def find_by_dataset_fingerprint(
        self,
        *,
        project_id: str,
        model_type: str,
        dataset_fingerprint: str,
        training_code_version: str,
    ) -> UpliftModelVersionRecord | None:
        return self._repository.find_by_dataset_fingerprint(
            project_id=project_id,
            model_type=model_type,
            dataset_fingerprint=dataset_fingerprint,
            training_code_version=training_code_version,
        )

    def validate_candidate_model(
        self,
        model_version_id: str,
        *,
        validation_result: Mapping[str, Any],
    ) -> UpliftModelVersionRecord:
        if validation_result.get("passed") is True:
            return self._repository.mark_validated(
                model_version_id,
                validation_result=validation_result,
            )
        return self._repository.record_validation_result(
            model_version_id,
            validation_result=validation_result,
        )

    def activate_validated_model(
        self,
        model_version_id: str,
        *,
        approved_by: str,
    ) -> UpliftModelVersionRecord:
        approver = approved_by.strip()
        if not approver:
            raise UpliftModelActivationError("approved_by is required")
        record = self._repository.get_by_id(model_version_id)
        if record is None:
            raise UpliftModelNotFoundError(model_version_id)
        if record.lifecycle_status != "validated":
            raise UpliftModelActivationError("only validated models can be activated")
        if record.validation_scope != LOOPAD_VALIDATION_SCOPE:
            raise UpliftModelActivationError(
                "external or synthetic models cannot be activated"
            )
        if record.validation_result_json.get("passed") is not True:
            raise UpliftModelActivationError("validation policy did not pass")
        _require_hash(record.feature_contract_hash, "feature contract hash")
        _require_hash(record.outcome_contract_hash, "outcome contract hash")
        validate_model_payload(record.model_payload_json)
        return self._repository.activate_validated(
            model_version_id,
            approved_by=approver,
        )

    def retire_active_model(
        self,
        model_version_id: str,
    ) -> UpliftModelVersionRecord:
        return self._repository.retire_active(model_version_id)

    def load_active_compatible_model(
        self,
        *,
        project_id: str,
        model_type: str,
        feature_contract_hash: str,
        outcome_contract_hash: str,
    ) -> tuple[TransformedOutcomeRidgeModel, UpliftModelMetadata] | None:
        record = self._repository.load_active_compatible(
            project_id=project_id,
            model_type=model_type,
            feature_contract_hash=feature_contract_hash,
            outcome_contract_hash=outcome_contract_hash,
        )
        if record is None:
            return None
        model = TransformedOutcomeRidgeModel.from_payload(record.model_payload_json)
        metadata = UpliftModelMetadata(
            model_lifecycle_status=UpliftModelLifecycleStatus.ACTIVE,
            validation_scope=record.validation_scope,
            dataset="loopad_randomized_experiments",
            serving_eligible=record.serving_eligible,
            model_version=model.model_version,
            validation_policy_version=record.validation_policy_version,
            feature_contract_hash=record.feature_contract_hash,
            outcome_contract_hash=record.outcome_contract_hash,
            registry_verified=True,
        )
        return model, metadata


class RegistryUpliftServingGuard:
    def __init__(self, lifecycle_service: UpliftModelLifecycleService) -> None:
        self._lifecycle_service = lifecycle_service

    def score(
        self,
        *,
        project_id: str,
        model_type: str,
        feature_contract_hash: str,
        outcome_contract_hash: str,
        examples: Sequence[UpliftTrainingExample],
    ) -> list[float] | None:
        try:
            loaded = self._lifecycle_service.load_active_compatible_model(
                project_id=project_id,
                model_type=model_type,
                feature_contract_hash=feature_contract_hash,
                outcome_contract_hash=outcome_contract_hash,
            )
            if loaded is None:
                return None
            model, metadata = loaded
            return serving_cate_scores(
                model=model,
                metadata=metadata,
                examples=examples,
            )
        except Exception:
            return None


def _record(row: Mapping[str, Any]) -> UpliftModelVersionRecord:
    return UpliftModelVersionRecord(**dict(row))


def _require_hash(value: str, label: str) -> None:
    normalized = value.strip()
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError(f"{label} must be a lowercase SHA-256 hash")
