from __future__ import annotations

import hashlib
import hmac
import json
import math
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Mapping, Sequence


AssignmentMode = Literal["all_treatment", "randomized_holdout"]
ExperimentArm = Literal["treatment", "control"]

EXECUTION_SCHEMA_VERSION = "segment-assignment-execution.v2"
ALL_TREATMENT_VERSION = "all-treatment.v1"
HOLDOUT_RANDOMIZATION_VERSION = "holdout.v1"
COMPLETE_RANDOMIZATION_VERSION = "complete-randomization.v1"


class ExperimentDesignValidationError(ValueError):
    code = "experiment_design_invalid"


class RandomizedHoldoutAudienceTooSmallError(ExperimentDesignValidationError):
    code = "randomized_holdout_audience_too_small"


class RandomizedHoldoutConfigurationError(RuntimeError):
    code = "randomized_holdout_configuration_unavailable"


class ExperimentDesignConflictError(RuntimeError):
    code = "experiment_design_conflict"


@dataclass(frozen=True, slots=True)
class ExperimentDesign:
    mode: AssignmentMode
    requested_treatment_ratio: Decimal
    outcome_window_days: int
    randomization_version: str
    quota_policy_version: str
    randomization_salt_fingerprint: str | None
    outcome_spec_hash: str
    fingerprint: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "requested_treatment_ratio": float(self.requested_treatment_ratio),
            "outcome_window_days": self.outcome_window_days,
            "randomization_version": self.randomization_version,
            "quota_policy_version": self.quota_policy_version,
            "randomization_salt_fingerprint": self.randomization_salt_fingerprint,
            "outcome_spec_hash": self.outcome_spec_hash,
        }


@dataclass(frozen=True, slots=True)
class ExperimentAudienceMember:
    user_id: str
    segment_id: str
    ad_experiment_id: str
    audience_snapshot_id: str
    vector_generation_id: str
    behavior_fit_score: Decimal | None


@dataclass(frozen=True, slots=True)
class ExperimentUnitAllocation:
    member: ExperimentAudienceMember
    arm: ExperimentArm
    treatment_probability: Decimal


@dataclass(frozen=True, slots=True)
class ExperimentAllocationResult:
    ad_experiment_id: str
    segment_id: str
    audience_snapshot_id: str
    unit_count: int
    treatment_count: int
    control_count: int
    requested_treatment_ratio: Decimal
    actual_treatment_ratio: Decimal
    quota_policy_version: str

    def as_manifest(self) -> dict[str, Any]:
        return {
            "ad_experiment_id": self.ad_experiment_id,
            "segment_id": self.segment_id,
            "audience_snapshot_id": self.audience_snapshot_id,
            "unit_count": self.unit_count,
            "treatment_count": self.treatment_count,
            "control_count": self.control_count,
            "requested_treatment_ratio": float(self.requested_treatment_ratio),
            "actual_treatment_ratio": float(self.actual_treatment_ratio),
            "quota_policy_version": self.quota_policy_version,
        }


def build_experiment_design(
    *,
    mode: AssignmentMode,
    treatment_ratio: float | Decimal | None,
    outcome_window_days: int,
    outcome_spec_hash: str,
    randomization_salt: str | None,
) -> ExperimentDesign:
    if outcome_window_days < 1:
        raise ExperimentDesignValidationError(
            "outcome_window_days must be at least one"
        )
    if len(outcome_spec_hash) != 64:
        raise ExperimentDesignValidationError("outcome_spec_hash is invalid")

    if mode == "all_treatment":
        if treatment_ratio not in (None, 1, 1.0, Decimal("1")):
            raise ExperimentDesignValidationError(
                "all_treatment treatment_ratio must be omitted or one"
            )
        requested_ratio = Decimal("1")
        randomization_version = ALL_TREATMENT_VERSION
        salt_fingerprint = None
    elif mode == "randomized_holdout":
        if treatment_ratio is None:
            raise ExperimentDesignValidationError(
                "randomized_holdout treatment_ratio is required"
            )
        requested_ratio = Decimal(str(treatment_ratio))
        if not Decimal("0") < requested_ratio < Decimal("1"):
            raise ExperimentDesignValidationError(
                "randomized_holdout treatment_ratio must be between zero and one"
            )
        if not randomization_salt:
            raise RandomizedHoldoutConfigurationError(
                "randomized holdout salt is not configured"
            )
        randomization_version = HOLDOUT_RANDOMIZATION_VERSION
        salt_fingerprint = hashlib.sha256(
            randomization_salt.encode("utf-8")
        ).hexdigest()
    else:
        raise ExperimentDesignValidationError(f"unsupported assignment mode: {mode}")

    normalized = {
        "mode": mode,
        "requested_treatment_ratio": _decimal_text(requested_ratio),
        "outcome_window_days": outcome_window_days,
        "randomization_version": randomization_version,
        "quota_policy_version": COMPLETE_RANDOMIZATION_VERSION,
        "randomization_salt_fingerprint": salt_fingerprint,
        "outcome_spec_hash": outcome_spec_hash,
    }
    return ExperimentDesign(
        mode=mode,
        requested_treatment_ratio=requested_ratio,
        outcome_window_days=outcome_window_days,
        randomization_version=randomization_version,
        quota_policy_version=COMPLETE_RANDOMIZATION_VERSION,
        randomization_salt_fingerprint=salt_fingerprint,
        outcome_spec_hash=outcome_spec_hash,
        fingerprint=fingerprint(normalized),
    )


def allocate_experiment_units(
    *,
    project_id: str,
    promotion_run_id: str,
    design: ExperimentDesign,
    members: Sequence[ExperimentAudienceMember],
    randomization_salt: str | None,
    experiment_bindings: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[list[ExperimentUnitAllocation], list[ExperimentAllocationResult]]:
    resolved_bindings = dict(experiment_bindings or {})
    members_by_experiment: dict[str, list[ExperimentAudienceMember]] = {
        experiment_id: [] for experiment_id in resolved_bindings
    }
    for member in members:
        members_by_experiment.setdefault(member.ad_experiment_id, []).append(member)

    allocations: list[ExperimentUnitAllocation] = []
    results: list[ExperimentAllocationResult] = []
    for experiment_id in sorted(members_by_experiment):
        experiment_members = members_by_experiment[experiment_id]
        if design.mode == "randomized_holdout" and len(experiment_members) < 2:
            raise RandomizedHoldoutAudienceTooSmallError(
                f"randomized holdout requires at least two users: {experiment_id}"
            )
        ordered = _ordered_members(
            project_id=project_id,
            promotion_run_id=promotion_run_id,
            experiment_id=experiment_id,
            randomization_version=design.randomization_version,
            members=experiment_members,
            randomization_salt=randomization_salt,
        )
        unit_count = len(ordered)
        if design.mode == "all_treatment":
            treatment_count = unit_count
        else:
            raw_quota = math.floor(
                unit_count * float(design.requested_treatment_ratio) + 0.5
            )
            treatment_count = min(max(raw_quota, 1), unit_count - 1)
        actual_ratio = (
            Decimal(treatment_count) / Decimal(unit_count)
            if unit_count
            else Decimal("0")
        )
        for index, member in enumerate(ordered):
            allocations.append(
                ExperimentUnitAllocation(
                    member=member,
                    arm="treatment" if index < treatment_count else "control",
                    treatment_probability=actual_ratio,
                )
            )
        first = ordered[0] if ordered else None
        fallback_binding = resolved_bindings.get(experiment_id, ("", ""))
        results.append(
            ExperimentAllocationResult(
                ad_experiment_id=experiment_id,
                segment_id=first.segment_id if first else fallback_binding[0],
                audience_snapshot_id=(
                    first.audience_snapshot_id if first else fallback_binding[1]
                ),
                unit_count=unit_count,
                treatment_count=treatment_count,
                control_count=unit_count - treatment_count,
                requested_treatment_ratio=design.requested_treatment_ratio,
                actual_treatment_ratio=actual_ratio,
                quota_policy_version=design.quota_policy_version,
            )
        )
    allocations.sort(key=lambda allocation: allocation.member.user_id)
    return allocations, results


def build_request_fingerprint(
    *,
    promotion_run_id: str,
    design_fingerprint: str,
    expires_in_days: int | None,
) -> str:
    return fingerprint(
        {
            "promotion_run_id": promotion_run_id,
            "experiment_design_fingerprint": design_fingerprint,
            "expires_in_days": expires_in_days,
        }
    )


def build_input_fingerprint(
    *,
    audience_bindings: Sequence[Mapping[str, Any]],
) -> str:
    return fingerprint(
        {
            "audience_bindings": sorted(
                (dict(binding) for binding in audience_bindings),
                key=lambda binding: (
                    str(binding.get("segment_id")),
                    str(binding.get("audience_snapshot_id")),
                ),
            )
        }
    )


def build_execution_id(promotion_run_id: str, request_fingerprint: str) -> str:
    digest = fingerprint(
        {
            "promotion_run_id": promotion_run_id,
            "request_fingerprint": request_fingerprint,
        }
    )
    return f"assign_exec_{digest[:48]}"


def build_experiment_unit_id(
    *,
    promotion_run_id: str,
    user_id: str,
) -> str:
    return "adunit_" + fingerprint(
        {"promotion_run_id": promotion_run_id, "user_id": user_id}
    )[:56]


def fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _ordered_members(
    *,
    project_id: str,
    promotion_run_id: str,
    experiment_id: str,
    randomization_version: str,
    members: Sequence[ExperimentAudienceMember],
    randomization_salt: str | None,
) -> list[ExperimentAudienceMember]:
    if randomization_salt is None:
        return sorted(members, key=lambda member: member.user_id)

    def randomization_key(member: ExperimentAudienceMember) -> tuple[bytes, str]:
        message = "\x1f".join(
            (
                project_id,
                promotion_run_id,
                experiment_id,
                member.audience_snapshot_id,
                member.user_id,
                randomization_version,
            )
        ).encode("utf-8")
        return (
            hmac.new(
                randomization_salt.encode("utf-8"),
                message,
                hashlib.sha256,
            ).digest(),
            member.user_id,
        )

    return sorted(members, key=randomization_key)


def _decimal_text(value: Decimal) -> str:
    return format(value.normalize(), "f")
