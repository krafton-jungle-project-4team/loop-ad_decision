from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol

from app.logging import log


AUDIENCE_SELECTION_ARTIFACT_SCHEMA_VERSION = (
    "dec.segment-audience-selection-artifact.v1"
)
AUDIENCE_SELECTION_POLICY_VERSION = "dec.segment-audience-selection.v2"
DEFAULT_AUDIENCE_SELECTION_POLICY_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "audience_selection_policy_v1.json"
)


class AudienceSelectionPolicyError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class AudienceSelectionDecision:
    matching_user_count: int
    selected_user_count: int
    configured_ratio: float
    applied_ratio: float
    method: str
    calibration_status: str
    policy_version: str
    artifact_hash: str | None
    fallback_reason: str | None = None

    @property
    def selection_limited(self) -> bool:
        return self.selected_user_count < self.matching_user_count

    def to_metadata(self) -> dict[str, Any]:
        return {
            "version": self.policy_version,
            "method": self.method,
            "configured_ratio": round(self.configured_ratio, 6),
            "applied_ratio": round(self.applied_ratio, 6),
            "calibration_status": self.calibration_status,
            "artifact_hash": self.artifact_hash,
            "fallback_reason": self.fallback_reason,
        }


class AudienceSelectionPolicyProtocol(Protocol):
    def decide(
        self,
        *,
        goal_metric: str,
        candidate_type: str,
        matching_user_count: int,
    ) -> AudienceSelectionDecision:
        ...


@dataclass(frozen=True, slots=True)
class AudienceSelectionRule:
    goal_metric: str
    selected_ratio: float
    minimum_selected_user_count: int
    candidate_types: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.goal_metric.strip():
            raise AudienceSelectionPolicyError("goal_metric must not be empty")
        if not 0.0 < self.selected_ratio <= 1.0:
            raise AudienceSelectionPolicyError(
                "selected_ratio must be greater than 0 and at most 1"
            )
        if self.minimum_selected_user_count <= 0:
            raise AudienceSelectionPolicyError(
                "minimum_selected_user_count must be positive"
            )

    def supports(self, *, goal_metric: str, candidate_type: str) -> bool:
        return self.goal_metric == goal_metric and (
            not self.candidate_types or candidate_type in self.candidate_types
        )


@dataclass(frozen=True, slots=True)
class AudienceSelectionPolicy:
    policy_version: str
    calibration_status: str
    rules: tuple[AudienceSelectionRule, ...]
    artifact_hash: str | None = None
    default_fallback_reason: str | None = None

    def decide(
        self,
        *,
        goal_metric: str,
        candidate_type: str,
        matching_user_count: int,
    ) -> AudienceSelectionDecision:
        if matching_user_count < 0:
            raise ValueError("matching_user_count must not be negative")
        if matching_user_count == 0:
            return self._all_matching_decision(
                matching_user_count=0,
                configured_ratio=1.0,
                fallback_reason="empty_candidate",
            )
        if self.calibration_status != "validated":
            return self._all_matching_decision(
                matching_user_count=matching_user_count,
                configured_ratio=1.0,
                fallback_reason=(
                    self.default_fallback_reason or "policy_not_validated"
                ),
            )

        rule = next(
            (
                item
                for item in self.rules
                if item.supports(
                    goal_metric=goal_metric,
                    candidate_type=candidate_type,
                )
            ),
            None,
        )
        if rule is None:
            return self._all_matching_decision(
                matching_user_count=matching_user_count,
                configured_ratio=1.0,
                fallback_reason="unsupported_goal_or_candidate_type",
            )

        requested_count = min(
            matching_user_count,
            max(1, math.ceil(matching_user_count * rule.selected_ratio)),
        )
        if requested_count < rule.minimum_selected_user_count:
            return self._all_matching_decision(
                matching_user_count=matching_user_count,
                configured_ratio=rule.selected_ratio,
                fallback_reason="minimum_selected_user_count_not_met",
            )
        if requested_count >= matching_user_count:
            return self._all_matching_decision(
                matching_user_count=matching_user_count,
                configured_ratio=rule.selected_ratio,
                fallback_reason=None,
            )
        return AudienceSelectionDecision(
            matching_user_count=matching_user_count,
            selected_user_count=requested_count,
            configured_ratio=rule.selected_ratio,
            applied_ratio=requested_count / matching_user_count,
            method="top_behavior_strength_ratio",
            calibration_status=self.calibration_status,
            policy_version=self.policy_version,
            artifact_hash=self.artifact_hash,
        )

    def _all_matching_decision(
        self,
        *,
        matching_user_count: int,
        configured_ratio: float,
        fallback_reason: str | None,
    ) -> AudienceSelectionDecision:
        return AudienceSelectionDecision(
            matching_user_count=matching_user_count,
            selected_user_count=matching_user_count,
            configured_ratio=configured_ratio,
            applied_ratio=1.0,
            method="all_matching",
            calibration_status=self.calibration_status,
            policy_version=self.policy_version,
            artifact_hash=self.artifact_hash,
            fallback_reason=fallback_reason,
        )


def all_matching_audience_selection_policy(
    *,
    calibration_status: str = "pending_backtest",
    fallback_reason: str = "artifact_missing",
) -> AudienceSelectionPolicy:
    return AudienceSelectionPolicy(
        policy_version=AUDIENCE_SELECTION_POLICY_VERSION,
        calibration_status=calibration_status,
        rules=(),
        default_fallback_reason=fallback_reason,
    )


def fixed_ratio_audience_selection_policy(
    *,
    goal_metric: str,
    selected_ratio: float,
    minimum_selected_user_count: int,
    candidate_types: tuple[str, ...] = (),
    policy_version: str = "offline-evaluation",
) -> AudienceSelectionPolicy:
    return AudienceSelectionPolicy(
        policy_version=policy_version,
        calibration_status="validated",
        rules=(
            AudienceSelectionRule(
                goal_metric=goal_metric,
                selected_ratio=selected_ratio,
                minimum_selected_user_count=minimum_selected_user_count,
                candidate_types=candidate_types,
            ),
        ),
    )


def build_audience_selection_policy(
    artifact_path: str | Path | None = None,
) -> AudienceSelectionPolicy:
    path = (
        Path(artifact_path).expanduser()
        if artifact_path is not None
        else DEFAULT_AUDIENCE_SELECTION_POLICY_PATH
    )
    if not path.exists():
        return all_matching_audience_selection_policy()
    try:
        return load_audience_selection_policy(path)
    except (AudienceSelectionPolicyError, OSError, json.JSONDecodeError) as exc:
        log.warn(
            "audience_selection_policy_invalid",
            {"err": exc, "artifactPath": path},
        )
        return all_matching_audience_selection_policy(
            calibration_status="invalid_artifact",
            fallback_reason="artifact_invalid",
        )


def load_audience_selection_policy(path: Path) -> AudienceSelectionPolicy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise AudienceSelectionPolicyError("policy artifact must be a JSON object")
    if payload.get("schema_version") != AUDIENCE_SELECTION_ARTIFACT_SCHEMA_VERSION:
        raise AudienceSelectionPolicyError("unsupported policy artifact schema")
    expected_hash = str(payload.get("artifact_hash", "")).strip()
    actual_hash = audience_selection_artifact_hash(payload)
    if not expected_hash or expected_hash != actual_hash:
        raise AudienceSelectionPolicyError("policy artifact hash mismatch")

    raw_rules = payload.get("rules")
    if not isinstance(raw_rules, list):
        raise AudienceSelectionPolicyError("policy artifact rules must be a list")
    rules: list[AudienceSelectionRule] = []
    for raw_rule in raw_rules:
        if not isinstance(raw_rule, Mapping):
            raise AudienceSelectionPolicyError("policy rule must be an object")
        raw_candidate_types = raw_rule.get("candidate_types", ())
        if not isinstance(raw_candidate_types, (list, tuple)):
            raise AudienceSelectionPolicyError(
                "policy candidate_types must be a list"
            )
        rules.append(
            AudienceSelectionRule(
                goal_metric=str(raw_rule.get("goal_metric", "")),
                selected_ratio=float(raw_rule.get("selected_ratio", 0.0) or 0.0),
                minimum_selected_user_count=int(
                    raw_rule.get("minimum_selected_user_count", 0) or 0
                ),
                candidate_types=tuple(
                    str(value) for value in raw_candidate_types if str(value).strip()
                ),
            )
        )
    policy_version = str(payload.get("policy_version", "")).strip()
    if not policy_version:
        raise AudienceSelectionPolicyError("policy_version must not be empty")
    calibration_status = str(payload.get("calibration_status", "")).strip()
    if calibration_status not in {"validated", "fallback_all_matching"}:
        raise AudienceSelectionPolicyError("unsupported calibration_status")
    return AudienceSelectionPolicy(
        policy_version=policy_version,
        calibration_status=calibration_status,
        rules=tuple(rules),
        artifact_hash=expected_hash,
        default_fallback_reason=(
            "validation_not_passed"
            if calibration_status != "validated"
            else None
        ),
    )


def finalize_audience_selection_artifact(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    finalized = dict(payload)
    finalized.pop("artifact_hash", None)
    finalized["artifact_hash"] = audience_selection_artifact_hash(finalized)
    return finalized


def audience_selection_artifact_hash(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("artifact_hash", None)
    canonical = json.dumps(
        normalized,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def write_audience_selection_artifact(
    payload: Mapping[str, Any],
    path: Path,
) -> dict[str, Any]:
    finalized = finalize_audience_selection_artifact(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(finalized, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return finalized
