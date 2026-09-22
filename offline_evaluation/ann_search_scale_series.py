"""Scale-ready contracts for the offline Audience ANN benchmark v2.

This module is intentionally offline-only.  It defines deterministic cohort
membership, evidence fingerprints, scenario census buckets, and the two
screening survivor sets required by Goal 1.  It does not change the runtime
selector or any public/Data Contract schema.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from app.analysis.audience_search_repository import (
    HardMatchAggregateRequest,
    _hard_predicate_batch_query,
    _hard_predicate_query,
)


SCALE_EXPERIMENT_VERSION = "audience_search.benchmark.v2"
SCALE_OUTPUT_ROOT = Path("artifacts/ann-search/scale-series-v2")
SCALE_VECTOR_VERSION = "hotel_behavior.v2"
SCALE_WINDOW_START = "2013-01-01T00:00:00+00:00"
SCALE_WINDOW_END = "2015-01-01T00:00:00+00:00"
SCALE_COHORT_SIZES = (50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000)
REFERENCE_SAMPLE_SIZE = 50_000
COHORT_MEMBERSHIP_RELATION = "ann_benchmark_scale_membership"
COHORT_SIGNAL_RELATION = "ann_benchmark_user_signals"

FINGERPRINT_REQUIRED_FIELDS = (
    "experiment_version",
    "scale_series_id",
    "project_id",
    "vector_version",
    "vector_manifest_hash",
    "vector_generation_id",
    "window_start",
    "window_end",
    "source_revision_cutoff",
    "source_user_count",
    "source_vector_revision_count",
    "raw_event_count",
    "cohort_seed",
    "membership_sha256",
    "scenario_manifest_sha256",
    "environment_sha256",
    "resource_manifest_sha256",
    "code_revision",
)


class ScenarioSet(StrEnum):
    TUNING = "tuning"
    CONFIRMATION = "confirmation"


class HardMatchBucket(StrEnum):
    LE_005 = "le_0_05"
    GT_005_LE_020 = "gt_0_05_le_0_20"
    GT_020 = "gt_0_20"


class ExpectedMemberBucket(StrEnum):
    LE_001 = "le_0_01"
    GT_001_LE_005 = "gt_0_01_le_0_05"
    GT_005_LE_010 = "gt_0_05_le_0_10"
    GT_010_LE_025 = "gt_0_10_le_0_25"
    GT_025 = "gt_0_25"


@dataclass(frozen=True, slots=True, order=True)
class CohortMembership:
    scale_series_id: str
    user_id: str
    cohort_rank: int

    def __post_init__(self) -> None:
        if not self.scale_series_id or not self.user_id or self.cohort_rank <= 0:
            raise ValueError("cohort membership fields are invalid")


@dataclass(frozen=True, slots=True)
class ScaleCohortScope:
    scale_series_id: str
    cohort_size: int
    reference_sample_seed: str
    reference_sample_size: int = REFERENCE_SAMPLE_SIZE

    def __post_init__(self) -> None:
        if not self.scale_series_id or not self.reference_sample_seed:
            raise ValueError("scale cohort scope identity is required")
        if self.cohort_size <= 0 or self.reference_sample_size <= 0:
            raise ValueError("scale cohort scope sizes must be positive")

    @property
    def effective_reference_sample_size(self) -> int:
        return min(self.cohort_size, self.reference_sample_size)

    def clickhouse_parameters(self) -> Mapping[str, Any]:
        return {
            "scale_series_id": self.scale_series_id,
            "cohort_size": self.cohort_size,
            "reference_sample_seed": self.reference_sample_seed,
            "reference_sample_size": self.reference_sample_size,
        }


@dataclass(frozen=True, slots=True)
class FingerprintCheck:
    complete: bool
    fingerprint_sha256: str | None
    missing_fields: tuple[str, ...]
    invalid_fields: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReuseDecision:
    reusable: bool
    decision: str
    reason: str
    expected_fingerprint_sha256: str | None
    existing_fingerprint_sha256: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ScenarioCensusRecord:
    scenario_id: str
    definition_sha256: str
    candidate_type: str
    scenario_set: ScenarioSet
    corpus_user_count: int
    hard_match_user_count: int
    estimated_member_count: float
    exact_positive_count: int

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.candidate_type:
            raise ValueError("scenario census identity is required")
        if len(self.definition_sha256) != 64:
            raise ValueError("scenario definition hash must be SHA-256")
        if self.corpus_user_count <= 0:
            raise ValueError("scenario census corpus must be positive")
        if not 0 <= self.hard_match_user_count <= self.corpus_user_count:
            raise ValueError("scenario census H is invalid")
        if not 0 <= self.estimated_member_count <= self.hard_match_user_count:
            raise ValueError("scenario census E is invalid")
        if not 0 <= self.exact_positive_count <= self.hard_match_user_count:
            raise ValueError("scenario census exact positives are invalid")

    @property
    def hard_match_ratio(self) -> float:
        return self.hard_match_user_count / self.corpus_user_count

    @property
    def expected_member_ratio(self) -> float:
        return self.estimated_member_count / self.corpus_user_count

    @property
    def bucket_key(self) -> tuple[str, HardMatchBucket, ExpectedMemberBucket]:
        return (
            self.candidate_type,
            hard_match_bucket(self.hard_match_ratio),
            expected_member_bucket(self.expected_member_ratio),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario_set"] = self.scenario_set.value
        payload["hard_match_ratio"] = self.hard_match_ratio
        payload["expected_member_ratio"] = self.expected_member_ratio
        payload["hard_match_bucket"] = self.bucket_key[1].value
        payload["expected_member_bucket"] = self.bucket_key[2].value
        return payload


@dataclass(frozen=True, slots=True)
class ScenarioPair:
    candidate_type: str
    hard_match_bucket: HardMatchBucket
    expected_member_bucket: ExpectedMemberBucket
    tuning_scenario_id: str | None
    confirmation_scenario_id: str | None
    validated: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["hard_match_bucket"] = self.hard_match_bucket.value
        payload["expected_member_bucket"] = self.expected_member_bucket.value
        return payload


@dataclass(frozen=True, slots=True)
class ScreeningCandidate:
    scenario_id: str
    candidate_type: str
    scenario_set: ScenarioSet
    corpus_user_count: int
    requested_k: int
    exact_positive_count: int
    recall: float
    ann_p95_ms: float
    exact_all_p95_ms: float
    filter_first_p95_ms: float | None
    filter_first_results_equal: bool
    hnsw_index_used: bool
    temp_spill: bool
    oom: bool

    def __post_init__(self) -> None:
        if not self.scenario_id or not self.candidate_type:
            raise ValueError("screening candidate identity is required")
        if self.corpus_user_count <= 0:
            raise ValueError("screening corpus must be positive")
        if not 0 < self.requested_k <= self.corpus_user_count:
            raise ValueError("screening K is outside the corpus")
        if self.exact_positive_count < 0 or not 0 <= self.recall <= 1:
            raise ValueError("screening quality values are invalid")
        if min(self.ann_p95_ms, self.exact_all_p95_ms) < 0:
            raise ValueError("screening latency must not be negative")
        if self.filter_first_p95_ms is not None and self.filter_first_p95_ms < 0:
            raise ValueError("filter-first latency must not be negative")

    @property
    def identity(self) -> tuple[int, str, int]:
        return (self.corpus_user_count, self.scenario_id, self.requested_k)

    @property
    def scale_survivor(self) -> bool:
        return (
            self.scenario_set == ScenarioSet.TUNING
            and self.exact_positive_count >= 100
            and self.recall >= 0.90
            and self.ann_p95_ms <= self.exact_all_p95_ms * 1.20
            and self.hnsw_index_used
            and not self.temp_spill
            and not self.oom
        )

    @property
    def policy_exact_p95_ms(self) -> float:
        if (
            self.filter_first_results_equal
            and self.filter_first_p95_ms is not None
        ):
            return min(self.exact_all_p95_ms, self.filter_first_p95_ms)
        return self.exact_all_p95_ms

    @property
    def policy_survivor(self) -> bool:
        return (
            self.scale_survivor
            and self.ann_p95_ms <= self.policy_exact_p95_ms * 1.20
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["scenario_set"] = self.scenario_set.value
        payload["scale_survivor"] = self.scale_survivor
        payload["policy_survivor"] = self.policy_survivor
        payload["policy_exact_p95_ms"] = self.policy_exact_p95_ms
        return payload


@dataclass(frozen=True, slots=True)
class SurvivorSets:
    scale: tuple[ScreeningCandidate, ...]
    policy: tuple[ScreeningCandidate, ...]
    goal2_union: tuple[ScreeningCandidate, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "scale": [item.to_dict() for item in self.scale],
            "policy": [item.to_dict() for item in self.policy],
            "goal2_union": [item.to_dict() for item in self.goal2_union],
        }


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def inspect_fingerprint(payload: Mapping[str, Any]) -> FingerprintCheck:
    missing = tuple(
        field
        for field in FINGERPRINT_REQUIRED_FIELDS
        if field not in payload or payload[field] in (None, "")
    )
    invalid: list[str] = []
    if payload.get("experiment_version") not in (None, SCALE_EXPERIMENT_VERSION):
        invalid.append("experiment_version")
    if payload.get("vector_version") not in (None, SCALE_VECTOR_VERSION):
        invalid.append("vector_version")
    if payload.get("window_start") not in (None, SCALE_WINDOW_START):
        invalid.append("window_start")
    if payload.get("window_end") not in (None, SCALE_WINDOW_END):
        invalid.append("window_end")
    for count_field in (
        "source_user_count",
        "source_vector_revision_count",
        "raw_event_count",
    ):
        value = payload.get(count_field)
        if value is not None and (
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
        ):
            invalid.append(count_field)
    for hash_field in (
        "vector_manifest_hash",
        "membership_sha256",
        "scenario_manifest_sha256",
        "environment_sha256",
        "resource_manifest_sha256",
    ):
        value = payload.get(hash_field)
        if value is not None and (
            not isinstance(value, str)
            or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)
        ):
            invalid.append(hash_field)
    complete = not missing and not invalid
    return FingerprintCheck(
        complete=complete,
        fingerprint_sha256=(
            canonical_json_sha256(
                {field: payload[field] for field in FINGERPRINT_REQUIRED_FIELDS}
            )
            if complete
            else None
        ),
        missing_fields=missing,
        invalid_fields=tuple(sorted(set(invalid))),
    )


def decide_artifact_reuse(
    existing: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> ReuseDecision:
    existing_check = inspect_fingerprint(existing)
    expected_check = inspect_fingerprint(expected)
    if not expected_check.complete:
        return ReuseDecision(
            reusable=False,
            decision="reject",
            reason="expected fingerprint incomplete",
            expected_fingerprint_sha256=None,
            existing_fingerprint_sha256=existing_check.fingerprint_sha256,
        )
    if not existing_check.complete:
        return ReuseDecision(
            reusable=False,
            decision="reject",
            reason="existing fingerprint incomplete",
            expected_fingerprint_sha256=expected_check.fingerprint_sha256,
            existing_fingerprint_sha256=None,
        )
    if existing_check.fingerprint_sha256 != expected_check.fingerprint_sha256:
        return ReuseDecision(
            reusable=False,
            decision="reject",
            reason="fingerprint mismatch",
            expected_fingerprint_sha256=expected_check.fingerprint_sha256,
            existing_fingerprint_sha256=existing_check.fingerprint_sha256,
        )
    return ReuseDecision(
        reusable=True,
        decision="reuse",
        reason="complete fingerprint match",
        expected_fingerprint_sha256=expected_check.fingerprint_sha256,
        existing_fingerprint_sha256=existing_check.fingerprint_sha256,
    )


def old_50k_reuse_decision() -> Mapping[str, Any]:
    return {
        "decision": "pilot_only",
        "latency_reused": False,
        "ground_truth_reused": False,
        "reason": (
            "historical fingerprint incomplete and new full-source snapshot differs"
        ),
    }


def rank_cohort_members(
    *,
    scale_series_id: str,
    cohort_seed: str,
    user_ids: Iterable[str],
) -> tuple[CohortMembership, ...]:
    if not scale_series_id or not cohort_seed:
        raise ValueError("scale series and cohort seed are required")
    unique = {str(user_id) for user_id in user_ids}
    if not unique or "" in unique:
        raise ValueError("cohort user IDs must be non-empty")
    ordered = sorted(
        unique,
        key=lambda user_id: (
            hashlib.sha256(f"{cohort_seed}|{user_id}".encode()).digest(),
            user_id,
        ),
    )
    return tuple(
        CohortMembership(scale_series_id, user_id, rank)
        for rank, user_id in enumerate(ordered, start=1)
    )


def cohort_prefix(
    memberships: Sequence[CohortMembership],
    cohort_size: int,
) -> tuple[str, ...]:
    if cohort_size <= 0 or cohort_size > len(memberships):
        raise ValueError("cohort size exceeds prepared membership")
    ordered = sorted(memberships, key=lambda item: item.cohort_rank)
    _validate_contiguous_membership(ordered)
    return tuple(item.user_id for item in ordered[:cohort_size])


def cohort_prefix_sha256(user_ids: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for user_id in user_ids:
        digest.update(str(user_id).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_nested_prefixes(
    memberships: Sequence[CohortMembership],
    cohort_sizes: Sequence[int],
) -> Mapping[int, str]:
    sizes = tuple(sorted(set(int(value) for value in cohort_sizes)))
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("cohort sizes must be positive")
    prefixes = {size: cohort_prefix(memberships, size) for size in sizes}
    previous: tuple[str, ...] = ()
    for size in sizes:
        current = prefixes[size]
        if current[: len(previous)] != previous:
            raise ValueError("smaller cohort is not an exact larger-cohort prefix")
        previous = current
    return {size: cohort_prefix_sha256(prefixes[size]) for size in sizes}


def membership_create_sql() -> str:
    return f"""
        CREATE TABLE IF NOT EXISTS {COHORT_MEMBERSHIP_RELATION} (
            scale_series_id String,
            user_id String,
            cohort_rank UInt64,
            loaded_at DateTime64(6, 'UTC') DEFAULT now64(6)
        )
        ENGINE = MergeTree
        ORDER BY (scale_series_id, cohort_rank, user_id)
    """.strip()


def membership_insert_sql() -> str:
    return (
        f"INSERT INTO {COHORT_MEMBERSHIP_RELATION} "
        "(scale_series_id, user_id, cohort_rank) VALUES"
    )


def membership_population_sql() -> str:
    return f"""
        INSERT INTO {COHORT_MEMBERSHIP_RELATION} (
            scale_series_id, user_id, cohort_rank
        )
        SELECT
            {{scale_series_id:String}} AS scale_series_id,
            user_id,
            row_number() OVER (
                ORDER BY
                    hex(SHA256(concat({{cohort_seed:String}}, '|', user_id))),
                    user_id
            ) AS cohort_rank
        FROM (
            SELECT user_id
            FROM user_behavior_vector_revisions
            WHERE project_id = {{project_id:String}}
              AND vector_version = {{vector_version:String}}
              AND window_start = {{window_start:DateTime64(3, 'UTC')}}
              AND window_end = {{window_end:DateTime64(3, 'UTC')}}
              AND ingested_at <= {{source_revision_cutoff:DateTime64(6, 'UTC')}}
            GROUP BY user_id
        ) AS frozen_vector_users
        ORDER BY cohort_rank
    """.strip()


def build_environment_manifest(
    *,
    code_revision: str,
    postgres_version: str,
    pgvector_version: str,
    clickhouse_version: str,
    container_identity: Mapping[str, str],
) -> Mapping[str, Any]:
    required = (
        code_revision,
        postgres_version,
        pgvector_version,
        clickhouse_version,
    )
    if any(not value for value in required) or not container_identity:
        raise ValueError("environment identity is incomplete")
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "code_revision": code_revision,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "postgres_version": postgres_version,
        "pgvector_version": pgvector_version,
        "clickhouse_version": clickhouse_version,
        "containers": dict(sorted(container_identity.items())),
    }


def build_resource_manifest(
    *,
    postgres_settings: Mapping[str, str],
    clickhouse_settings: Mapping[str, str],
    container_limits: Mapping[str, Mapping[str, int | str]],
    filesystem_path: Path = Path("."),
    logical_cpu_count: int | None = None,
    physical_memory_bytes: int | None = None,
) -> Mapping[str, Any]:
    if not postgres_settings or not clickhouse_settings or not container_limits:
        raise ValueError("resource manifest inputs are incomplete")
    disk = shutil.disk_usage(filesystem_path)
    cpu_count = logical_cpu_count if logical_cpu_count is not None else os.cpu_count()
    if cpu_count is None or cpu_count <= 0:
        raise ValueError("logical CPU count is unavailable")
    if physical_memory_bytes is not None and physical_memory_bytes <= 0:
        raise ValueError("physical memory must be positive")
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "logical_cpu_count": cpu_count,
        "physical_memory_bytes": physical_memory_bytes,
        "filesystem": {
            "path": str(filesystem_path.resolve()),
            "total_bytes": disk.total,
            "free_bytes_at_capture": disk.free,
        },
        "postgres_settings": dict(sorted(postgres_settings.items())),
        "clickhouse_settings": dict(sorted(clickhouse_settings.items())),
        "container_limits": {
            name: dict(sorted(values.items()))
            for name, values in sorted(container_limits.items())
        },
    }


def build_fingerprint(
    **fields: Any,
) -> Mapping[str, Any]:
    payload = dict(fields)
    check = inspect_fingerprint(payload)
    if not check.complete:
        raise ValueError(
            "fingerprint is incomplete: missing="
            + ",".join(check.missing_fields)
            + "; invalid="
            + ",".join(check.invalid_fields)
        )
    return {
        **payload,
        "fingerprint_sha256": check.fingerprint_sha256,
    }


def bind_query_to_cohort(query: str) -> str:
    marker = "FROM raw_events"
    if query.count(marker) != 1:
        raise ValueError("hard-predicate query must contain one raw_events source")
    membership_join = f"""FROM raw_events
        INNER JOIN (
            SELECT user_id
            FROM {COHORT_MEMBERSHIP_RELATION}
            WHERE scale_series_id = {{scale_series_id:String}}
              AND cohort_rank <= {{cohort_size:UInt64}}
        ) AS benchmark_cohort USING (user_id)"""
    return query.replace(marker, membership_join, 1)


def cohort_bound_hard_match_query(
    hard_predicate_keys: Sequence[str],
) -> str:
    users = cohort_bound_hard_match_users_query(hard_predicate_keys)
    return "SELECT count() AS matching_user_count FROM (" + users + ")"


def cohort_bound_hard_match_users_query(
    hard_predicate_keys: Sequence[str],
) -> str:
    predicate, _ = cohort_signal_predicate(hard_predicate_keys, {})
    return f"""
        SELECT signals.user_id
        FROM {COHORT_SIGNAL_RELATION} AS signals
        INNER JOIN {COHORT_MEMBERSHIP_RELATION} AS membership
            ON membership.scale_series_id = {{scale_series_id:String}}
           AND membership.cohort_rank <= {{cohort_size:UInt64}}
           AND membership.user_id = signals.user_id
        WHERE signals.scale_series_id = {{scale_series_id:String}}
          AND signals.cohort_rank <= {{cohort_size:UInt64}}
          AND ({predicate})
        ORDER BY signals.user_id
    """.strip()


def cohort_bound_reference_sample_query(
    hard_predicate_keys: Sequence[str],
) -> str:
    hard_users = cohort_bound_hard_match_users_query(hard_predicate_keys)
    return f"""
        SELECT user_id
        FROM ({hard_users}) AS hard_match_users
        ORDER BY
            hex(SHA256(concat({{reference_sample_seed:String}}, '|', user_id))),
            user_id
        LIMIT least({{reference_sample_size:UInt32}}, {{cohort_size:UInt64}})
    """.strip()


def cohort_signal_predicate(
    keys: Sequence[str],
    raw_parameters: Mapping[str, Sequence[Any]],
) -> tuple[str, Mapping[str, Any]]:
    """Compile production hard predicates over the frozen per-user signal row.

    ``ann_benchmark_user_signals`` is built once from the immutable raw-event
    window with the production predicate formulas.  Using it here prevents the
    benchmark itself from issuing one full raw-event scan per 1,000 candidates.
    The membership join above remains the authoritative cohort boundary.
    """
    parameters = {
        "destinations": list(raw_parameters.get("destinations", ())),
        "season_months": [
            int(value) for value in raw_parameters.get("season_months", ())
        ],
        "benefit_keys": list(raw_parameters.get("benefit_keys", ())),
    }
    conditions: list[str] = []
    for key in keys:
        if key == "hotel_product_interest":
            conditions.append("signals.hotel_interest_count > 0")
        elif key == "target_destination_affinity":
            conditions.append(
                "arrayCount(value -> value IN {destinations:Array(String)}, "
                "signals.destination_values) >= 2"
            )
        elif key == "recent_destination_search":
            conditions.append(
                "arrayExists(value -> value IN {destinations:Array(String)}, "
                "signals.destination_values)"
            )
        elif key == "booking_start_without_complete":
            conditions.append(
                "signals.booking_start_count > signals.booking_complete_count"
            )
        elif key == "benefit_interest":
            conditions.append(
                "((empty({benefit_keys:Array(String)}) AND "
                "signals.deal_count + signals.price_count + "
                "signals.free_cancellation_count + signals.breakfast_count > 0) OR "
                "(arrayExists(value -> value IN ('discount','early_booking'), "
                "{benefit_keys:Array(String)}) AND "
                "signals.deal_count + signals.price_count > 0) OR "
                "(has({benefit_keys:Array(String)}, 'free_cancellation') AND "
                "signals.free_cancellation_count > 0) OR "
                "(has({benefit_keys:Array(String)}, 'breakfast_included') AND "
                "signals.breakfast_count > 0))"
            )
        elif key == "promotion_response":
            conditions.append("signals.promotion_response_count > 0")
        elif key == "general_destination_exploration":
            conditions.append("signals.destination_count >= 2")
        elif key == "season_match":
            conditions.append(
                "arrayExists(value -> value IN {season_months:Array(UInt8)}, "
                "signals.checkin_months)"
            )
        else:
            raise ValueError(f"unsupported signal predicate: {key}")
    return " AND ".join(f"({value})" for value in conditions) or "1", parameters


def cohort_bound_batch_query(
    requests: Sequence[HardMatchAggregateRequest],
) -> tuple[str, dict[str, Any]]:
    query, parameters = _hard_predicate_batch_query(requests)
    return bind_query_to_cohort(query), parameters


def validate_cohort_bound_users(
    *,
    cohort_user_ids: Iterable[str],
    hard_match_user_ids: Iterable[str] = (),
    reference_sample_user_ids: Iterable[str] = (),
    filter_first_user_ids: Iterable[str] = (),
) -> None:
    cohort = {str(value) for value in cohort_user_ids}
    if not cohort:
        raise ValueError("cohort membership is empty")
    for label, values in (
        ("H", hard_match_user_ids),
        ("P", reference_sample_user_ids),
        ("filter_first", filter_first_user_ids),
    ):
        outsiders = sorted({str(value) for value in values} - cohort)
        if outsiders:
            raise ValueError(
                f"{label} contains users outside the cohort: {outsiders[:3]}"
            )


def validate_exact_result_sets(
    exact_all_user_ids: Iterable[str],
    filter_first_user_ids: Iterable[str],
) -> None:
    exact = {str(value) for value in exact_all_user_ids}
    filtered = {str(value) for value in filter_first_user_ids}
    if exact != filtered:
        missing = sorted(exact - filtered)[:3]
        extra = sorted(filtered - exact)[:3]
        raise ValueError(
            "filter_first_exact differs from exact_all: "
            f"missing={missing}, extra={extra}"
        )


def hard_match_bucket(value: float) -> HardMatchBucket:
    _validate_unit_ratio(value)
    if value <= 0.05:
        return HardMatchBucket.LE_005
    if value <= 0.20:
        return HardMatchBucket.GT_005_LE_020
    return HardMatchBucket.GT_020


def expected_member_bucket(value: float) -> ExpectedMemberBucket:
    _validate_unit_ratio(value)
    if value <= 0.01:
        return ExpectedMemberBucket.LE_001
    if value <= 0.05:
        return ExpectedMemberBucket.GT_001_LE_005
    if value <= 0.10:
        return ExpectedMemberBucket.GT_005_LE_010
    if value <= 0.25:
        return ExpectedMemberBucket.GT_010_LE_025
    return ExpectedMemberBucket.GT_025


def select_scenario_pairs(
    records: Sequence[ScenarioCensusRecord],
) -> tuple[ScenarioPair, ...]:
    groups: dict[
        tuple[str, HardMatchBucket, ExpectedMemberBucket],
        list[ScenarioCensusRecord],
    ] = {}
    for record in records:
        groups.setdefault(record.bucket_key, []).append(record)
    pairs: list[ScenarioPair] = []
    for key, group in sorted(
        groups.items(),
        key=lambda item: tuple(str(value) for value in item[0]),
    ):
        tuning = sorted(
            (item for item in group if item.scenario_set == ScenarioSet.TUNING),
            key=lambda item: (item.scenario_id, item.definition_sha256),
        )
        confirmation = sorted(
            (
                item
                for item in group
                if item.scenario_set == ScenarioSet.CONFIRMATION
            ),
            key=lambda item: (item.scenario_id, item.definition_sha256),
        )
        selected: tuple[ScenarioCensusRecord, ScenarioCensusRecord] | None = None
        for tuning_item in tuning:
            for confirmation_item in confirmation:
                if (
                    tuning_item.scenario_id != confirmation_item.scenario_id
                    and tuning_item.definition_sha256
                    != confirmation_item.definition_sha256
                ):
                    selected = (tuning_item, confirmation_item)
                    break
            if selected is not None:
                break
        candidate_type, hard_bucket, expected_bucket = key
        if selected is None:
            pairs.append(
                ScenarioPair(
                    candidate_type=candidate_type,
                    hard_match_bucket=hard_bucket,
                    expected_member_bucket=expected_bucket,
                    tuning_scenario_id=None,
                    confirmation_scenario_id=None,
                    validated=False,
                    fallback_reason="distinct tuning/confirmation pair unavailable",
                )
            )
        else:
            pairs.append(
                ScenarioPair(
                    candidate_type=candidate_type,
                    hard_match_bucket=hard_bucket,
                    expected_member_bucket=expected_bucket,
                    tuning_scenario_id=selected[0].scenario_id,
                    confirmation_scenario_id=selected[1].scenario_id,
                    validated=True,
                )
            )
    return tuple(pairs)


def confirmation_baseline_k(
    *,
    corpus_user_count: int,
    estimated_member_count: float,
) -> int:
    if corpus_user_count <= 0 or estimated_member_count < 0:
        raise ValueError("confirmation baseline inputs are invalid")
    return min(
        corpus_user_count,
        max(5_000, math.ceil(1.5 * estimated_member_count)),
    )


def select_survivors(
    candidates: Sequence[ScreeningCandidate],
) -> SurvivorSets:
    identities = [item.identity for item in candidates]
    if len(identities) != len(set(identities)):
        raise ValueError("screening candidates contain duplicate cells")
    scale = tuple(sorted(
        (item for item in candidates if item.scale_survivor),
        key=lambda item: item.identity,
    ))
    policy = tuple(sorted(
        (item for item in candidates if item.policy_survivor),
        key=lambda item: item.identity,
    ))
    union = {
        item.identity: item
        for item in (*scale, *policy)
    }
    goal2 = tuple(union[key] for key in sorted(union))
    if {item.identity for item in goal2} != (
        {item.identity for item in scale} | {item.identity for item in policy}
    ):
        raise RuntimeError("Goal 2 input is not the survivor union")
    return SurvivorSets(scale=scale, policy=policy, goal2_union=goal2)


def scale_manifest_template() -> Mapping[str, Any]:
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": "planned",
        "output_root": str(SCALE_OUTPUT_ROOT),
        "vector_version": SCALE_VECTOR_VERSION,
        "window_start": SCALE_WINDOW_START,
        "window_end": SCALE_WINDOW_END,
        "cohort_order": "SHA256(seed + '|' + user_id), user_id",
        "cohort_sizes": list(SCALE_COHORT_SIZES),
        "reference_sample_size": REFERENCE_SAMPLE_SIZE,
        "cohort_membership_relation": COHORT_MEMBERSHIP_RELATION,
        "candidate_policy_generated": False,
        "goal2_executed": False,
    }


def _validate_contiguous_membership(
    memberships: Sequence[CohortMembership],
) -> None:
    if len({item.scale_series_id for item in memberships}) != 1:
        raise ValueError("membership contains multiple scale series")
    if len({item.user_id for item in memberships}) != len(memberships):
        raise ValueError("membership contains duplicate users")
    expected = list(range(1, len(memberships) + 1))
    if [item.cohort_rank for item in memberships] != expected:
        raise ValueError("membership ranks are not contiguous")


def _validate_unit_ratio(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("ratio must be finite and within [0, 1]")


def _json_default(value: Any) -> Any:
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("fingerprint timestamps must be timezone-aware")
        return value.astimezone(UTC).isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, StrEnum):
        return value.value
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")
