from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
import hashlib
import json
import math
from typing import Any, Iterable, Iterator, Mapping, Protocol, Sequence

from app.analysis.behavior_manifest import (
    canonical_destination_id,
    clickhouse_canonical_destination_sql,
)
from app.decision.experiment_design import EXECUTION_SCHEMA_VERSION
from app.decision.matcher import parse_vector_values
from app.decision.outcome_spec import (
    BOOKING_CONVERSION_RATE,
    outcome_spec_hash,
)
from app.decision.repositories import ClickHouseClient, PostgresExecutor
from app.uplift.contracts import UpliftDatasetBuildResult, UpliftTrainingExample


DEFAULT_OUTCOME_USER_BATCH_SIZE = 5000
DEFAULT_POSTGRES_UNIT_PAGE_SIZE = 5000
DATASET_MANIFEST_VERSION = "uplift-dataset-manifest.v1"
FEATURE_CONTRACT_VERSION = "uplift-feature-contract.v1"
OUTCOME_CONTRACT_VERSION = "uplift-outcome-contract.v1"
TRAINING_CODE_VERSION = "uplift-training.v1"
MODEL_TYPE = "transformed-outcome-ridge.v1"
SPLIT_POLICY_VERSION = "experiment-time-holdout.v1"
_OUTCOME_DESTINATION_PROPERTY_KEYS = (
    "destination_id",
    "destination_name",
    "hotel_city",
    "hotel_country",
)


@dataclass(frozen=True, slots=True)
class UpliftUnitSourceRecord:
    experiment_unit_id: str
    project_id: str
    promotion_run_id: str
    ad_experiment_id: str
    segment_id: str
    audience_snapshot_id: str
    vector_generation_id: str
    user_id: str
    arm: str
    treatment_probability: Decimal
    assigned_at: datetime
    outcome_window_start: datetime
    outcome_window_end: datetime
    execution_source_cutoff_at: datetime
    generation_window_end: datetime
    generation_source_revision_cutoff: datetime
    audience_source_cutoff: datetime
    generation_vector_version: str
    generation_manifest_hash: str
    execution_manifest: Mapping[str, Any]
    goal_snapshot_json: Mapping[str, Any]
    feature_vector: Any | None


class UpliftUnitSourceReader(Protocol):
    def iter_unit_pages(
        self,
        *,
        project_id: str,
        reference_time: datetime,
        after_promotion_run_id: str | None = None,
        after_ad_experiment_id: str | None = None,
        after_user_id: str | None = None,
        page_size: int = DEFAULT_POSTGRES_UNIT_PAGE_SIZE,
    ) -> Iterator[tuple[UpliftUnitSourceRecord, ...]]:
        ...


class OutcomeEventReader(Protocol):
    def list_success_user_ids(
        self,
        *,
        project_id: str,
        user_ids: Sequence[str],
        event_name: str,
        window_start: datetime,
        window_end: datetime,
        destination_ids: Sequence[str],
    ) -> set[str]:
        ...


class PostgresUpliftUnitSourceRepository:
    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def iter_unit_pages(
        self,
        *,
        project_id: str,
        reference_time: datetime,
        after_promotion_run_id: str | None = None,
        after_ad_experiment_id: str | None = None,
        after_user_id: str | None = None,
        page_size: int = DEFAULT_POSTGRES_UNIT_PAGE_SIZE,
    ) -> Iterator[tuple[UpliftUnitSourceRecord, ...]]:
        if not project_id.strip():
            raise ValueError("project_id is required")
        if page_size < 1:
            raise ValueError("unit page size must be positive")
        cursor = (
            after_promotion_run_id,
            after_ad_experiment_id,
            after_user_id,
        )
        if any(value is not None for value in cursor) and not all(
            value is not None for value in cursor
        ):
            raise ValueError("all unit cursor fields must be provided together")

        while True:
            rows = self._db.fetchall(
                """
                SELECT
                    unit.experiment_unit_id,
                    unit.project_id,
                    unit.promotion_run_id,
                    unit.ad_experiment_id,
                    unit.segment_id,
                    unit.audience_snapshot_id,
                    unit.vector_generation_id,
                    unit.user_id,
                    unit.arm,
                    unit.treatment_probability,
                    unit.assigned_at,
                    unit.outcome_window_start,
                    unit.outcome_window_end,
                    execution.source_cutoff_at AS execution_source_cutoff_at,
                    generation.window_end AS generation_window_end,
                    generation.source_revision_cutoff
                        AS generation_source_revision_cutoff,
                    snapshot.source_cutoff AS audience_source_cutoff,
                    generation.vector_version AS generation_vector_version,
                    generation.manifest_hash AS generation_manifest_hash,
                    execution.input_manifest_json AS execution_manifest,
                    run.goal_snapshot_json,
                    vector.embedding::text AS feature_vector
                FROM ad_experiment_units AS unit
                JOIN segment_assignment_executions AS execution
                  ON execution.promotion_run_id = unit.promotion_run_id
                 AND execution.segment_assignment_execution_id =
                     unit.segment_assignment_execution_id
                JOIN promotion_runs AS run
                  ON run.promotion_run_id = unit.promotion_run_id
                JOIN user_behavior_vector_search_generations AS generation
                  ON generation.vector_generation_id = unit.vector_generation_id
                JOIN segment_audience_snapshots AS snapshot
                  ON snapshot.snapshot_id = unit.audience_snapshot_id
                LEFT JOIN user_behavior_vector_search AS vector
                  ON vector.vector_generation_id = unit.vector_generation_id
                 AND vector.user_id = unit.user_id
                WHERE unit.project_id = %s
                  AND unit.outcome_window_end <= %s
                  AND execution.uplift_assignment_status = 'finalized'
                  AND execution.input_manifest_json
                      ->'experiment_design'->>'mode' = 'randomized_holdout'
                  AND execution.input_manifest_json
                      ->'outcome_spec'->>'uplift_training_eligible' = 'true'
                  AND (
                      %s::text IS NULL
                      OR (
                          unit.promotion_run_id,
                          unit.ad_experiment_id,
                          unit.user_id
                      ) > (%s, %s, %s)
                  )
                ORDER BY
                    unit.promotion_run_id ASC,
                    unit.ad_experiment_id ASC,
                    unit.user_id ASC
                LIMIT %s
                """,
                (
                    project_id,
                    reference_time,
                    cursor[0],
                    cursor[0],
                    cursor[1],
                    cursor[2],
                    page_size,
                ),
            )
            if not rows:
                return
            page = tuple(UpliftUnitSourceRecord(**row) for row in rows)
            yield page
            if len(page) < page_size:
                return
            last = page[-1]
            cursor = (
                last.promotion_run_id,
                last.ad_experiment_id,
                last.user_id,
            )


class ClickHouseOutcomeEventRepository:
    def __init__(
        self,
        client: ClickHouseClient,
        *,
        user_batch_size: int = DEFAULT_OUTCOME_USER_BATCH_SIZE,
    ) -> None:
        if user_batch_size < 1:
            raise ValueError("outcome user batch size must be positive")
        self._client = client
        self._user_batch_size = user_batch_size

    def list_success_user_ids(
        self,
        *,
        project_id: str,
        user_ids: Sequence[str],
        event_name: str,
        window_start: datetime,
        window_end: datetime,
        destination_ids: Sequence[str],
    ) -> set[str]:
        if not user_ids:
            return set()
        canonical_destination_ids = tuple(
            dict.fromkeys(
                canonical
                for value in destination_ids
                if (canonical := canonical_destination_id(value))
            )
        )
        destination_clause = ""
        if canonical_destination_ids:
            destination_predicates = [
                "(" + clickhouse_canonical_destination_sql(
                    "JSONExtractString(properties_json, "
                    f"'{property_key}')"
                ) + ") IN {destination_ids:Array(String)}"
                for property_key in _OUTCOME_DESTINATION_PROPERTY_KEYS
            ]
            destination_clause = " AND (\n" + " OR\n".join(
                destination_predicates
            ) + "\n)"

        unique_user_ids = tuple(dict.fromkeys(str(value) for value in user_ids))
        success_user_ids: set[str] = set()
        for start in range(0, len(unique_user_ids), self._user_batch_size):
            batch = unique_user_ids[start : start + self._user_batch_size]
            result = self._client.query(
                f"""
                SELECT DISTINCT user_id
                FROM raw_events
                WHERE project_id = {{project_id:String}}
                  AND event_name = {{event_name:String}}
                  AND validation_status = 'valid'
                  AND user_id IN {{user_ids:Array(String)}}
                  AND event_time >= {{window_start:DateTime64(3, 'UTC')}}
                  AND event_time < {{window_end:DateTime64(3, 'UTC')}}
                  {destination_clause}
                """,
                parameters={
                    "project_id": project_id,
                    "event_name": event_name,
                    "user_ids": list(batch),
                    "window_start": window_start,
                    "window_end": window_end,
                    "destination_ids": list(canonical_destination_ids),
                },
            )
            success_user_ids.update(str(row[0]) for row in result.result_rows)
        return success_user_ids


class UpliftDatasetBuilder:
    def __init__(
        self,
        *,
        unit_reader: UpliftUnitSourceReader,
        outcome_reader: OutcomeEventReader,
        unit_page_size: int = DEFAULT_POSTGRES_UNIT_PAGE_SIZE,
    ) -> None:
        if unit_page_size < 1:
            raise ValueError("unit page size must be positive")
        self._unit_reader = unit_reader
        self._outcome_reader = outcome_reader
        self._unit_page_size = unit_page_size

    def build(
        self,
        *,
        reference_time: datetime,
        project_id: str,
    ) -> UpliftDatasetBuildResult:
        excluded: Counter[str] = Counter()
        examples: list[UpliftTrainingExample] = []
        source_unit_count = 0
        for page in self._unit_reader.iter_unit_pages(
            project_id=project_id,
            reference_time=reference_time,
            page_size=self._unit_page_size,
        ):
            source_unit_count += len(page)
            prepared: list[
                tuple[UpliftUnitSourceRecord, tuple[float, ...], dict[str, Any]]
            ] = []
            for unit in page:
                preparation = _prepare_unit(unit, reference_time=reference_time)
                if isinstance(preparation, str):
                    excluded[preparation] += 1
                    continue
                features, outcome_spec = preparation
                prepared.append((unit, features, outcome_spec))
            examples.extend(
                self._build_page_examples(prepared)
            )
        examples.sort(
            key=lambda item: (
                item.promotion_run_id,
                item.ad_experiment_id,
                item.user_id,
            )
        )
        return UpliftDatasetBuildResult(
            examples=tuple(examples),
            excluded_reason_counts=dict(sorted(excluded.items())),
            source_unit_count=source_unit_count,
        )

    def _build_page_examples(
        self,
        prepared: Sequence[
            tuple[UpliftUnitSourceRecord, tuple[float, ...], Mapping[str, Any]]
        ],
    ) -> list[UpliftTrainingExample]:
        grouped: dict[
            tuple[str, str, datetime, datetime, tuple[str, ...]],
            list[
                tuple[
                    UpliftUnitSourceRecord,
                    tuple[float, ...],
                    Mapping[str, Any],
                ]
            ],
        ] = defaultdict(list)
        for unit, features, outcome_spec in prepared:
            raw_filter = outcome_spec.get("outcome_filter")
            destination_ids = tuple(
                sorted(
                    dict.fromkeys(
                        canonical
                        for value in (
                            raw_filter.get("destination_ids", [])
                            if isinstance(raw_filter, Mapping)
                            else []
                        )
                        if (canonical := canonical_destination_id(str(value)))
                    )
                )
            )
            grouped[
                (
                    unit.project_id,
                    str(outcome_spec["outcome_event_name"]),
                    unit.outcome_window_start,
                    unit.outcome_window_end,
                    destination_ids,
                )
            ].append((unit, features, outcome_spec))

        examples: list[UpliftTrainingExample] = []
        for key in sorted(grouped, key=_outcome_group_sort_key):
            group = grouped[key]
            project, event_name, window_start, window_end, destinations = key
            success_user_ids = self._outcome_reader.list_success_user_ids(
                project_id=project,
                user_ids=[unit.user_id for unit, _features, _spec in group],
                event_name=event_name,
                window_start=window_start,
                window_end=window_end,
                destination_ids=destinations,
            )
            for unit, features, outcome_spec in group:
                spec_hash = outcome_spec_hash(outcome_spec)
                examples.append(
                    UpliftTrainingExample(
                        experiment_unit_id=unit.experiment_unit_id,
                        project_id=unit.project_id,
                        promotion_run_id=unit.promotion_run_id,
                        ad_experiment_id=unit.ad_experiment_id,
                        segment_id=unit.segment_id,
                        user_id=unit.user_id,
                        audience_snapshot_id=unit.audience_snapshot_id,
                        vector_generation_id=unit.vector_generation_id,
                        features=features,
                        treatment=1 if unit.arm == "treatment" else 0,
                        outcome=1 if unit.user_id in success_user_ids else 0,
                        treatment_probability=float(unit.treatment_probability),
                        assigned_at=unit.assigned_at,
                        outcome_window_start=unit.outcome_window_start,
                        outcome_window_end=unit.outcome_window_end,
                        vector_version=unit.generation_vector_version,
                        feature_contract_hash=_feature_contract_hash(
                            vector_version=unit.generation_vector_version,
                            manifest_hash=unit.generation_manifest_hash,
                            feature_dimension=len(features),
                        ),
                        outcome_spec_hash=spec_hash,
                        outcome_contract_hash=_outcome_contract_hash(outcome_spec),
                    )
                )
        return examples


def _prepare_unit(
    unit: UpliftUnitSourceRecord,
    *,
    reference_time: datetime,
) -> tuple[tuple[float, ...], dict[str, Any]] | str:
    if unit.outcome_window_end > reference_time:
        return "outcome_window_not_ended"
    manifest = unit.execution_manifest
    if manifest.get("schema_version") != EXECUTION_SCHEMA_VERSION:
        return "legacy_execution"
    outcome_spec = manifest.get("outcome_spec")
    design = manifest.get("experiment_design")
    run_spec = unit.goal_snapshot_json.get("outcome_spec")
    run_hash = unit.goal_snapshot_json.get("outcome_spec_hash")
    if (
        not isinstance(outcome_spec, Mapping)
        or not isinstance(design, Mapping)
        or not isinstance(run_spec, Mapping)
        or not isinstance(run_hash, str)
        or outcome_spec != run_spec
        or design.get("outcome_spec_hash") != run_hash
        or outcome_spec_hash(outcome_spec) != run_hash
    ):
        return "outcome_spec_mismatch"
    if design.get("mode") != "randomized_holdout":
        return "non_randomized_execution"
    if (
        outcome_spec.get("outcome_metric") != BOOKING_CONVERSION_RATE
        or outcome_spec.get("outcome_event_name") != "booking_complete"
        or outcome_spec.get("uplift_training_eligible") is not True
    ):
        return "unsupported_goal_metric"
    if unit.feature_vector is None:
        return "feature_snapshot_missing"
    if (
        not unit.generation_vector_version.strip()
        or len(unit.generation_manifest_hash) != 64
    ):
        return "feature_contract_unsupported"
    if (
        unit.generation_window_end > unit.assigned_at
        or unit.generation_source_revision_cutoff > unit.assigned_at
        or unit.audience_source_cutoff > unit.assigned_at
        or unit.execution_source_cutoff_at > unit.assigned_at
        or unit.outcome_window_start != unit.assigned_at
        or unit.outcome_window_end <= unit.outcome_window_start
    ):
        return "assignment_time_contract_invalid"
    try:
        features = tuple(parse_vector_values(unit.feature_vector))
    except (TypeError, ValueError):
        return "feature_snapshot_invalid"
    if len(features) != 64 or not all(math.isfinite(value) for value in features):
        return "feature_snapshot_invalid"
    return features, dict(outcome_spec)


class UpliftDatasetCompatibilityError(ValueError):
    pass


def build_dataset_manifest(
    result: UpliftDatasetBuildResult,
    *,
    project_id: str,
    reference_time: datetime,
    training_code_version: str = TRAINING_CODE_VERSION,
    model_type: str = MODEL_TYPE,
    split_policy_version: str = SPLIT_POLICY_VERSION,
) -> tuple[dict[str, Any], str]:
    examples = result.examples
    if not examples:
        raise UpliftDatasetCompatibilityError(
            "uplift dataset contains no eligible examples"
        )
    if not project_id.strip() or any(
        example.project_id != project_id for example in examples
    ):
        raise UpliftDatasetCompatibilityError(
            "uplift dataset must contain exactly one project"
        )
    feature_contract_hashes = _required_single_hash(
        (example.feature_contract_hash for example in examples),
        label="feature contract",
    )
    outcome_contract_hashes = _required_single_hash(
        (example.outcome_contract_hash for example in examples),
        label="outcome contract",
    )
    unit_ids = sorted(example.experiment_unit_id for example in examples)
    if len(unit_ids) != len(set(unit_ids)):
        raise UpliftDatasetCompatibilityError(
            "uplift dataset contains duplicate experiment unit ids"
        )
    manifest = {
        "schema_version": DATASET_MANIFEST_VERSION,
        "project_id": project_id,
        "reference_time": _canonical_datetime(reference_time),
        "experiment_ids": sorted(
            {example.ad_experiment_id for example in examples}
        ),
        "experiment_unit_set_hash": _sha256_json(unit_ids),
        "example_content_hash": _sha256_json(
            [
                {
                    "experiment_unit_id": example.experiment_unit_id,
                    "treatment": example.treatment,
                    "outcome": example.outcome,
                    "treatment_probability": example.treatment_probability,
                    "feature_hash": _sha256_json(list(example.features)),
                    "assigned_at": (
                        _canonical_datetime(example.assigned_at)
                        if example.assigned_at is not None
                        else None
                    ),
                    "outcome_window_end": (
                        _canonical_datetime(example.outcome_window_end)
                        if example.outcome_window_end is not None
                        else None
                    ),
                }
                for example in sorted(
                    examples,
                    key=lambda item: item.experiment_unit_id,
                )
            ]
        ),
        "experiment_unit_count": len(unit_ids),
        "outcome_contract_hash": outcome_contract_hashes,
        "outcome_spec_hashes": sorted(
            {
                str(example.outcome_spec_hash)
                for example in examples
                if example.outcome_spec_hash
            }
        ),
        "feature_contract_hash": feature_contract_hashes,
        "vector_versions": sorted(
            {
                str(example.vector_version)
                for example in examples
                if example.vector_version
            }
        ),
        "vector_generation_ids": sorted(
            {example.vector_generation_id for example in examples}
        ),
        "training_code_version": training_code_version,
        "model_type": model_type,
        "split_policy_version": split_policy_version,
    }
    return manifest, _sha256_json(manifest)


def _feature_contract_hash(
    *,
    vector_version: str,
    manifest_hash: str,
    feature_dimension: int,
) -> str:
    return _sha256_json(
        {
            "contract_version": FEATURE_CONTRACT_VERSION,
            "vector_version": vector_version,
            "vector_manifest_hash": manifest_hash,
            "feature_dimension": feature_dimension,
        }
    )


def _outcome_contract_hash(outcome_spec: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "contract_version": OUTCOME_CONTRACT_VERSION,
            "outcome_metric": outcome_spec.get("outcome_metric"),
            "outcome_event_name": outcome_spec.get("outcome_event_name"),
            "outcome_definition_version": outcome_spec.get(
                "outcome_definition_version"
            ),
            "filter_contract": {
                "destination_ids": "frozen_unit_allowlist",
                "canonical_fields": list(_OUTCOME_DESTINATION_PROPERTY_KEYS),
                "match_operator": "any",
            },
            "window_contract": "assigned_at_inclusive_outcome_end_exclusive",
        }
    )


def _required_single_hash(
    values: Iterable[str | None],
    *,
    label: str,
) -> str:
    normalized = {str(value) for value in values if value}
    if len(normalized) != 1:
        raise UpliftDatasetCompatibilityError(
            f"uplift dataset requires one compatible {label}"
        )
    value = next(iter(normalized))
    if len(value) != 64:
        raise UpliftDatasetCompatibilityError(
            f"uplift dataset {label} hash is invalid"
        )
    return value


def _canonical_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise UpliftDatasetCompatibilityError(
            "uplift dataset reference_time must include a timezone"
        )
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _sha256_json(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _outcome_group_sort_key(
    value: tuple[str, str, datetime, datetime, tuple[str, ...]],
) -> tuple[str, str, str, str, tuple[str, ...]]:
    return (
        value[0],
        value[1],
        value[2].isoformat(),
        value[3].isoformat(),
        value[4],
    )
