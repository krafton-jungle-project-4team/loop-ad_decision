from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
import math
from typing import Any, Mapping, Protocol, Sequence

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
    execution_manifest: Mapping[str, Any]
    goal_snapshot_json: Mapping[str, Any]
    feature_vector: Any | None


class UpliftUnitSourceReader(Protocol):
    def list_units(
        self,
        *,
        project_id: str | None = None,
    ) -> list[UpliftUnitSourceRecord]:
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

    def list_units(
        self,
        *,
        project_id: str | None = None,
    ) -> list[UpliftUnitSourceRecord]:
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
            WHERE (%s::text IS NULL OR unit.project_id = %s)
              AND execution.uplift_assignment_status = 'finalized'
            ORDER BY
                unit.promotion_run_id ASC,
                unit.ad_experiment_id ASC,
                unit.user_id ASC
            """,
            (project_id, project_id),
        )
        return [UpliftUnitSourceRecord(**row) for row in rows]


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
    ) -> None:
        self._unit_reader = unit_reader
        self._outcome_reader = outcome_reader

    def build(
        self,
        *,
        reference_time: datetime,
        project_id: str | None = None,
    ) -> UpliftDatasetBuildResult:
        source_units = self._unit_reader.list_units(project_id=project_id)
        excluded: Counter[str] = Counter()
        prepared: list[tuple[UpliftUnitSourceRecord, tuple[float, ...], dict[str, Any]]] = []
        for unit in source_units:
            preparation = _prepare_unit(unit, reference_time=reference_time)
            if isinstance(preparation, str):
                excluded[preparation] += 1
                continue
            features, outcome_spec = preparation
            prepared.append((unit, features, outcome_spec))

        grouped: dict[
            tuple[str, str, datetime, datetime, tuple[str, ...]],
            list[tuple[UpliftUnitSourceRecord, tuple[float, ...]]],
        ] = defaultdict(list)
        for unit, features, outcome_spec in prepared:
            raw_filter = outcome_spec.get("outcome_filter")
            destination_ids = tuple(
                str(value)
                for value in (
                    raw_filter.get("destination_ids", [])
                    if isinstance(raw_filter, Mapping)
                    else []
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
            ].append((unit, features))

        examples: list[UpliftTrainingExample] = []
        for key, group in grouped.items():
            project, event_name, window_start, window_end, destinations = key
            success_user_ids = self._outcome_reader.list_success_user_ids(
                project_id=project,
                user_ids=[unit.user_id for unit, _features in group],
                event_name=event_name,
                window_start=window_start,
                window_end=window_end,
                destination_ids=destinations,
            )
            for unit, features in group:
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
                    )
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
            source_unit_count=len(source_units),
        )


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

