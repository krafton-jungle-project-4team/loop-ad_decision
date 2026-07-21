from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Any, Mapping, Protocol, Sequence

from app.decision.repositories import PostgresExecutor


@dataclass(frozen=True, slots=True)
class SegmentAssignmentExecutionWrite:
    segment_assignment_execution_id: str
    promotion_run_id: str
    request_fingerprint: str
    input_fingerprint: str
    matcher_strategy: str
    matcher_version: str
    vector_version: str
    source_cutoff_at: datetime
    input_manifest_json: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class SegmentAssignmentExecutionRecord:
    segment_assignment_execution_id: str
    promotion_run_id: str
    request_fingerprint: str
    input_fingerprint: str
    matcher_strategy: str
    matcher_version: str
    vector_version: str
    source_cutoff_at: datetime
    input_manifest_json: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class AdExperimentUnitWrite:
    experiment_unit_id: str
    project_id: str
    promotion_run_id: str
    ad_experiment_id: str
    segment_id: str
    audience_snapshot_id: str
    vector_generation_id: str
    segment_assignment_execution_id: str
    user_id: str
    arm: str
    treatment_probability: Decimal
    assigned_at: datetime
    outcome_window_start: datetime
    outcome_window_end: datetime


@dataclass(frozen=True, slots=True)
class AdExperimentUnitRecord:
    experiment_unit_id: str
    project_id: str
    promotion_run_id: str
    ad_experiment_id: str
    segment_id: str
    audience_snapshot_id: str
    vector_generation_id: str
    segment_assignment_execution_id: str
    user_id: str
    arm: str
    treatment_probability: Decimal
    assigned_at: datetime
    outcome_window_start: datetime
    outcome_window_end: datetime


class ExperimentAssignmentWriter(Protocol):
    def database_clock(self) -> datetime:
        ...

    def list_uplift_ready_executions(
        self,
        promotion_run_id: str,
    ) -> list[SegmentAssignmentExecutionRecord]:
        ...

    def insert_execution(
        self,
        execution: SegmentAssignmentExecutionWrite,
    ) -> SegmentAssignmentExecutionRecord:
        ...

    def insert_units(self, units: Sequence[AdExperimentUnitWrite]) -> None:
        ...

    def finalize_execution(
        self,
        segment_assignment_execution_id: str,
    ) -> None:
        ...

    def list_units_by_execution(
        self,
        segment_assignment_execution_id: str,
    ) -> list[AdExperimentUnitRecord]:
        ...


class ExperimentAssignmentRepository:
    INSERT_BATCH_SIZE = 1000

    def __init__(self, db: PostgresExecutor) -> None:
        self._db = db

    def database_clock(self) -> datetime:
        row = self._db.fetchone("SELECT clock_timestamp() AS assigned_at")
        if row is None or not isinstance(row.get("assigned_at"), datetime):
            raise RuntimeError("database clock_timestamp() returned no timestamp")
        return row["assigned_at"]

    def list_uplift_ready_executions(
        self,
        promotion_run_id: str,
    ) -> list[SegmentAssignmentExecutionRecord]:
        rows = self._db.fetchall(
            """
            SELECT
                segment_assignment_execution_id,
                promotion_run_id,
                request_fingerprint,
                input_fingerprint,
                matcher_strategy,
                matcher_version,
                vector_version,
                source_cutoff_at,
                input_manifest_json
            FROM segment_assignment_executions
            WHERE promotion_run_id = %s
              AND input_manifest_json->>'schema_version' =
                  'segment-assignment-execution.v2'
              AND uplift_assignment_status = 'finalized'
            ORDER BY created_at ASC, segment_assignment_execution_id ASC
            """,
            (promotion_run_id,),
        )
        return [SegmentAssignmentExecutionRecord(**row) for row in rows]

    def insert_execution(
        self,
        execution: SegmentAssignmentExecutionWrite,
    ) -> SegmentAssignmentExecutionRecord:
        row = self._db.fetchone(
            """
            INSERT INTO segment_assignment_executions (
                segment_assignment_execution_id,
                promotion_run_id,
                request_fingerprint,
                input_fingerprint,
                matcher_strategy,
                matcher_version,
                vector_version,
                source_cutoff_at,
                input_manifest_json,
                uplift_assignment_status
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, 'preparing')
            RETURNING
                segment_assignment_execution_id,
                promotion_run_id,
                request_fingerprint,
                input_fingerprint,
                matcher_strategy,
                matcher_version,
                vector_version,
                source_cutoff_at,
                input_manifest_json
            """,
            (
                execution.segment_assignment_execution_id,
                execution.promotion_run_id,
                execution.request_fingerprint,
                execution.input_fingerprint,
                execution.matcher_strategy,
                execution.matcher_version,
                execution.vector_version,
                execution.source_cutoff_at,
                execution.input_manifest_json,
            ),
        )
        if row is None:
            raise RuntimeError("assignment execution was not inserted")
        return SegmentAssignmentExecutionRecord(**row)

    def finalize_execution(
        self,
        segment_assignment_execution_id: str,
    ) -> None:
        self._db.execute(
            "SELECT finalize_uplift_assignment_execution(%s)",
            (segment_assignment_execution_id,),
        )

    def insert_units(self, units: Sequence[AdExperimentUnitWrite]) -> None:
        for chunk_start in range(0, len(units), self.INSERT_BATCH_SIZE):
            chunk = units[chunk_start : chunk_start + self.INSERT_BATCH_SIZE]
            self._db.execute(
                """
                INSERT INTO ad_experiment_units (
                    experiment_unit_id,
                    project_id,
                    promotion_run_id,
                    ad_experiment_id,
                    segment_id,
                    audience_snapshot_id,
                    vector_generation_id,
                    segment_assignment_execution_id,
                    user_id,
                    arm,
                    treatment_probability,
                    assigned_at,
                    outcome_window_start,
                    outcome_window_end
                )
                SELECT *
                FROM unnest(
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::text[],
                    %s::numeric[],
                    %s::timestamptz[],
                    %s::timestamptz[],
                    %s::timestamptz[]
                )
                """,
                (
                    [unit.experiment_unit_id for unit in chunk],
                    [unit.project_id for unit in chunk],
                    [unit.promotion_run_id for unit in chunk],
                    [unit.ad_experiment_id for unit in chunk],
                    [unit.segment_id for unit in chunk],
                    [unit.audience_snapshot_id for unit in chunk],
                    [unit.vector_generation_id for unit in chunk],
                    [unit.segment_assignment_execution_id for unit in chunk],
                    [unit.user_id for unit in chunk],
                    [unit.arm for unit in chunk],
                    [unit.treatment_probability for unit in chunk],
                    [unit.assigned_at for unit in chunk],
                    [unit.outcome_window_start for unit in chunk],
                    [unit.outcome_window_end for unit in chunk],
                ),
            )

    def list_units_by_execution(
        self,
        segment_assignment_execution_id: str,
    ) -> list[AdExperimentUnitRecord]:
        rows = self._db.fetchall(
            """
            SELECT
                experiment_unit_id,
                project_id,
                promotion_run_id,
                ad_experiment_id,
                segment_id,
                audience_snapshot_id,
                vector_generation_id,
                segment_assignment_execution_id,
                user_id,
                arm,
                treatment_probability,
                assigned_at,
                outcome_window_start,
                outcome_window_end
            FROM ad_experiment_units
            WHERE segment_assignment_execution_id = %s
            ORDER BY user_id ASC
            """,
            (segment_assignment_execution_id,),
        )
        return [AdExperimentUnitRecord(**row) for row in rows]
