from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.decision.experiment_assignment_repository import (
    AdExperimentUnitWrite,
    ExperimentAssignmentRepository,
    SegmentAssignmentExecutionWrite,
)


NOW = datetime(2026, 7, 21, 12, tzinfo=UTC)


class _Db:
    def __init__(self, *, fetchone_rows=(), fetchall_rows=()) -> None:
        self.fetchone_rows = list(fetchone_rows)
        self.fetchall_rows = list(fetchall_rows)
        self.calls = []

    def fetchone(self, query, params=()):
        self.calls.append(("fetchone", query, params))
        return self.fetchone_rows.pop(0) if self.fetchone_rows else None

    def fetchall(self, query, params=()):
        self.calls.append(("fetchall", query, params))
        return self.fetchall_rows.pop(0) if self.fetchall_rows else []

    def execute(self, query, params=()):
        self.calls.append(("execute", query, params))


def test_repository_uses_database_clock_for_assignment_timestamp() -> None:
    repository = ExperimentAssignmentRepository(
        _Db(fetchone_rows=({"assigned_at": NOW},))
    )

    assert repository.database_clock() == NOW


def test_repository_persists_execution_design_manifest() -> None:
    execution = execution_write()
    returned_row = {
        "segment_assignment_execution_id": execution.segment_assignment_execution_id,
        "promotion_run_id": execution.promotion_run_id,
        "request_fingerprint": execution.request_fingerprint,
        "input_fingerprint": execution.input_fingerprint,
        "matcher_strategy": execution.matcher_strategy,
        "matcher_version": execution.matcher_version,
        "vector_version": execution.vector_version,
        "source_cutoff_at": execution.source_cutoff_at,
        "input_manifest_json": execution.input_manifest_json,
    }
    db = _Db(fetchone_rows=(returned_row,))
    repository = ExperimentAssignmentRepository(db)

    record = repository.insert_execution(execution)

    assert record.input_manifest_json["experiment_design_fingerprint"] == "d" * 64
    operation, query, params = db.calls[0]
    assert operation == "fetchone"
    assert "INSERT INTO segment_assignment_executions" in query
    assert params[-1] == execution.input_manifest_json


def test_repository_bulk_inserts_all_experiment_units() -> None:
    db = _Db()
    repository = ExperimentAssignmentRepository(db)
    units = [unit_write("user_a", "treatment"), unit_write("user_b", "control")]

    repository.insert_units(units)

    operation, query, params = db.calls[0]
    assert operation == "execute"
    assert "INSERT INTO ad_experiment_units" in query
    assert params[8] == ["user_a", "user_b"]
    assert params[9] == ["treatment", "control"]
    assert params[11] == [NOW, NOW]
    assert params[12] == [NOW, NOW]


def execution_write() -> SegmentAssignmentExecutionWrite:
    return SegmentAssignmentExecutionWrite(
        segment_assignment_execution_id="execution",
        promotion_run_id="run",
        request_fingerprint="a" * 64,
        input_fingerprint="b" * 64,
        matcher_strategy="analysis_snapshot_complete_randomization",
        matcher_version="uplift-ready-assignment.v1",
        vector_version="hotel_behavior.v2",
        source_cutoff_at=NOW - timedelta(hours=1),
        input_manifest_json={
            "schema_version": "segment-assignment-execution.v2",
            "experiment_design_fingerprint": "d" * 64,
        },
    )


def unit_write(user_id: str, arm: str) -> AdExperimentUnitWrite:
    return AdExperimentUnitWrite(
        experiment_unit_id=f"unit_{user_id}",
        project_id="project",
        promotion_run_id="run",
        ad_experiment_id="experiment",
        segment_id="segment",
        audience_snapshot_id="snapshot",
        vector_generation_id="generation",
        segment_assignment_execution_id="execution",
        user_id=user_id,
        arm=arm,
        treatment_probability=Decimal("0.5"),
        assigned_at=NOW,
        outcome_window_start=NOW,
        outcome_window_end=NOW + timedelta(days=30),
    )
