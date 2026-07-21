import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.decision.experiment_design import EXECUTION_SCHEMA_VERSION
from app.decision.outcome_spec import build_frozen_outcome_spec
from app.uplift.dataset import (
    ClickHouseOutcomeEventRepository,
    UpliftDatasetBuilder,
    UpliftUnitSourceRecord,
)


ASSIGNED_AT = datetime(2026, 7, 1, tzinfo=UTC)
WINDOW_END = ASSIGNED_AT + timedelta(days=30)
CUTOFF = ASSIGNED_AT - timedelta(minutes=1)


class _UnitReader:
    def __init__(self, records):
        self.records = records

    def list_units(self, *, project_id=None):
        return [
            record
            for record in self.records
            if project_id is None or record.project_id == project_id
        ]


class _OutcomeReader:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def list_success_user_ids(
        self,
        *,
        project_id,
        user_ids,
        event_name,
        window_start,
        window_end,
        destination_ids,
    ):
        self.calls.append(
            {
                "project_id": project_id,
                "user_ids": user_ids,
                "event_name": event_name,
                "window_start": window_start,
                "window_end": window_end,
                "destination_ids": destination_ids,
            }
        )
        return {
            user_id
            for user_id, event, destination, event_time in self.events
            if user_id in user_ids
            and event == event_name
            and window_start <= event_time < window_end
            and (not destination_ids or destination in destination_ids)
        }


def test_dataset_uses_frozen_destination_specific_booking_outcome() -> None:
    outcome_reader = _OutcomeReader(
        [
            ("jeju_user", "booking_complete", "jeju", ASSIGNED_AT + timedelta(days=1)),
            ("seoul_user", "booking_complete", "seoul", ASSIGNED_AT + timedelta(days=1)),
            ("late_user", "booking_complete", "jeju", WINDOW_END),
        ]
    )
    builder = UpliftDatasetBuilder(
        unit_reader=_UnitReader(
            [source_record("jeju_user"), source_record("seoul_user"), source_record("late_user")]
        ),
        outcome_reader=outcome_reader,
    )

    result = builder.build(reference_time=WINDOW_END)

    assert {example.user_id: example.outcome for example in result.examples} == {
        "jeju_user": 1,
        "late_user": 0,
        "seoul_user": 0,
    }
    assert outcome_reader.calls[0]["destination_ids"] == ("jeju", "okinawa")


def test_dataset_excludes_unfinished_missing_invalid_and_unsupported_units() -> None:
    unfinished = source_record("unfinished", outcome_window_end=WINDOW_END + timedelta(days=1))
    missing = source_record("missing", feature_vector=None)
    invalid_time = source_record(
        "invalid_time",
        generation_window_end=ASSIGNED_AT + timedelta(seconds=1),
    )
    unsupported = source_record("unsupported")
    unsupported_spec, unsupported_hash = build_frozen_outcome_spec(
        goal_metric="inflow_rate",
        target_segment_rules=[],
    )
    unsupported = replace_outcome(unsupported, unsupported_spec, unsupported_hash)
    builder = UpliftDatasetBuilder(
        unit_reader=_UnitReader([unfinished, missing, invalid_time, unsupported]),
        outcome_reader=_OutcomeReader([]),
    )

    result = builder.build(reference_time=WINDOW_END)

    assert result.examples == ()
    assert result.excluded_reason_counts == {
        "assignment_time_contract_invalid": 1,
        "feature_snapshot_missing": 1,
        "outcome_window_not_ended": 1,
        "unsupported_goal_metric": 1,
    }


def test_dataset_rejects_mutated_outcome_spec_hash() -> None:
    record = source_record("user")
    mutated_manifest = dict(record.execution_manifest)
    mutated_spec = dict(mutated_manifest["outcome_spec"])
    mutated_spec["outcome_filter"] = {"destination_ids": ["seoul"]}
    mutated_manifest["outcome_spec"] = mutated_spec
    mutated = replace(record, execution_manifest=mutated_manifest)
    builder = UpliftDatasetBuilder(
        unit_reader=_UnitReader([mutated]),
        outcome_reader=_OutcomeReader([]),
    )

    result = builder.build(reference_time=WINDOW_END)

    assert result.excluded_reason_counts == {"outcome_spec_mismatch": 1}


def test_clickhouse_outcome_query_uses_destination_aliases_and_half_open_window() -> None:
    class _Result:
        result_rows = [("jeju_user",)]

    class _Client:
        def __init__(self):
            self.sql = ""
            self.parameters = {}

        def query(self, sql, *, parameters):
            self.sql = sql
            self.parameters = parameters
            return _Result()

    client = _Client()
    repository = ClickHouseOutcomeEventRepository(client)

    success_user_ids = repository.list_success_user_ids(
        project_id="project",
        user_ids=["jeju_user", "seoul_user"],
        event_name="booking_complete",
        window_start=ASSIGNED_AT,
        window_end=WINDOW_END,
        destination_ids=["okinawa", "jeju"],
    )

    assert success_user_ids == {"jeju_user"}
    assert "event_time >=" in client.sql
    assert "event_time <" in client.sql
    assert "promotion_id" not in client.sql
    assert set(client.parameters["destination_aliases"]) >= {
        "jeju",
        "제주",
        "okinawa",
        "오키나와",
    }


def source_record(user_id: str, **overrides) -> UpliftUnitSourceRecord:
    spec, spec_hash = build_frozen_outcome_spec(
        goal_metric="booking_conversion_rate",
        target_segment_rules=[],
    )
    spec["outcome_filter"] = {"destination_ids": ["jeju", "okinawa"]}
    from app.decision.outcome_spec import outcome_spec_hash

    spec_hash = outcome_spec_hash(spec)
    values = {
        "experiment_unit_id": f"unit_{user_id}",
        "project_id": "project",
        "promotion_run_id": "run",
        "ad_experiment_id": "experiment",
        "segment_id": "segment",
        "audience_snapshot_id": "snapshot",
        "vector_generation_id": "generation",
        "user_id": user_id,
        "arm": "treatment" if user_id == "jeju_user" else "control",
        "treatment_probability": Decimal("0.5"),
        "assigned_at": ASSIGNED_AT,
        "outcome_window_start": ASSIGNED_AT,
        "outcome_window_end": WINDOW_END,
        "execution_source_cutoff_at": CUTOFF,
        "generation_window_end": CUTOFF,
        "generation_source_revision_cutoff": CUTOFF,
        "audience_source_cutoff": CUTOFF,
        "execution_manifest": {
            "schema_version": EXECUTION_SCHEMA_VERSION,
            "experiment_design": {
                "mode": "randomized_holdout",
                "outcome_spec_hash": spec_hash,
            },
            "outcome_spec": spec,
        },
        "goal_snapshot_json": {
            "outcome_spec": spec,
            "outcome_spec_hash": spec_hash,
        },
        "feature_vector": json.dumps([0.125] * 64),
    }
    values.update(overrides)
    return UpliftUnitSourceRecord(**values)


def replace_outcome(record, spec, spec_hash):
    execution_manifest = {
        **record.execution_manifest,
        "experiment_design": {
            "mode": "randomized_holdout",
            "outcome_spec_hash": spec_hash,
        },
        "outcome_spec": spec,
    }
    return replace(
        record,
        execution_manifest=execution_manifest,
        goal_snapshot_json={
            "outcome_spec": spec,
            "outcome_spec_hash": spec_hash,
        },
    )
