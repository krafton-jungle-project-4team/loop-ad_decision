import json
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from app.decision.experiment_design import EXECUTION_SCHEMA_VERSION
from app.decision.outcome_spec import build_frozen_outcome_spec
from app.uplift.dataset import (
    ClickHouseOutcomeEventRepository,
    PostgresUpliftUnitSourceRepository,
    UpliftDatasetBuilder,
    UpliftUnitSourceRecord,
    build_dataset_manifest,
)


ASSIGNED_AT = datetime(2026, 7, 1, tzinfo=UTC)
WINDOW_END = ASSIGNED_AT + timedelta(days=30)
CUTOFF = ASSIGNED_AT - timedelta(minutes=1)


class _UnitReader:
    def __init__(self, records):
        self.records = records

    def iter_unit_pages(
        self,
        *,
        project_id,
        reference_time,
        after_promotion_run_id=None,
        after_ad_experiment_id=None,
        after_user_id=None,
        page_size=1000,
    ):
        del reference_time
        records = [
            record
            for record in self.records
            if record.project_id == project_id
        ]
        for start in range(0, len(records), page_size):
            yield tuple(records[start : start + page_size])


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

    result = builder.build(reference_time=WINDOW_END, project_id="project")

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

    result = builder.build(reference_time=WINDOW_END, project_id="project")

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

    result = builder.build(reference_time=WINDOW_END, project_id="project")

    assert result.excluded_reason_counts == {"outcome_spec_mismatch": 1}


def test_unit_source_reads_only_finalized_uplift_executions() -> None:
    class _Db:
        def __init__(self):
            self.query = ""

        def fetchall(self, query, params=()):
            self.query = query
            return []

    db = _Db()
    repository = PostgresUpliftUnitSourceRepository(db)

    assert list(
        repository.iter_unit_pages(
            project_id="project",
            reference_time=WINDOW_END,
        )
    ) == []
    assert "execution.uplift_assignment_status = 'finalized'" in db.query
    assert "unit.outcome_window_end <=" in db.query
    assert "randomized_holdout" in db.query
    assert "uplift_training_eligible" in db.query


def test_unit_source_uses_stable_composite_cursor_without_offset() -> None:
    rows = [
        asdict(
            source_record(
                "u1",
                promotion_run_id="run_a",
                ad_experiment_id="exp_a",
            )
        ),
        asdict(
            source_record(
                "u2",
                promotion_run_id="run_a",
                ad_experiment_id="exp_b",
            )
        ),
    ]

    class _Db:
        def __init__(self):
            self.calls = []

        def fetchall(self, query, params=()):
            self.calls.append((query, params))
            return rows if len(self.calls) == 1 else []

    db = _Db()
    pages = list(
        PostgresUpliftUnitSourceRepository(db).iter_unit_pages(
            project_id="project",
            reference_time=WINDOW_END,
            page_size=2,
        )
    )

    assert [[record.user_id for record in page] for page in pages] == [
        ["u1", "u2"]
    ]
    assert len(db.calls) == 2
    assert "OFFSET" not in db.calls[0][0]
    assert db.calls[0][1][2:6] == (None, None, None, None)
    assert db.calls[1][1][2:6] == ("run_a", "run_a", "exp_b", "u2")


def test_clickhouse_outcome_query_checks_every_destination_field_with_or() -> None:
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
    assert client.parameters["destination_ids"] == ["okinawa", "jeju"]
    for property_key in (
        "destination_id",
        "destination_name",
        "hotel_city",
        "hotel_country",
    ):
        assert f"'{property_key}'" in client.sql
    assert client.sql.count(" OR\n") == 3


def test_clickhouse_outcome_query_batches_users_and_combines_matches() -> None:
    class _Result:
        def __init__(self, user_ids):
            self.result_rows = [(user_id,) for user_id in user_ids]

    class _Client:
        def __init__(self):
            self.calls = []

        def query(self, sql, *, parameters):
            self.calls.append((sql, parameters))
            return _Result(parameters["user_ids"][:1])

    client = _Client()
    repository = ClickHouseOutcomeEventRepository(client, user_batch_size=2)

    success_user_ids = repository.list_success_user_ids(
        project_id="project",
        user_ids=["u1", "u2", "u3", "u4", "u5"],
        event_name="booking_complete",
        window_start=ASSIGNED_AT,
        window_end=WINDOW_END,
        destination_ids=["jeju"],
    )

    assert [call[1]["user_ids"] for call in client.calls] == [
        ["u1", "u2"],
        ["u3", "u4"],
        ["u5"],
    ]
    assert success_user_ids == {"u1", "u3", "u5"}


def test_destination_match_contract_accepts_alias_in_any_supported_field() -> None:
    class _Result:
        result_rows = []

    class _Client:
        def __init__(self):
            self.sql = ""

        def query(self, sql, *, parameters):
            self.sql = sql
            return _Result()

    client = _Client()
    repository = ClickHouseOutcomeEventRepository(client)
    repository.list_success_user_ids(
        project_id="project",
        user_ids=["numeric_id_with_jeju_name", "seoul_id_with_jeju_name"],
        event_name="booking_complete",
        window_start=ASSIGNED_AT,
        window_end=WINDOW_END,
        destination_ids=["jeju"],
    )

    # Outcome v1 intentionally uses OR: a canonical match in destination_name
    # remains valid even when destination_id is numeric or names another place.
    assert "JSONExtractString(properties_json, 'destination_id')" in client.sql
    assert "JSONExtractString(properties_json, 'destination_name')" in client.sql
    assert client.sql.count(" OR\n") == 3


def test_dataset_fingerprint_is_independent_of_postgres_page_size_and_order() -> None:
    records = [source_record(f"user_{index}") for index in range(6)]
    events = [
        (
            "user_1",
            "booking_complete",
            "jeju",
            ASSIGNED_AT + timedelta(days=1),
        )
    ]
    first = UpliftDatasetBuilder(
        unit_reader=_UnitReader(records),
        outcome_reader=_OutcomeReader(events),
        unit_page_size=1,
    ).build(project_id="project", reference_time=WINDOW_END)
    second = UpliftDatasetBuilder(
        unit_reader=_UnitReader(list(reversed(records))),
        outcome_reader=_OutcomeReader(events),
        unit_page_size=4,
    ).build(project_id="project", reference_time=WINDOW_END)

    first_manifest, first_fingerprint = build_dataset_manifest(
        first,
        project_id="project",
        reference_time=WINDOW_END,
    )
    second_manifest, second_fingerprint = build_dataset_manifest(
        second,
        project_id="project",
        reference_time=WINDOW_END,
    )

    assert first_manifest == second_manifest
    assert first_fingerprint == second_fingerprint
    assert first_manifest["experiment_unit_set_hash"]
    assert first_manifest["feature_contract_hash"]
    assert first_manifest["outcome_contract_hash"]


def test_dataset_fingerprint_is_independent_of_clickhouse_batch_size() -> None:
    class _Result:
        def __init__(self, user_ids):
            self.result_rows = [
                (user_id,) for user_id in user_ids if user_id.endswith("1")
            ]

    class _Client:
        def query(self, _sql, *, parameters):
            return _Result(parameters["user_ids"])

    records = [source_record(f"user_{index}") for index in range(5)]
    fingerprints = []
    for batch_size in (1, 5000):
        result = UpliftDatasetBuilder(
            unit_reader=_UnitReader(records),
            outcome_reader=ClickHouseOutcomeEventRepository(
                _Client(),
                user_batch_size=batch_size,
            ),
            unit_page_size=2,
        ).build(project_id="project", reference_time=WINDOW_END)
        _manifest, fingerprint = build_dataset_manifest(
            result,
            project_id="project",
            reference_time=WINDOW_END,
        )
        fingerprints.append(fingerprint)

    assert fingerprints[0] == fingerprints[1]


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
        "generation_vector_version": "hotel_behavior.v2",
        "generation_manifest_hash": "a" * 64,
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
