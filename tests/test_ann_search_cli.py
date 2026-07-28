from __future__ import annotations

from datetime import UTC, datetime
import json
from pathlib import Path

import pytest

from app.internal.user_behavior_vectors import _build_hotel_behavior_v2_insert_sql
from scripts.benchmark_audience_search import (
    _ClickHouseSettingsClient,
    _ShardedVectorBuildRepository,
    _inject_user_shard_predicate,
)
from scripts.benchmark_ann_scale_series import _parse_utc


class _Client:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, int]]] = []

    def query(self, query, *, parameters=None, settings=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("query", dict(settings or {})))
        return "query-result"

    def command(self, query, *, parameters=None, settings=None, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(("command", dict(settings or {})))
        return "command-result"


def test_offline_clickhouse_settings_wrapper_merges_call_overrides() -> None:
    client = _Client()
    wrapped = _ClickHouseSettingsClient(
        client,
        {"max_threads": 4, "max_bytes_before_external_group_by": 1_000},
    )
    assert wrapped.query("SELECT 1", settings={"max_threads": 2}) == "query-result"
    assert wrapped.command("INSERT", settings={"max_memory_usage": 9}) == "command-result"
    assert client.calls == [
        (
            "query",
            {"max_threads": 2, "max_bytes_before_external_group_by": 1_000},
        ),
        (
            "command",
            {
                "max_threads": 4,
                "max_bytes_before_external_group_by": 1_000,
                "max_memory_usage": 9,
            },
        ),
    ]


def test_scale_membership_cli_requires_explicit_timezone() -> None:
    assert _parse_utc("2015-01-01T09:00:00+09:00").tzinfo is UTC
    assert _parse_utc("2015-01-01T09:00:00+09:00").hour == 0
    with pytest.raises(ValueError, match="timezone"):
        _parse_utc("2015-01-01T00:00:00")


def test_vector_shard_filter_changes_only_the_raw_event_user_scope() -> None:
    production_sql = _build_hotel_behavior_v2_insert_sql()
    sharded_sql = _inject_user_shard_predicate(production_sql)

    assert sharded_sql.count("build_shard_count") == 1
    assert sharded_sql.count("build_shard_index") == 1
    assert "modulo(cityHash64(user_id)" in sharded_sql
    assert sharded_sql.replace(
        "\n                      AND modulo(cityHash64(user_id), "
        "{build_shard_count:UInt64}) = {build_shard_index:UInt64}",
        "",
    ) == production_sql


def test_vector_shard_checkpoint_refuses_a_different_frozen_window(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "vector-build.jsonl"
    checkpoint.write_text(
        json.dumps(
            {
                "experiment_version": "audience_search.benchmark.v2",
                "project_id": "project",
                "vector_version": "hotel_behavior.v2",
                "window_start": "2013-01-01T00:00:00+00:00",
                "window_end": "2015-01-01T00:00:00+00:00",
                "build_shard_count": 32,
                "build_shard_index": 0,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    repository = _ShardedVectorBuildRepository(
        _Client(),
        shard_count=32,
        checkpoint_path=checkpoint,
    )

    with pytest.raises(RuntimeError, match="mismatched window_end"):
        repository._completed_shards(
            project_id="project",
            vector_version="hotel_behavior.v2",
            window_start=datetime(2013, 1, 1, tzinfo=UTC),
            window_end=datetime(2016, 1, 1, tzinfo=UTC),
        )


def test_vector_shard_materialized_count_requires_vectors_and_revisions() -> None:
    _ShardedVectorBuildRepository._require_materialized_count(
        shard_index=7,
        expected_count=11,
        actual=(11, 11),
    )
    with pytest.raises(RuntimeError, match="vector shard 7"):
        _ShardedVectorBuildRepository._require_materialized_count(
            shard_index=7,
            expected_count=11,
            actual=(11, 10),
        )
