from __future__ import annotations

from datetime import UTC, datetime

import pytest

from scripts.prepare_ann_vector_build_database import (
    build_copy_sql,
    build_schema_sql,
    checked_identifier,
    require_matching_census,
)


def test_build_database_uses_hash_partition_and_production_table_names() -> None:
    sql = build_schema_sql(
        source_database="loopad",
        target_database="ann_scale_v2_build",
        shard_count=32,
    )

    assert "PARTITION BY modulo(cityHash64(user_id), 32)" in sql
    assert "ann_scale_v2_build.raw_events" in sql
    assert "ann_scale_v2_build.user_behavior_vectors" in sql
    assert "ann_scale_v2_build.user_behavior_vector_revisions" in sql
    assert "FROM ann_scale_v2_build.user_behavior_vectors" in sql


def test_build_database_rejects_unsafe_identifiers() -> None:
    assert checked_identifier("ann_scale_v2") == "ann_scale_v2"
    with pytest.raises(ValueError, match="unsafe"):
        checked_identifier("loopad; DROP DATABASE loopad")


def test_copy_sql_is_frozen_to_project_and_half_open_window() -> None:
    sql = build_copy_sql(
        source_database="loopad",
        target_database="ann_scale_v2_build",
        project_id="expedia_ann_scale_v2",
        window_start=datetime(2013, 1, 1, tzinfo=UTC),
        window_end=datetime(2015, 1, 1, tzinfo=UTC),
    )

    assert "project_id = 'expedia_ann_scale_v2'" in sql
    assert "event_time >= toDateTime64('2013-01-01 00:00:00'" in sql
    assert "event_time < toDateTime64('2015-01-01 00:00:00'" in sql


def test_source_target_census_must_match_every_shard() -> None:
    source = [{"shard": 0, "row_count": 5, "user_count": 2}]
    require_matching_census(source, list(source))
    with pytest.raises(RuntimeError, match="differs"):
        require_matching_census(
            source,
            [{"shard": 0, "row_count": 4, "user_count": 2}],
        )
