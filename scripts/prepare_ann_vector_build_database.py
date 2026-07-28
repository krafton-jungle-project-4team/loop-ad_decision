#!/usr/bin/env python3
"""Prepare an exact, hash-partitioned ClickHouse source for scale vector builds."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_EXPERIMENT_VERSION,
)


IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container", default="loop-ad_data-source_contract-clickhouse-1"
    )
    parser.add_argument("--source-database", default="loopad")
    parser.add_argument("--target-database", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--build-shard-count", type=positive_int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-local-clickhouse-write", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_local_clickhouse_write:
        raise ValueError("preparation requires --confirm-local-clickhouse-write")
    source = checked_identifier(args.source_database)
    target = checked_identifier(args.target_database)
    if source == target:
        raise ValueError("vector build database must differ from the source database")
    window_start = parse_utc(args.window_start)
    window_end = parse_utc(args.window_end)
    if window_start >= window_end:
        raise ValueError("window_start must precede window_end")

    source_census = query_census(
        args,
        database=source,
        project_id=args.project_id,
        window_start=window_start,
        window_end=window_end,
    )
    if not source_census or sum(row["row_count"] for row in source_census) <= 0:
        raise RuntimeError("source census is empty")

    run_sql(
        args,
        build_schema_sql(
            source_database=source,
            target_database=target,
            shard_count=args.build_shard_count,
        ),
    )
    target_census = query_census(
        args,
        database=target,
        project_id=args.project_id,
        window_start=window_start,
        window_end=window_end,
    )
    if not target_census:
        run_sql(
            args,
            build_copy_sql(
                source_database=source,
                target_database=target,
                project_id=args.project_id,
                window_start=window_start,
                window_end=window_end,
            ),
        )
        target_census = query_census(
            args,
            database=target,
            project_id=args.project_id,
            window_start=window_start,
            window_end=window_end,
        )

    require_matching_census(source_census, target_census)
    output_rows = query_json_rows(
        args,
        f"""
        SELECT
            (SELECT count() FROM {target}.user_behavior_vectors) AS vector_rows,
            (SELECT count() FROM {target}.user_behavior_vector_revisions)
                AS revision_rows
        FORMAT JSONEachRow
        """,
    )
    if len(output_rows) != 1:
        raise RuntimeError("vector output census returned an unexpected shape")
    if int(output_rows[0]["vector_rows"]) != 0 or int(
        output_rows[0]["revision_rows"]
    ) != 0:
        raise RuntimeError("benchmark vector output tables must be empty")

    payload = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": "complete",
        "source_database": source,
        "target_database": target,
        "project_id": args.project_id,
        "window_start": window_start.isoformat(),
        "window_end": window_end.isoformat(),
        "build_shard_count": args.build_shard_count,
        "row_count": sum(row["row_count"] for row in target_census),
        "user_count": sum(row["user_count"] for row in target_census),
        "shard_census": target_census,
        "source_target_census_equal": True,
        "production_vector_sql_table_name": "raw_events",
    }
    write_immutable_json(args.output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def checked_identifier(value: str) -> str:
    if not IDENTIFIER.fullmatch(value):
        raise ValueError(f"unsafe ClickHouse identifier: {value!r}")
    return value


def build_schema_sql(
    *,
    source_database: str,
    target_database: str,
    shard_count: int,
) -> str:
    source = checked_identifier(source_database)
    target = checked_identifier(target_database)
    if shard_count <= 1:
        raise ValueError("staging source requires at least two hash partitions")
    return f"""
    CREATE DATABASE IF NOT EXISTS {target};

    CREATE TABLE IF NOT EXISTS {target}.raw_events AS {source}.raw_events
    ENGINE = MergeTree
    PARTITION BY modulo(cityHash64(user_id), {shard_count})
    ORDER BY (project_id, event_name, event_time, user_id, event_id)
    SETTINGS index_granularity = 8192;

    CREATE TABLE IF NOT EXISTS {target}.user_behavior_vectors
    AS {source}.user_behavior_vectors;

    CREATE TABLE IF NOT EXISTS {target}.user_behavior_vector_revisions
    AS {source}.user_behavior_vector_revisions;

    CREATE MATERIALIZED VIEW IF NOT EXISTS
        {target}.mv_user_behavior_vectors_to_revisions
    TO {target}.user_behavior_vector_revisions
    AS SELECT
        project_id,
        user_id,
        vector_dim,
        vector_values,
        vector_version,
        source,
        window_start,
        window_end,
        updated_at,
        lower(hex(SHA256(toJSONString(tuple(
            project_id,
            user_id,
            vector_version,
            toUnixTimestamp64Milli(updated_at),
            vector_dim,
            vector_values,
            CAST(source, 'String'),
            toUnixTimestamp64Milli(window_start),
            toUnixTimestamp64Milli(window_end)
        ))))) AS vector_row_id,
        now64(6, 'UTC') AS ingested_at
    FROM {target}.user_behavior_vectors;
    """


def build_copy_sql(
    *,
    source_database: str,
    target_database: str,
    project_id: str,
    window_start: datetime,
    window_end: datetime,
) -> str:
    source = checked_identifier(source_database)
    target = checked_identifier(target_database)
    return f"""
    INSERT INTO {target}.raw_events
    SELECT *
    FROM {source}.raw_events
    WHERE project_id = {sql_string(project_id)}
      AND validation_status = 'valid'
      AND event_time >= toDateTime64({sql_string(clickhouse_time(window_start))}, 3, 'UTC')
      AND event_time < toDateTime64({sql_string(clickhouse_time(window_end))}, 3, 'UTC')
    SETTINGS max_threads = 4;
    """


def query_census(
    args: argparse.Namespace,
    *,
    database: str,
    project_id: str,
    window_start: datetime,
    window_end: datetime,
) -> list[dict[str, int]]:
    database = checked_identifier(database)
    rows = query_json_rows(
        args,
        f"""
        SELECT
            modulo(cityHash64(user_id), {args.build_shard_count}) AS shard,
            count() AS row_count,
            uniqExact(user_id) AS user_count
        FROM {database}.raw_events
        WHERE project_id = {sql_string(project_id)}
          AND validation_status = 'valid'
          AND event_time >= toDateTime64({sql_string(clickhouse_time(window_start))}, 3, 'UTC')
          AND event_time < toDateTime64({sql_string(clickhouse_time(window_end))}, 3, 'UTC')
        GROUP BY shard
        ORDER BY shard
        FORMAT JSONEachRow
        """,
    )
    return [
        {
            "shard": int(row["shard"]),
            "row_count": int(row["row_count"]),
            "user_count": int(row["user_count"]),
        }
        for row in rows
    ]


def require_matching_census(
    source: list[Mapping[str, int]],
    target: list[Mapping[str, int]],
) -> None:
    if [dict(row) for row in source] != [dict(row) for row in target]:
        raise RuntimeError("hash-partitioned source differs from frozen raw source")


def run_sql(args: argparse.Namespace, query: str) -> None:
    subprocess.run(
        [
            "docker",
            "exec",
            args.container,
            "clickhouse-client",
            "--multiquery",
            "--query",
            query,
        ],
        check=True,
    )


def query_json_rows(args: argparse.Namespace, query: str) -> list[Mapping[str, Any]]:
    completed = subprocess.run(
        [
            "docker",
            "exec",
            args.container,
            "clickhouse-client",
            "--query",
            query,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in completed.stdout.splitlines() if line.strip()]


def sql_string(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def clickhouse_time(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%d %H:%M:%S")


def parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
