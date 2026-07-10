#!/usr/bin/env python3
"""Backfill SDK-like raw_events from ClickHouse expedia_hotel_events.

Default mode is preview. It prints the number of raw_events that would be
created per event_name without inserting rows.

Examples:
    python scripts/backfill_expedia_raw_events.py --mode preview
    python scripts/backfill_expedia_raw_events.py --mode sample --max-source-rows 3
    python scripts/backfill_expedia_raw_events.py --mode execute --project-id demo_project
    python scripts/backfill_expedia_raw_events.py --mode execute --user-sample-modulo 1 --max-source-rows 0
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SQL_PATH = Path(__file__).with_name("expedia_to_raw_events.sql")
START_MARKER = "-- TRANSFORM_SELECT_START"
END_MARKER = "-- TRANSFORM_SELECT_END"


def main() -> int:
    args = parse_args()
    sql = SQL_PATH.read_text(encoding="utf-8")
    query = build_query(sql, mode=args.mode)
    command = clickhouse_command(args)
    completed = subprocess.run(  # noqa: S603 - command is fixed, arguments are parsed.
        command,
        input=query,
        text=True,
        check=False,
    )
    return completed.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backfill raw_events from expedia_hotel_events using deterministic Expedia-derived events.",
    )
    parser.add_argument(
        "--mode",
        choices=("preview", "sample", "execute"),
        default="preview",
        help="preview prints transformed event counts; sample prints transformed rows; execute inserts into raw_events.",
    )
    parser.add_argument(
        "--container",
        default="loop-ad_data-source_contract-clickhouse-1",
        help="Docker container name for local ClickHouse. Ignored when --local is set.",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Use local clickhouse-client instead of docker exec.",
    )
    parser.add_argument("--project-id", default="demo_project")
    parser.add_argument("--write-key", default="expedia-backfill")
    parser.add_argument("--schema-version", default="expedia.raw_events.v1")
    parser.add_argument("--source", default="expedia_hotel_events_backfill")
    parser.add_argument(
        "--user-sample-modulo",
        type=positive_int,
        default=20,
        help="Deterministic user sampling denominator. 20 means about 5%% of users. Use 1 for all users.",
    )
    parser.add_argument(
        "--user-sample-remainder",
        type=nonnegative_int,
        default=0,
        help="Deterministic user sampling remainder.",
    )
    parser.add_argument(
        "--max-source-rows",
        type=nonnegative_int,
        default=1_000_000,
        help="Max Expedia source rows to transform after sampling. 0 means no limit.",
    )
    parser.add_argument(
        "--start-datetime",
        default="",
        help="Optional inclusive Expedia date_time lower bound, e.g. 2014-01-01 00:00:00.",
    )
    parser.add_argument(
        "--end-datetime",
        default="",
        help="Optional exclusive Expedia date_time upper bound, e.g. 2015-01-01 00:00:00.",
    )
    args = parser.parse_args()
    if args.user_sample_remainder >= args.user_sample_modulo:
        parser.error("--user-sample-remainder must be smaller than --user-sample-modulo")
    return args


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return parsed


def build_query(sql: str, *, mode: str) -> str:
    if mode == "execute":
        return sql
    transform_select = extract_transform_select(sql)
    if mode == "sample":
        return f"""
SELECT
    event_name,
    event_time,
    user_id,
    session_id,
    properties_json
FROM (
{transform_select}
) AS transformed_raw_events
ORDER BY event_time ASC, user_id ASC, event_name ASC
LIMIT 12
FORMAT PrettyCompact
"""
    return f"""
SELECT
    event_name,
    count() AS raw_event_count,
    countDistinct(user_id) AS user_count,
    min(event_time) AS first_event_time,
    max(event_time) AS last_event_time
FROM (
{transform_select}
) AS transformed_raw_events
GROUP BY event_name
ORDER BY event_name ASC
FORMAT PrettyCompact
"""


def extract_transform_select(sql: str) -> str:
    start = sql.find(START_MARKER)
    end = sql.find(END_MARKER)
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError("SQL transform markers are missing or invalid")
    return sql[start + len(START_MARKER) : end].strip()


def clickhouse_command(args: argparse.Namespace) -> list[str]:
    params = [
        "--param_project_id",
        args.project_id,
        "--param_write_key",
        args.write_key,
        "--param_schema_version",
        args.schema_version,
        "--param_source",
        args.source,
        "--param_user_sample_modulo",
        str(args.user_sample_modulo),
        "--param_user_sample_remainder",
        str(args.user_sample_remainder),
        "--param_max_source_rows",
        str(args.max_source_rows),
        "--param_start_datetime",
        args.start_datetime,
        "--param_end_datetime",
        args.end_datetime,
        "--multiquery",
    ]
    if args.local:
        return ["clickhouse-client", *params]
    return [
        "docker",
        "exec",
        "-i",
        args.container,
        "sh",
        "-lc",
        'clickhouse-client --database "$CLICKHOUSE_DB" --user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" "$@"',
        "clickhouse-client",
        *params,
    ]


if __name__ == "__main__":
    sys.exit(main())
