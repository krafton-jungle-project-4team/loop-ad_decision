#!/usr/bin/env python3
"""Checkpointed full-source Expedia backfill for ANN scale-series v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_scale_artifacts import AppendOnlyJsonl  # noqa: E402


DEFAULT_PARTITION_COUNT = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--container", default="loop-ad_data-source_contract-clickhouse-1")
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--write-key", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--start-datetime", required=True)
    parser.add_argument("--end-datetime", required=True)
    parser.add_argument("--partition-count", type=positive_int, default=DEFAULT_PARTITION_COUNT)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = query_expected_partitions(args)
    write_or_verify_plan(args.plan, expected, args)
    actual_before = query_actual_partitions(args)
    completed = completed_partitions(args.checkpoint)
    actions = plan_resume_actions(
        expected=expected,
        actual=actual_before,
        completed=completed,
    )
    if args.plan_only:
        print(json.dumps({"status": "planned", "actions": actions}, indent=2))
        return 0

    checkpoint = AppendOnlyJsonl(
        args.checkpoint,
        identity_fields=("partition_count", "partition_remainder"),
    )
    for remainder in actions:
        run_partition(args, remainder)
        checkpoint.append(
            (
                {
                    "partition_count": args.partition_count,
                    "partition_remainder": remainder,
                    "expected_raw_event_count": expected[remainder]["raw_event_count"],
                    "expected_user_count": expected[remainder]["user_count"],
                    "status": "insert_command_succeeded",
                },
            )
        )
        print(
            json.dumps(
                {
                    "completed_partition": remainder,
                    "partition_count": args.partition_count,
                }
            ),
            flush=True,
        )

    actual_after = query_actual_partitions(args)
    remaining = plan_resume_actions(
        expected=expected,
        actual=actual_after,
        completed=completed_partitions(args.checkpoint),
    )
    if remaining:
        raise RuntimeError(f"backfill validation left pending partitions: {remaining}")
    print(
        json.dumps(
            {
                "status": "complete",
                "partition_count": args.partition_count,
                "raw_event_count": sum(
                    item["raw_event_count"] for item in expected.values()
                ),
                "source_user_count": sum(
                    item["user_count"] for item in expected.values()
                ),
            },
            indent=2,
        )
    )
    return 0


def query_expected_partitions(args: argparse.Namespace) -> dict[int, dict[str, int]]:
    query = """
        SELECT
            modulo(cityHash64(toString(user_id)), {partition_count:UInt64})
                AS partition_remainder,
            count() * 3
                + countIf(cnt >= 2 OR is_booking = 1)
                + countIf(is_booking = 1 OR (is_booking = 0 AND cnt >= 4))
                + countIf(is_booking = 1) AS raw_event_count,
            uniqExact(user_id) AS user_count
        FROM expedia_hotel_events
        WHERE date_time >= {start:DateTime}
          AND date_time < {end:DateTime}
        GROUP BY partition_remainder
        ORDER BY partition_remainder
        FORMAT JSONEachRow
    """
    rows = clickhouse_json_rows(args, query)
    result = {
        int(row["partition_remainder"]): {
            "raw_event_count": int(row["raw_event_count"]),
            "user_count": int(row["user_count"]),
        }
        for row in rows
    }
    expected_keys = set(range(args.partition_count))
    if set(result) != expected_keys:
        raise RuntimeError("source does not populate every configured partition")
    return result


def query_actual_partitions(args: argparse.Namespace) -> dict[int, dict[str, int]]:
    query = """
        SELECT
            modulo(
                cityHash64(replaceOne(user_id, {prefix:String}, {empty:String})),
                {partition_count:UInt64}
            ) AS partition_remainder,
            count() AS raw_event_count,
            uniqExact(user_id) AS user_count
        FROM raw_events
        WHERE project_id = {project_id:String}
        GROUP BY partition_remainder
        ORDER BY partition_remainder
        FORMAT JSONEachRow
    """
    return {
        int(row["partition_remainder"]): {
            "raw_event_count": int(row["raw_event_count"]),
            "user_count": int(row["user_count"]),
        }
        for row in clickhouse_json_rows(args, query)
    }


def clickhouse_json_rows(
    args: argparse.Namespace,
    query: str,
) -> list[Mapping[str, Any]]:
    command = [
        "docker",
        "exec",
        "-i",
        args.container,
        "sh",
        "-lc",
        'clickhouse-client --database "$CLICKHOUSE_DB" '
        '--user "$CLICKHOUSE_USER" --password "$CLICKHOUSE_PASSWORD" '
        '"$@"',
        "clickhouse-client",
        "--param_partition_count",
        str(args.partition_count),
        "--param_start",
        args.start_datetime,
        "--param_end",
        args.end_datetime,
        "--param_project_id",
        args.project_id,
        "--param_prefix",
        "expedia-user-",
        "--param_empty",
        "",
        "--query",
        query,
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    return [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]


def write_or_verify_plan(
    path: Path,
    expected: Mapping[int, Mapping[str, int]],
    args: argparse.Namespace,
) -> None:
    payload = {
        "experiment_version": "audience_search.benchmark.v2",
        "project_id": args.project_id,
        "partition_count": args.partition_count,
        "start_datetime": args.start_datetime,
        "end_datetime": args.end_datetime,
        "partitions": [
            {"partition_remainder": remainder, **expected[remainder]}
            for remainder in sorted(expected)
        ],
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("existing partition plan differs from frozen source")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def completed_partitions(path: Path) -> set[int]:
    if not path.exists():
        return set()
    result: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            raise RuntimeError("blank partition checkpoint row")
        row = json.loads(line)
        if row.get("status") != "insert_command_succeeded":
            raise RuntimeError("partition checkpoint contains incomplete status")
        remainder = int(row["partition_remainder"])
        if remainder in result:
            raise RuntimeError("partition checkpoint contains duplicates")
        result.add(remainder)
    return result


def plan_resume_actions(
    *,
    expected: Mapping[int, Mapping[str, int]],
    actual: Mapping[int, Mapping[str, int]],
    completed: set[int],
) -> list[int]:
    unknown = set(actual) - set(expected)
    if unknown:
        raise RuntimeError(f"actual backfill contains unknown partitions: {unknown}")
    pending: list[int] = []
    for remainder in sorted(expected):
        observed = actual.get(remainder)
        if remainder in completed:
            if observed != expected[remainder]:
                raise RuntimeError(
                    f"completed partition {remainder} differs from frozen plan"
                )
            continue
        if observed is not None:
            raise RuntimeError(
                f"partition {remainder} has uncheckpointed partial rows"
            )
        pending.append(remainder)
    return pending


def run_partition(args: argparse.Namespace, remainder: int) -> None:
    command = [
        sys.executable,
        str(ROOT / "scripts/backfill_expedia_raw_events.py"),
        "--mode",
        "execute",
        "--container",
        args.container,
        "--project-id",
        args.project_id,
        "--write-key",
        args.write_key,
        "--schema-version",
        "expedia.raw_events.v1",
        "--source",
        args.source,
        "--user-sample-modulo",
        str(args.partition_count),
        "--user-sample-remainder",
        str(remainder),
        "--max-source-rows",
        "0",
        "--start-datetime",
        args.start_datetime,
        "--end-datetime",
        args.end_datetime,
    ]
    subprocess.run(command, check=True)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
