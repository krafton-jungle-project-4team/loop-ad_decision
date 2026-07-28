#!/usr/bin/env python3
"""Create fresh, separate Data Contract PostgreSQL DBs for scale cohorts."""

from __future__ import annotations

import argparse
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


IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--container", default="loop-ad_data-source_contract-postgres-1"
    )
    parser.add_argument("--user", default="loopad")
    parser.add_argument("--admin-database", default="loopad")
    parser.add_argument("--database-prefix", default="loopad_ann_v2")
    parser.add_argument(
        "--cohort-sizes", default="50000,100000,250000,500000,750000,1000000"
    )
    parser.add_argument(
        "--schema", default="/docker-entrypoint-initdb.d/01-schema.sql"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-create-disposable-databases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_create_disposable_databases:
        raise ValueError("database preparation requires explicit confirmation")
    sizes = positive_ints(args.cohort_sizes)
    databases: list[Mapping[str, Any]] = []
    for size in sizes:
        database = cohort_database_name(args.database_prefix, size)
        if not database_exists(args, database):
            run(
                args,
                ["createdb", "-U", args.user, "-T", "template0", database],
            )
        run(
            args,
            [
                "psql",
                "-v",
                "ON_ERROR_STOP=1",
                "-U",
                args.user,
                "-d",
                database,
                "-f",
                args.schema,
            ],
        )
        census = database_census(args, database)
        if any(census[key] != 0 for key in (
            "project_count", "generation_count", "search_row_count"
        )):
            raise RuntimeError(f"disposable cohort DB is not empty: {database}")
        databases.append(
            {
                "cohort_size": size,
                "database": database,
                **census,
            }
        )
        print(json.dumps({"prepared_database": database, "cohort_size": size}))
    write_immutable_json(
        args.output,
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "container": args.container,
            "schema_path": args.schema,
            "databases": databases,
        },
    )
    return 0


def cohort_database_name(prefix: str, size: int) -> str:
    if not IDENTIFIER.fullmatch(prefix) or size <= 0:
        raise ValueError("unsafe cohort database identity")
    suffix = f"{size // 1000}k" if size % 1000 == 0 else str(size)
    database = f"{prefix}_{suffix}"
    if not IDENTIFIER.fullmatch(database):
        raise ValueError("unsafe cohort database name")
    return database


def database_exists(args: argparse.Namespace, database: str) -> bool:
    output = capture(
        args,
        [
            "psql",
            "-U",
            args.user,
            "-d",
            args.admin_database,
            "-Atc",
            "SELECT 1 FROM pg_database WHERE datname = " + sql_literal(database),
        ],
    )
    return output.strip() == "1"


def database_census(args: argparse.Namespace, database: str) -> dict[str, int]:
    output = capture(
        args,
        [
            "psql",
            "-U",
            args.user,
            "-d",
            database,
            "-AtF",
            "|",
            "-c",
            "SELECT (SELECT count(*) FROM projects), "
            "(SELECT count(*) FROM user_behavior_vector_search_generations), "
            "(SELECT count(*) FROM user_behavior_vector_search), "
            "(SELECT count(*) FROM pg_class WHERE relkind IN ('r','p') "
            "AND relnamespace='public'::regnamespace)",
        ],
    )
    parts = output.strip().split("|")
    if len(parts) != 4:
        raise RuntimeError("unexpected PostgreSQL census output")
    return {
        "project_count": int(parts[0]),
        "generation_count": int(parts[1]),
        "search_row_count": int(parts[2]),
        "public_table_count": int(parts[3]),
    }


def run(args: argparse.Namespace, command: list[str]) -> None:
    subprocess.run(["docker", "exec", args.container, *command], check=True)


def capture(args: argparse.Namespace, command: list[str]) -> str:
    completed = subprocess.run(
        ["docker", "exec", args.container, *command],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("cohort sizes must be unique positive integers")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
