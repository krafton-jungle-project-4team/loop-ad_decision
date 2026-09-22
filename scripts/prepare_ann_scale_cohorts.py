#!/usr/bin/env python3
"""Prepare and prefix-verify every PostgreSQL ANN scale cohort."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
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
from scripts.prepare_ann_scale_postgres import cohort_database_name, positive_ints  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--clickhouse-database", required=True)
    parser.add_argument("--postgres-prefix", required=True)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--vector-version", required=True)
    parser.add_argument("--manifest-hash", required=True)
    parser.add_argument("--window-start", required=True)
    parser.add_argument("--window-end", required=True)
    parser.add_argument("--source-revision-cutoff", required=True)
    parser.add_argument("--cohort-seed", required=True)
    parser.add_argument("--membership-manifest", type=Path, required=True)
    parser.add_argument(
        "--cohort-sizes", default="50000,100000,250000,500000,750000,1000000"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confirm-empty-disposable-postgres", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm_empty_disposable_postgres:
        raise ValueError("cohort preparation requires explicit confirmation")
    membership = object_at(args.membership_manifest)
    expected_hashes = membership["cohort_prefix_sha256"]
    prepared: list[Mapping[str, Any]] = []
    for size in positive_ints(args.cohort_sizes):
        database = cohort_database_name(args.postgres_prefix, size)
        output_path = args.output_dir / f"cohort-{size}.json"
        if not output_path.exists():
            environment = {
                **os.environ,
                "LOOPAD_AURORA_DATABASE": database,
                "LOOPAD_CLICKHOUSE_DATABASE": args.clickhouse_database,
            }
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/benchmark_audience_search.py"),
                    "prepare-cohort",
                    "--env-file",
                    str(args.env_file),
                    "--project-id",
                    args.project_id,
                    "--vector-version",
                    args.vector_version,
                    "--manifest-hash",
                    args.manifest_hash,
                    "--window-start",
                    args.window_start,
                    "--window-end",
                    args.window_end,
                    "--source-revision-cutoff",
                    args.source_revision_cutoff,
                    "--cohort-size",
                    str(size),
                    "--cohort-seed",
                    args.cohort_seed,
                    "--output",
                    str(output_path),
                    "--confirm-empty-disposable-postgres",
                ],
                check=True,
                cwd=ROOT,
                env=environment,
            )
        payload = object_at(output_path)
        result = payload["result"]
        require_expected_prefix(
            cohort_size=size,
            observed_hash=str(result["cohort_sha256"]),
            expected_hashes=expected_hashes,
        )
        if int(result["postgres_row_count"]) != size:
            raise RuntimeError("prepared cohort row count differs from requested size")
        prepared.append(
            {
                "cohort_size": size,
                "database": database,
                "artifact": str(output_path),
                **dict(result),
                "membership_prefix_verified": True,
            }
        )
        print(
            json.dumps(
                {
                    "prepared_cohort": size,
                    "database": database,
                    "prefix_verified": True,
                }
            ),
            flush=True,
        )
    write_immutable_json(
        args.output,
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "membership_manifest": str(args.membership_manifest),
            "cohort_seed": args.cohort_seed,
            "cohorts": prepared,
        },
    )
    return 0


def require_expected_prefix(
    *,
    cohort_size: int,
    observed_hash: str,
    expected_hashes: Mapping[str, Any],
) -> None:
    expected = str(expected_hashes.get(str(cohort_size), ""))
    if len(expected) != 64 or observed_hash != expected:
        raise RuntimeError(
            f"cohort {cohort_size} hash does not match frozen membership prefix"
        )


def object_at(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain an object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
