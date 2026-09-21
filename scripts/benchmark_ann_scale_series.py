#!/usr/bin/env python3
"""Build and validate offline ANN scale-series benchmark v2 artifacts."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import load_settings  # noqa: E402
from app.db import create_clickhouse_client  # noqa: E402
from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    ActualScalePoint,
    ScaleArtifactValidator,
    write_actual_only_reports,
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    COHORT_MEMBERSHIP_RELATION,
    SCALE_COHORT_SIZES,
    SCALE_EXPERIMENT_VERSION,
    ScenarioCensusRecord,
    ScenarioSet,
    ScreeningCandidate,
    decide_artifact_reuse,
    inspect_fingerprint,
    membership_create_sql,
    membership_population_sql,
    old_50k_reuse_decision,
    rank_cohort_members,
    scale_manifest_template,
    select_scenario_pairs,
    select_survivors,
    validate_nested_prefixes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    template = subparsers.add_parser("manifest-template")
    template.add_argument("--output", type=Path, required=True)

    old = subparsers.add_parser("old-50k-decision")
    old.add_argument("--output", type=Path, required=True)

    fingerprint = subparsers.add_parser("check-fingerprint")
    fingerprint.add_argument("--input", type=Path, required=True)
    fingerprint.add_argument("--expected", type=Path)

    membership = subparsers.add_parser("rank-membership")
    membership.add_argument("--input", type=Path, required=True)
    membership.add_argument("--scale-series-id", required=True)
    membership.add_argument("--cohort-seed", required=True)
    membership.add_argument("--cohort-sizes", default="50000,100000,250000,500000,750000,1000000")
    membership.add_argument("--output", type=Path, required=True)
    membership.add_argument("--prefix-hashes", type=Path, required=True)

    live_membership = subparsers.add_parser(
        "prepare-membership-live",
        help="freeze deterministic nested membership from one vector revision",
    )
    live_membership.add_argument("--env-file", type=Path, default=Path(".env"))
    live_membership.add_argument("--project-id", required=True)
    live_membership.add_argument("--vector-version", default="hotel_behavior.v2")
    live_membership.add_argument("--scale-series-id", required=True)
    live_membership.add_argument("--cohort-seed", required=True)
    live_membership.add_argument("--window-start", required=True)
    live_membership.add_argument("--window-end", required=True)
    live_membership.add_argument("--source-revision-cutoff", required=True)
    live_membership.add_argument(
        "--cohort-sizes",
        default=",".join(str(value) for value in SCALE_COHORT_SIZES),
    )
    live_membership.add_argument("--output", type=Path, required=True)
    live_membership.add_argument("--confirm-local-clickhouse-write", action="store_true")

    census = subparsers.add_parser("scenario-pairs")
    census.add_argument("--input", type=Path, required=True)
    census.add_argument("--output", type=Path, required=True)

    survivors = subparsers.add_parser("survivors")
    survivors.add_argument("--input", type=Path, required=True)
    survivors.add_argument("--output-dir", type=Path, required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--root", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    report = subparsers.add_parser("report")
    report.add_argument("--input", type=Path, required=True)
    report.add_argument("--cohort-sizes", required=True)
    report.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "manifest-template":
        write_immutable_json(args.output, scale_manifest_template())
        return 0
    if args.command == "old-50k-decision":
        write_immutable_json(args.output, old_50k_reuse_decision())
        return 0
    if args.command == "check-fingerprint":
        payload = _object(args.input)
        if args.expected is None:
            result: Mapping[str, Any] = inspect_fingerprint(payload).to_dict()
        else:
            result = decide_artifact_reuse(payload, _object(args.expected)).to_dict()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0 if bool(result.get("complete", result.get("reusable"))) else 2
    if args.command == "rank-membership":
        return _rank_membership(args)
    if args.command == "prepare-membership-live":
        return _prepare_membership_live(args)
    if args.command == "scenario-pairs":
        return _scenario_pairs(args)
    if args.command == "survivors":
        return _survivors(args)
    if args.command == "validate":
        result = ScaleArtifactValidator(args.root).validate()
        payload = result.to_dict()
        if args.output is not None:
            write_immutable_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if result.passed else 2
    if args.command == "report":
        points = tuple(
            ActualScalePoint(**row)
            for row in _jsonl(args.input)
        )
        write_actual_only_reports(
            points=points,
            expected_cohort_sizes=_positive_ints(args.cohort_sizes),
            csv_path=args.output_dir / "preliminary-scale-points.csv",
            markdown_path=args.output_dir / "preliminary-scale-report.md",
            png_path=args.output_dir / "preliminary-scale-report.png",
        )
        return 0
    raise RuntimeError("unreachable command")


def _rank_membership(args: argparse.Namespace) -> int:
    payload = json.loads(args.input.read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        raw_user_ids = payload.get("user_ids")
    else:
        raw_user_ids = payload
    if not isinstance(raw_user_ids, list):
        raise ValueError("membership input must be an array or user_ids object")
    memberships = rank_cohort_members(
        scale_series_id=args.scale_series_id,
        cohort_seed=args.cohort_seed,
        user_ids=(str(value) for value in raw_user_ids),
    )
    sizes = tuple(
        value for value in _positive_ints(args.cohort_sizes) if value <= len(memberships)
    )
    hashes = validate_nested_prefixes(memberships, sizes)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite membership: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        for item in memberships:
            handle.write(json.dumps(asdict(item), sort_keys=True))
            handle.write("\n")
    write_immutable_json(
        args.prefix_hashes,
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "scale_series_id": args.scale_series_id,
            "available_user_count": len(memberships),
            "cohort_prefix_sha256": {str(key): value for key, value in hashes.items()},
        },
    )
    return 0


def _prepare_membership_live(args: argparse.Namespace) -> int:
    if not args.confirm_local_clickhouse_write:
        raise ValueError(
            "prepare-membership-live requires --confirm-local-clickhouse-write"
        )
    load_dotenv(args.env_file, override=False)
    settings = load_settings()
    hostname = (urlparse(settings.clickhouse_url).hostname or "").lower()
    if hostname not in {"localhost", "127.0.0.1", "::1", "host.docker.internal"}:
        raise ValueError("prepare-membership-live only accepts local ClickHouse")

    client = create_clickhouse_client(settings)
    try:
        existing = client.query(
            f"SELECT count() FROM {COHORT_MEMBERSHIP_RELATION} "
            "WHERE scale_series_id = {scale_series_id:String}",
            parameters={"scale_series_id": args.scale_series_id},
        ) if _relation_exists(client, COHORT_MEMBERSHIP_RELATION) else None
        if existing is not None and _first_scalar(existing) != 0:
            raise ValueError(
                "refusing to append to existing scale-series membership"
            )

        client.command(membership_create_sql())
        parameters = {
            "scale_series_id": args.scale_series_id,
            "cohort_seed": args.cohort_seed,
            "project_id": args.project_id,
            "vector_version": args.vector_version,
            "window_start": _parse_utc(args.window_start),
            "window_end": _parse_utc(args.window_end),
            "source_revision_cutoff": _parse_utc(args.source_revision_cutoff),
        }
        client.command(membership_population_sql(), parameters=parameters)
        summary = client.query(
            f"""
            SELECT
                count() AS row_count,
                uniqExact(user_id) AS user_count,
                min(cohort_rank) AS min_rank,
                max(cohort_rank) AS max_rank
            FROM {COHORT_MEMBERSHIP_RELATION}
            WHERE scale_series_id = {{scale_series_id:String}}
            """,
            parameters={"scale_series_id": args.scale_series_id},
        )
        row_count, user_count, min_rank, max_rank = _first_row(summary)
        if row_count <= 0 or (row_count, min_rank, max_rank) != (
            user_count,
            1,
            row_count,
        ):
            raise RuntimeError("prepared membership is not unique and contiguous")

        sizes = tuple(
            value
            for value in _positive_ints(args.cohort_sizes)
            if value <= row_count
        )
        if not sizes:
            raise RuntimeError("no requested cohort fits the prepared membership")
        prefix_digests = {size: hashlib.sha256() for size in sizes}
        complete_digest = hashlib.sha256()
        expected_rank = 1
        query = f"""
            SELECT cohort_rank, user_id
            FROM {COHORT_MEMBERSHIP_RELATION}
            WHERE scale_series_id = {{scale_series_id:String}}
            ORDER BY cohort_rank
        """
        with client.query_rows_stream(
            query,
            parameters={"scale_series_id": args.scale_series_id},
        ) as stream:
            for raw_rank, raw_user_id in stream:
                rank = int(raw_rank)
                if rank != expected_rank:
                    raise RuntimeError("membership stream is not contiguous")
                encoded = str(raw_user_id).encode("utf-8") + b"\n"
                complete_digest.update(encoded)
                for size, digest in prefix_digests.items():
                    if rank <= size:
                        digest.update(encoded)
                expected_rank += 1
        if expected_rank - 1 != row_count:
            raise RuntimeError("membership stream count changed during freeze")

        payload = {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "relation": COHORT_MEMBERSHIP_RELATION,
            "scale_series_id": args.scale_series_id,
            "cohort_seed": args.cohort_seed,
            "project_id": args.project_id,
            "vector_version": args.vector_version,
            "window_start": parameters["window_start"].isoformat(),
            "window_end": parameters["window_end"].isoformat(),
            "source_revision_cutoff": parameters[
                "source_revision_cutoff"
            ].isoformat(),
            "available_user_count": row_count,
            "membership_sha256": complete_digest.hexdigest(),
            "cohort_prefix_sha256": {
                str(size): digest.hexdigest()
                for size, digest in prefix_digests.items()
            },
        }
        write_immutable_json(args.output, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()


def _relation_exists(client: Any, relation: str) -> bool:
    result = client.query(
        "SELECT count() FROM system.tables "
        "WHERE database = currentDatabase() AND name = {relation:String}",
        parameters={"relation": relation},
    )
    return _first_scalar(result) == 1


def _first_row(result: Any) -> tuple[int, ...]:
    rows = getattr(result, "result_rows", None)
    if not rows:
        raise RuntimeError("ClickHouse query returned no rows")
    return tuple(int(value) for value in rows[0])


def _first_scalar(result: Any) -> int:
    return _first_row(result)[0]


def _parse_utc(value: str) -> datetime:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed.astimezone(UTC)


def _scenario_pairs(args: argparse.Namespace) -> int:
    records = tuple(
        ScenarioCensusRecord(
            scenario_id=str(row["scenario_id"]),
            definition_sha256=str(row["definition_sha256"]),
            candidate_type=str(row["candidate_type"]),
            scenario_set=ScenarioSet(str(row["scenario_set"])),
            corpus_user_count=int(row["corpus_user_count"]),
            hard_match_user_count=int(row["hard_match_user_count"]),
            estimated_member_count=float(row["estimated_member_count"]),
            exact_positive_count=int(row["exact_positive_count"]),
        )
        for row in _jsonl(args.input)
    )
    pairs = select_scenario_pairs(records)
    write_immutable_json(
        args.output,
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "pairs": [item.to_dict() for item in pairs],
            "fallback_count": sum(not item.validated for item in pairs),
        },
    )
    return 0


def _survivors(args: argparse.Namespace) -> int:
    candidates = tuple(
        ScreeningCandidate(
            scenario_id=str(row["scenario_id"]),
            candidate_type=str(row["candidate_type"]),
            scenario_set=ScenarioSet(str(row["scenario_set"])),
            corpus_user_count=int(row["corpus_user_count"]),
            requested_k=int(row["requested_k"]),
            exact_positive_count=int(row["exact_positive_count"]),
            recall=float(row["recall"]),
            ann_p95_ms=float(row["ann_p95_ms"]),
            exact_all_p95_ms=float(row["exact_all_p95_ms"]),
            filter_first_p95_ms=(
                float(row["filter_first_p95_ms"])
                if row.get("filter_first_p95_ms") is not None
                else None
            ),
            filter_first_results_equal=bool(row["filter_first_results_equal"]),
            hnsw_index_used=bool(row["hnsw_index_used"]),
            temp_spill=bool(row["temp_spill"]),
            oom=bool(row["oom"]),
        )
        for row in _jsonl(args.input)
    )
    selected = select_survivors(candidates)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, values in (
        ("scale-hnsw-tuning-inputs.json", selected.scale),
        ("policy-hnsw-tuning-inputs.json", selected.policy),
        ("goal2-tuning-inputs.json", selected.goal2_union),
    ):
        write_immutable_json(
            args.output_dir / filename,
            {
                "experiment_version": SCALE_EXPERIMENT_VERSION,
                "cells": [item.to_dict() for item in values],
            },
        )
    return 0


def _object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _jsonl(path: Path) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at line {line_number}")
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise ValueError(f"JSONL row {line_number} must be an object")
        result.append(payload)
    return result


def _positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise ValueError("cohort sizes must be positive integers")
    return values


if __name__ == "__main__":
    raise SystemExit(main())
