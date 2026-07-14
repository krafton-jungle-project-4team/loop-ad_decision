#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from offline_evaluation.assignment_corpus import (  # noqa: E402
    SUPPORTED_DISTRIBUTIONS,
    generate_synthetic_corpus,
)
from offline_evaluation.pgvector_assignment_benchmark import (  # noqa: E402
    benchmark_code_identity,
    benchmark_matrix,
    run_pgvector_benchmark_case,
    summarize_benchmark_cases,
    validate_local_postgres_dsn,
    write_json_report,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the production pgvector repository/matcher against the "
            "production exact matcher on a disposable localhost PostgreSQL session."
        )
    )
    parser.add_argument(
        "--dsn",
        required=True,
        help="localhost PostgreSQL DSN with the vector extension already installed",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/pgvector-assignment-benchmark"),
    )
    parser.add_argument("--representative", action="store_true")
    parser.add_argument(
        "--include-100k",
        action="store_true",
        help="also run the optional U=100K representative matrix",
    )
    parser.add_argument(
        "--distribution",
        action="append",
        choices=SUPPORTED_DISTRIBUTIONS,
        dest="distributions",
    )
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--timing-trials", type=int, default=3)
    parser.add_argument("--warmup-users", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    try:
        dsn = validate_local_postgres_dsn(args.dsn)
    except ValueError as exc:
        parser.error(str(exc))
    if args.include_100k and not args.representative:
        parser.error("--include-100k requires --representative")
    if args.timing_trials < 2:
        parser.error("--timing-trials must be at least 2")
    if args.warmup_users < 0:
        parser.error("--warmup-users must be non-negative")

    distributions = tuple(args.distributions or SUPPORTED_DISTRIBUTIONS)
    mode = "representative" if args.representative else "smoke"
    matrix = benchmark_matrix(
        representative=args.representative,
        include_100k=args.include_100k,
        distributions=distributions,
    )
    code_identity = benchmark_code_identity(REPOSITORY_ROOT)
    cases: list[dict[str, object]] = []
    for case_index, (user_count, segment_count, distribution) in enumerate(matrix):
        corpus = generate_synthetic_corpus(
            user_count=user_count,
            segment_count=segment_count,
            distribution=distribution,
            random_seed=args.seed + case_index,
            git_commit=str(code_identity["git_commit"] or "unknown"),
            matcher_config={
                "backend": "production_pgvector_repository",
                "production_policy": "exact_only",
                "timing_trial_count": args.timing_trials,
            },
        )
        report = run_pgvector_benchmark_case(
            dsn,
            corpus,
            timing_trial_count=args.timing_trials,
            warmup_user_count=args.warmup_users,
            repository_root=REPOSITORY_ROOT,
        )
        stem = f"{distribution}-u{user_count}-s{segment_count}"
        destination = args.output_dir / f"{stem}.pgvector.json"
        write_json_report(report, destination, overwrite=args.force)
        cases.append(
            {
                "corpus_manifest": report["corpus_manifest"],
                "timing": report["timing"],
                "agreement": report["agreement"],
                "rescue": report["rescue"],
                "database": report["database"],
                "report": str(destination),
            }
        )
        print(
            json.dumps(
                {
                    "case": stem,
                    "production_policy": "exact_only",
                    "report": str(destination),
                    "status": "complete",
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    summary = summarize_benchmark_cases(
        cases,
        mode=mode,
        include_100k=args.include_100k,
    )
    summary_path = args.output_dir / "summary.json"
    write_json_report(summary, summary_path, overwrite=args.force)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
