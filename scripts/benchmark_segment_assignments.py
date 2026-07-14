#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from offline_evaluation.assignment_benchmark import (  # noqa: E402
    BenchmarkReport,
    run_benchmark,
    write_benchmark_report,
)
from offline_evaluation.assignment_corpus import (  # noqa: E402
    SUPPORTED_DISTRIBUTIONS,
    controlled_thread_environment,
    current_git_commit,
    generate_synthetic_corpus,
    load_frozen_corpus,
)
from offline_evaluation.assignment_matchers import MatcherConfig  # noqa: E402
from offline_evaluation.assignment_shadow import (  # noqa: E402
    run_assignment_shadow,
    write_shadow_report,
)


SMOKE_USER_COUNTS = (256,)
SMOKE_SEGMENT_COUNTS = (256,)
FULL_USER_COUNTS = (10_000, 100_000)
FULL_SEGMENT_COUNTS = (50, 256, 1_000, 5_000)
SUMMARY_FORMAT_VERSION = "loopad.assignment-benchmark-summary.v2"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark exact and FAISS HNSW assignment matchers.",
    )
    parser.add_argument("--corpus", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/assignment-benchmark"))
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--seed", type=int, default=20260714)
    parser.add_argument("--source-cutoff", default="2026-01-01T00:00:00Z")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--warmup-users", type=int, default=16)
    parser.add_argument("--timing-trials", type=int, default=3)
    parser.add_argument("--candidate-k", type=int, default=50)
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--ef-construction", type=int, default=200)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--threshold-band", type=float, default=0.02)
    parser.add_argument("--rescue-margin", type=float, default=0.02)
    parser.add_argument("--faiss-threads", type=int, default=1)
    parser.add_argument("--blas-threads", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.corpus is not None and args.full:
        parser.error("--corpus and --full cannot be combined")
    if args.warmup_users < 0:
        parser.error("--warmup-users must be non-negative")
    if args.timing_trials < 2:
        parser.error("--timing-trials must be at least 2")
    if args.faiss_threads <= 0:
        parser.error("--faiss-threads must be positive")
    if args.blas_threads <= 0:
        parser.error("--blas-threads must be positive")

    # These modules import NumPy and FAISS lazily. Set process-start controls
    # before corpus materialization so CLI shadow runs and spawned matcher
    # workers observe the same requested native thread count.
    os.environ.update(controlled_thread_environment(args.blas_threads))

    config = MatcherConfig(
        threshold=args.threshold,
        exact_batch_size=args.batch_size,
        candidate_k=args.candidate_k,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.ef_construction,
        hnsw_ef_search=args.ef_search,
        rescue_threshold_band=args.threshold_band,
        rescue_margin=args.rescue_margin,
        faiss_threads=args.faiss_threads,
        blas_threads=args.blas_threads,
    )
    mode = (
        "provided_corpus"
        if args.corpus is not None
        else "full" if args.full else "smoke"
    )
    summary_path = args.output_dir / "summary.json"
    _begin_summary(summary_path, mode=mode, overwrite=args.force)
    completed: list[tuple[str, int, int, BenchmarkReport, Path]] = []
    if args.corpus is not None:
        corpus = load_frozen_corpus(args.corpus)
        report = run_benchmark(
            corpus,
            config=config,
            warmup_user_count=args.warmup_users,
            timing_trial_count=args.timing_trials,
        )
        destination = args.output_dir / "provided-corpus.benchmark.json"
        write_benchmark_report(report, destination, overwrite=args.force)
        completed.append(
            (
                corpus.manifest.distribution,
                corpus.manifest.user_count,
                corpus.manifest.segment_count,
                report,
                destination,
            )
        )
        if args.shadow:
            write_shadow_report(
                run_assignment_shadow(corpus, config=config),
                args.output_dir / "provided-corpus.shadow.json",
                overwrite=args.force,
            )
    else:
        matrix = _matrix(full=args.full)
        commit = current_git_commit(REPOSITORY_ROOT)
        for case_index, (users, segments, distribution) in enumerate(matrix):
            corpus = generate_synthetic_corpus(
                user_count=users,
                segment_count=segments,
                distribution=distribution,
                random_seed=args.seed + case_index,
                source_cutoff_at=args.source_cutoff,
                git_commit=commit,
                matcher_config=config.to_dict(),
                threshold=config.threshold,
            )
            stem = f"{distribution}-u{users}-s{segments}"
            report = run_benchmark(
                corpus,
                config=config,
                warmup_user_count=args.warmup_users,
                timing_trial_count=args.timing_trials,
            )
            destination = args.output_dir / f"{stem}.benchmark.json"
            write_benchmark_report(report, destination, overwrite=args.force)
            completed.append((distribution, users, segments, report, destination))
            if args.shadow:
                write_shadow_report(
                    run_assignment_shadow(corpus, config=config),
                    args.output_dir / f"{stem}.shadow.json",
                    overwrite=args.force,
                )

    summary = _summary(
        completed,
        mode=mode,
        config=config,
    )
    _write_json(summary_path, summary, overwrite=True)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0


def _matrix(*, full: bool) -> Iterable[tuple[int, int, str]]:
    user_counts = FULL_USER_COUNTS if full else SMOKE_USER_COUNTS
    segment_counts = FULL_SEGMENT_COUNTS if full else SMOKE_SEGMENT_COUNTS
    return (
        (users, segments, distribution)
        for users in user_counts
        for segments in segment_counts
        for distribution in SUPPORTED_DISTRIBUTIONS
    )


def _summary(
    completed: list[tuple[str, int, int, BenchmarkReport, Path]],
    *,
    mode: str,
    config: MatcherConfig,
) -> dict[str, Any]:
    if mode not in {"provided_corpus", "smoke", "full"}:
        raise ValueError(f"unsupported benchmark summary mode: {mode}")
    case_keys = [(value[0], value[1], value[2]) for value in completed]
    if len(case_keys) != len(set(case_keys)):
        raise ValueError("benchmark summary contains duplicate cases")
    code_hashes = {
        value[3].benchmark_code_identity["benchmark_source_sha256"]
        for value in completed
    }
    if len(code_hashes) > 1:
        raise ValueError("benchmark summary cannot mix different source identities")
    if any(value[3].matcher_config != config.to_dict() for value in completed):
        raise ValueError("benchmark summary cannot mix matcher configurations")
    requested_warmup_counts = {
        value[3].requested_warmup_user_count for value in completed
    }
    warmup_counts = {value[3].warmup_user_count for value in completed}
    timing_trial_counts = {value[3].timing_trial_count for value in completed}
    timing_trial_schedules = {
        value[3].timing_trial_schedule for value in completed
    }
    if (
        len(requested_warmup_counts) > 1
        or len(timing_trial_counts) > 1
        or len(timing_trial_schedules) > 1
    ):
        raise ValueError("benchmark summary cannot mix timing protocols")
    cases: list[dict[str, Any]] = []
    for distribution, users, segments, report, destination in completed:
        manifest = report.corpus_manifest
        if (
            manifest["distribution"] != distribution
            or manifest["user_count"] != users
            or manifest["segment_count"] != segments
        ):
            raise ValueError("benchmark case metadata does not match its report")
        if report.warmup_user_count != min(
            report.requested_warmup_user_count,
            users,
        ):
            raise ValueError("benchmark report records an invalid effective warmup")
        cases.append(
            {
                "distribution": distribution,
                "user_count": users,
                "segment_count": segments,
                "report": str(destination),
                "timing": {
                    result.matcher: asdict(result.timing) for result in report.matchers
                },
                "agreement": {
                    result.matcher: asdict(result.agreement) for result in report.matchers
                },
            }
        )
    crossover_observations: list[dict[str, Any]] = []
    if mode != "provided_corpus":
        group_keys = sorted({(value[0], value[1]) for value in completed})
        for distribution, users in group_keys:
            group = sorted(
                (
                    value
                    for value in completed
                    if value[0] == distribution and value[1] == users
                ),
                key=lambda value: value[2],
            )
            for matcher_index in (1, 2):
                crossover_observations.append(
                    _crossover_observation(
                        distribution=distribution,
                        user_count=users,
                        matcher_index=matcher_index,
                        group=group,
                    )
                )
    return {
        "format_version": SUMMARY_FORMAT_VERSION,
        "status": "complete",
        "mode": mode,
        "executed_case_count": len(cases),
        "matcher_config": config.to_dict(),
        "benchmark_code_identity": (
            completed[0][3].benchmark_code_identity if completed else None
        ),
        "requested_warmup_user_count": (
            completed[0][3].requested_warmup_user_count if completed else None
        ),
        "effective_warmup_user_counts": sorted(warmup_counts),
        "timing_trial_count": (
            completed[0][3].timing_trial_count if completed else None
        ),
        "timing_trial_schedule": (
            completed[0][3].timing_trial_schedule if completed else None
        ),
        "cases": cases,
        "crossover_observations": crossover_observations,
        "note": "Only executed cases are reported; no unexecuted numbers are synthesized.",
    }


def _begin_summary(path: Path, *, mode: str, overwrite: bool) -> None:
    _write_json(
        path,
        {
            "format_version": SUMMARY_FORMAT_VERSION,
            "status": "incomplete",
            "mode": mode,
            "note": (
                "The runner replaces this marker only after every case and optional "
                "shadow report completes. Do not use it as benchmark evidence."
            ),
        },
        overwrite=overwrite,
    )


def _crossover_observation(
    *,
    distribution: str,
    user_count: int,
    matcher_index: int,
    group: list[tuple[str, int, int, BenchmarkReport, Path]],
) -> dict[str, Any]:
    evaluated_segment_counts = [value[2] for value in group]
    faster_than_exact = [
        value[3].matchers[matcher_index].timing.end_to_end_p95_ms
        < value[3].matchers[0].timing.end_to_end_p95_ms
        for value in group
    ]
    minimum_faster = next(
        (
            segment_count
            for segment_count, is_faster in zip(
                evaluated_segment_counts,
                faster_than_exact,
                strict=True,
            )
            if is_faster
        ),
        None,
    )
    bracketed_crossover: int | None = None
    for index in range(1, len(group)):
        if (
            not any(faster_than_exact[:index])
            and all(faster_than_exact[index:])
        ):
            bracketed_crossover = evaluated_segment_counts[index]
            break

    if bracketed_crossover is not None:
        status = "observed_bracketed_trial_p95"
    elif not any(faster_than_exact):
        status = "not_observed"
    elif all(faster_than_exact):
        status = "unbracketed_faster_at_minimum"
    else:
        status = "non_monotonic_or_unstable"

    return {
        "distribution": distribution,
        "user_count": user_count,
        "matcher": group[0][3].matchers[matcher_index].matcher,
        "timing_comparison_basis": "end_to_end_p95_ms_across_trials",
        "evaluated_segment_counts": evaluated_segment_counts,
        "faster_than_exact_by_segment_count": faster_than_exact,
        "minimum_observed_faster_segment_count": minimum_faster,
        "observed_bracketed_crossover_segment_count": bracketed_crossover,
        "status": status,
    }


def _write_json(path: Path, payload: Any, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if overwrite else "x"
    with path.open(mode, encoding="utf-8") as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")


if __name__ == "__main__":
    raise SystemExit(main())
