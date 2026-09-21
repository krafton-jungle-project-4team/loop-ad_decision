#!/usr/bin/env python3
"""Sequential, checkpointed Goal 3 ANN efficiency-recovery runner.

This runner never changes the application runtime, API, DTO, Data Contract,
or persistent PostgreSQL configuration.  It opens one disposable cohort at a
time and applies HNSW controls with ``SET LOCAL`` inside each transaction.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import UTC, datetime
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.analysis.audience_search_repository import _vector_literal  # noqa: E402
from app.analysis.repositories import PsycopgPostgresExecutor  # noqa: E402
from app.config import load_settings  # noqa: E402
from app.db import create_clickhouse_client, create_postgres_connection  # noqa: E402
from offline_evaluation.ann_search_benchmark import (  # noqa: E402
    BenchmarkManifest,
    CellRunConfig,
    LiveAnnSearchBenchmark,
    _BackendRssSampler,
    _ann_params,
    _ann_sql,
    _candidates,
    _find_plan_values,
)
from offline_evaluation.ann_search_experiment import (  # noqa: E402
    BenchmarkPhase,
    CacheMode,
    HNSW_INDEX_NAME,
    HnswSettings,
    SearchPlan,
    summarize_observations,
)
from offline_evaluation.ann_search_goal3 import (  # noqa: E402
    COHORT_SIZES,
    FULL_HNSW_SETTINGS,
    GOAL1_ROOT,
    GOAL3_EXPERIMENT_VERSION,
    GOAL3_ROOT,
    REQUIRED_K_VALUES,
    append_jsonl_once,
    build_experiment_plan,
    build_integrity_report,
    canonical_json_sha256,
    combine_kernel_confirmations,
    crossover_report,
    empty_halving_partitions,
    k_provenance,
    kernel_final_gate,
    kernel_screen_gate,
    measurement_fingerprint,
    pareto_kernel_settings,
    read_json,
    read_jsonl,
    render_summary,
    scenario_splits,
    summarize_kernel_rows,
    part_b_gate,
    validate_halving_isolation,
    write_immutable_json,
)
from offline_evaluation.ann_search_goal3_artifacts import (  # noqa: E402
    validate_graph_sources,
    write_goal3_figures,
)
from offline_evaluation.ann_search_scale_artifacts import sha256_file  # noqa: E402
from offline_evaluation.ann_search_scale_series import ScaleCohortScope  # noqa: E402
from scripts.prepare_ann_scale_postgres import cohort_database_name  # noqa: E402


DEFAULT_ENV_FILE = Path(".env.ann-source.local")
DEFAULT_POSTGRES_PREFIX = "loopad_ann_v2"
DEFAULT_POSTGRES_CONTAINER = "loop-ad_data-source_contract-postgres-1"
DEFAULT_CLICKHOUSE_DATABASE = "ann_scale_v2_build"
DEFAULT_SCALE_SERIES_ID = "expedia-full-2015-v2-86ded0d7"
DEFAULT_SAMPLE_SEED = "ann-score-pass-v2-20260718"
RSS_CORRECTED_SCREENING_PHASE = "screening_rss_corrected"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("plan", "calibrate", "screen", "kernel-confirm", "kernel-cold", "part-b-preconfirm", "part-b-confirm", "finalize", "repair-fallback-audit", "all"),
    )
    parser.add_argument("--goal1-root", type=Path, default=GOAL1_ROOT)
    parser.add_argument("--output-root", type=Path, default=GOAL3_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--postgres-prefix", default=DEFAULT_POSTGRES_PREFIX)
    parser.add_argument("--postgres-container", default=DEFAULT_POSTGRES_CONTAINER)
    parser.add_argument("--clickhouse-database", default=DEFAULT_CLICKHOUSE_DATABASE)
    parser.add_argument("--scale-series-id", default=DEFAULT_SCALE_SERIES_ID)
    parser.add_argument("--reference-sample-seed", default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--cohort-sizes",
        type=_positive_ints,
        default=COHORT_SIZES,
        help="Must remain the full frozen Goal 1 scale series for the final run.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_full_scale(args)
    if args.command in {"plan", "all"}:
        build_plan(args)
    if args.command in {"calibrate", "all"}:
        run_calibration(args)
    if args.command in {"screen", "all"}:
        run_screening(args)
    if args.command in {"kernel-confirm", "all"}:
        run_kernel_confirmation(args)
    if args.command in {"kernel-cold", "all"}:
        run_kernel_cold(args)
    if args.command in {"part-b-preconfirm", "all"}:
        run_part_b_preconfirmation(args)
    if args.command in {"part-b-confirm", "all"}:
        run_part_b_confirmation(args)
    if args.command in {"finalize", "all"}:
        finalize(args)
    if args.command == "repair-fallback-audit":
        repair_fallback_audit(args)
    return 0


def build_plan(args: argparse.Namespace) -> None:
    manifest_path = args.goal1_root / "phase2/scenario-manifest.json"
    manifest_payload = read_json(manifest_path)
    plan = build_experiment_plan(goal1_root=args.goal1_root, manifest=manifest_payload)
    code_paths = (
        ROOT / "offline_evaluation/ann_search_benchmark.py",
        ROOT / "offline_evaluation/ann_search_goal3.py",
        ROOT / "scripts/run_ann_scale_goal3.py",
    )
    fingerprint = measurement_fingerprint(
        goal1_root=args.goal1_root, code_paths=code_paths
    )
    root = args.output_root
    invalidation = {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "invalidated_goal": "Goal 2 scale-scope ann_first",
        "invalidated_measurements": [
            "ANN precision",
            "ANN end-to-end latency",
            "ANN winner/failure decision",
            "all threshold-omitting ann_first observations",
        ],
        "reason": (
            "Goal 2 ANN membership applied hard predicates without applying the "
            "scenario score threshold. The corrected path now uses HNSW IDs -> "
            "exact score-threshold filter -> hard predicate filter."
        ),
        "reusable_inputs": [
            "source/vector manifest",
            "nested cohorts",
            "scenario definitions",
            "query vectors",
            "production compiler/calibration",
            "HNSW DDL",
        ],
    }
    dry_run = dict(plan["dry_run_estimate"])
    dry_run.update(
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "estimated_wall_clock": "reported after first completed calibration checkpoint; no synthetic ETA is used in conclusions",
            "artifact_root": str(root),
        }
    )
    write_immutable_json(root / "experiment-plan.json", plan)
    write_immutable_json(root / "experiment-fingerprint.json", fingerprint)
    write_immutable_json(root / "prior-goal2-invalidation.json", invalidation)
    write_immutable_json(root / "k-provenance.json", k_provenance(args.goal1_root))
    write_immutable_json(
        root / "tuning-confirmation-split.json",
        {
            "experiment_version": GOAL3_EXPERIMENT_VERSION,
            "scenario_manifest": str(manifest_path),
            "scenario_manifest_sha256": sha256_file(manifest_path),
            "splits": [item.to_dict() for item in scenario_splits(manifest_payload)],
        },
    )
    halving = empty_halving_partitions(plan)
    write_immutable_json(root / "halving-partitions.json", halving)
    write_immutable_json(
        root / "halving-isolation-validation.json", validate_halving_isolation(halving)
    )
    write_immutable_json(root / "dry-run-estimate.json", dry_run)
    (root / "checkpoints").mkdir(parents=True, exist_ok=True)
    _progress("goal3_plan_locked", root=str(root), partitions=dry_run["partition_count"])


def run_calibration(args: argparse.Namespace) -> None:
    plan = _plan(args)
    _lock_execution_code_fingerprint(args, phase="calibration")
    provenance = _k_provenance_map(args)
    manifest = BenchmarkManifest.load(args.goal1_root / "phase2/scenario-manifest.json")
    splits = scenario_splits(read_json(args.goal1_root / "phase2/scenario-manifest.json"))
    raw_path = args.output_root / "kernel-raw.jsonl"
    results_path = args.output_root / "kernel-results.jsonl"
    completed = {_calibration_key(row) for row in read_jsonl(results_path) if row.get("phase") == "calibration"}
    total = len(splits) * len(args.cohort_sizes) * len(REQUIRED_K_VALUES)
    done = 0
    for cohort_size in args.cohort_sizes:
        with _live_scope(args, cohort_size, manifest) as live:
            for split in splits:
                scenario = manifest.require_scenario(split.tuning_scenario_id)
                for requested_k in REQUIRED_K_VALUES:
                    key = (split.candidate_type, split.tuning_scenario_id, cohort_size, requested_k)
                    if key in completed:
                        done += 1
                        continue
                    rows = _run_kernel_scope(
                        live=live,
                        scenario=scenario,
                        candidate_type=split.candidate_type,
                        requested_k=requested_k,
                        settings=FULL_HNSW_SETTINGS,
                        phase="calibration",
                        warmups=2,
                        measured=5,
                        k_provenance=provenance[requested_k],
                        interleaved=False,
                        diagnostics=False,
                    )
                    append_jsonl_once(
                        raw_path,
                        rows,
                        identity_fields=("measurement_id",),
                    )
                    summaries = summarize_kernel_rows(rows)
                    append_jsonl_once(
                        results_path,
                        summaries,
                        identity_fields=(
                            "phase", "candidate_type", "query_id", "corpus_user_count",
                            "requested_k", "hnsw",
                        ),
                    )
                    _write_checkpoint(
                        args,
                        phase="calibration",
                        cohort_size=cohort_size,
                        candidate_type=split.candidate_type,
                        query_id=split.tuning_scenario_id,
                        requested_k=requested_k,
                        raw_rows=rows,
                        results=summaries,
                    )
                    done += 1
                    _progress("kernel_calibration_checkpoint", completed=done, total=total, cohort_size=cohort_size, candidate_type=split.candidate_type, requested_k=requested_k)


def run_screening(args: argparse.Namespace) -> None:
    _plan(args)
    _lock_execution_code_fingerprint(args, phase=RSS_CORRECTED_SCREENING_PHASE)
    manifest = BenchmarkManifest.load(args.goal1_root / "phase2/scenario-manifest.json")
    splits = scenario_splits(read_json(args.goal1_root / "phase2/scenario-manifest.json"))
    provenance = _k_provenance_map(args)
    calibration = [row for row in read_jsonl(args.output_root / "kernel-results.jsonl") if row.get("phase") == "calibration"]
    by_scope: dict[tuple[str, str, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in calibration:
        by_scope[(str(row["candidate_type"]), str(row["query_id"]), int(row["corpus_user_count"]), int(row["requested_k"]))].append(row)
    raw_path = args.output_root / "kernel-raw.jsonl"
    result_path = args.output_root / "kernel-results.jsonl"
    # Earlier screening rows are retained for audit only: their RSS sampler
    # had no container binding, so both plan peaks were null and the rows are
    # not valid operational evidence.  The corrected phase gets independent
    # append-only IDs and is the only screening source used downstream.
    legacy_rows = [
        row
        for row in read_jsonl(result_path)
        if row.get("phase") == "screening"
    ]
    write_immutable_json(
        args.output_root / "screening-rss-invalidation.json",
        {
            "experiment_version": GOAL3_EXPERIMENT_VERSION,
            "invalidated_phase": "screening",
            "invalidated_result_row_count": len(legacy_rows),
            "reason": (
                "the offline sampler was not passed ANN_POSTGRES_CONTAINER, "
                "so exact_peak_rss_bytes and ann_peak_rss_bytes were null; "
                "these rows cannot satisfy the required RSS operational check"
            ),
            "replacement_phase": RSS_CORRECTED_SCREENING_PHASE,
        },
    )
    completed = {
        _calibration_key(row)
        for row in read_jsonl(result_path)
        if row.get("phase") == RSS_CORRECTED_SCREENING_PHASE
    }
    for cohort_size in args.cohort_sizes:
        with _live_scope(args, cohort_size, manifest) as live:
            for split in splits:
                scenario = manifest.require_scenario(split.tuning_scenario_id)
                for requested_k in REQUIRED_K_VALUES:
                    key = (split.candidate_type, split.tuning_scenario_id, cohort_size, requested_k)
                    if key in completed:
                        continue
                    local = by_scope.get(key, [])
                    if not local:
                        raise RuntimeError(f"missing calibration evidence for screening scope {key}")
                    selected = pareto_kernel_settings(local)
                    # If quality failed for every local setting, preserve the full
                    # scope as an actual negative result rather than silently
                    # dropping a K/query/N partition.
                    settings = tuple(
                        HnswSettings(
                            int(row["hnsw"]["ef_search"]),
                            str(row["hnsw"]["iterative_scan"]),
                            int(row["hnsw"]["max_scan_tuples"]),
                        )
                        for row in (selected or local)
                    )
                    rows = _run_kernel_scope(
                        live=live,
                        scenario=scenario,
                        candidate_type=split.candidate_type,
                        requested_k=requested_k,
                        settings=settings,
                        phase=RSS_CORRECTED_SCREENING_PHASE,
                        warmups=5,
                        measured=30,
                        k_provenance=provenance[requested_k],
                        interleaved=True,
                        diagnostics=True,
                        diagnostics_root=args.output_root / "checkpoints" / "explain",
                    )
                    append_jsonl_once(raw_path, rows, identity_fields=("measurement_id",))
                    summaries = [kernel_screen_gate(row) for row in summarize_kernel_rows(rows)]
                    append_jsonl_once(
                        result_path,
                        summaries,
                        identity_fields=("phase", "candidate_type", "query_id", "corpus_user_count", "requested_k", "hnsw"),
                    )
                    _write_checkpoint(args, phase=RSS_CORRECTED_SCREENING_PHASE, cohort_size=cohort_size, candidate_type=split.candidate_type, query_id=split.tuning_scenario_id, requested_k=requested_k, raw_rows=rows, results=summaries)
                    _progress("kernel_screening_checkpoint", cohort_size=cohort_size, candidate_type=split.candidate_type, requested_k=requested_k, active_settings=len(settings))


def run_kernel_confirmation(args: argparse.Namespace) -> None:
    _plan(args)
    _lock_execution_code_fingerprint(
        args, phase="kernel-confirmation-diagnostic-resume"
    )
    manifest = BenchmarkManifest.load(args.goal1_root / "phase2/scenario-manifest.json")
    splits = scenario_splits(read_json(args.goal1_root / "phase2/scenario-manifest.json"))
    split_by_type = {item.candidate_type: item for item in splits}
    provenance = _k_provenance_map(args)
    screening = [
        row
        for row in read_jsonl(args.output_root / "kernel-results.jsonl")
        if row.get("phase") == RSS_CORRECTED_SCREENING_PHASE
    ]
    selected = [row for row in screening if bool(row.get("screening_gate_passed")) or bool(row.get("boundary_near"))]
    # A setting selected at a smaller N can also be selected at a larger N.
    # Confirmation is one observation per actual (type, query, N, K, setting),
    # not one observation per originating screening cell, so union targets
    # before opening the sequential live scope.
    targets: dict[tuple[str, int, int], set[HnswSettings]] = defaultdict(set)
    for row in selected:
        candidate_type = str(row["candidate_type"])
        split = split_by_type[candidate_type]
        if not split.independent_confirmation_query:
            continue
        setting = HnswSettings(
            int(row["hnsw"]["ef_search"]),
            str(row["hnsw"]["iterative_scan"]),
            int(row["hnsw"]["max_scan_tuples"]),
        )
        for confirmation_size in (
            size for size in args.cohort_sizes if size >= int(row["corpus_user_count"])
        ):
            targets[(candidate_type, int(row["requested_k"]), confirmation_size)].add(
                setting
            )
    raw_path = args.output_root / "kernel-raw.jsonl"
    result_path = args.output_root / "kernel-confirmation.jsonl"
    completed = {_confirmation_key(row) for row in read_jsonl(result_path)}
    for candidate_type, requested_k, confirmation_size in sorted(targets):
        split = split_by_type[candidate_type]
        if split.confirmation_scenario_id is None:
            continue
        pending_settings = tuple(
            setting
            for setting in sorted(
                targets[(candidate_type, requested_k, confirmation_size)],
                key=lambda item: (item.ef_search, item.iterative_scan, item.max_scan_tuples),
            )
            if (
                candidate_type,
                split.confirmation_scenario_id,
                confirmation_size,
                requested_k,
                setting.ef_search,
                setting.iterative_scan,
                setting.max_scan_tuples,
            ) not in completed
        )
        if not pending_settings:
            continue
        with _live_scope(args, confirmation_size, manifest) as live:
            rows = _run_kernel_scope(
                live=live,
                scenario=manifest.require_scenario(split.confirmation_scenario_id),
                candidate_type=candidate_type,
                requested_k=requested_k,
                settings=pending_settings,
                phase="confirmation_warm",
                warmups=10,
                measured=300,
                k_provenance=provenance[requested_k],
                interleaved=True,
                diagnostics=True,
                diagnostics_root=args.output_root / "checkpoints" / "explain",
            )
            append_jsonl_once(raw_path, rows, identity_fields=("measurement_id",))
            summaries = [kernel_final_gate(row) for row in summarize_kernel_rows(rows)]
            append_jsonl_once(result_path, summaries, identity_fields=("phase", "candidate_type", "query_id", "corpus_user_count", "requested_k", "hnsw"))
            _write_checkpoint(args, phase="kernel-confirmation", cohort_size=confirmation_size, candidate_type=candidate_type, query_id=split.confirmation_scenario_id, requested_k=requested_k, raw_rows=rows, results=summaries)
            completed.update(_confirmation_key(row) for row in summaries)
            _progress("kernel_confirmation_warm_checkpoint", cohort_size=confirmation_size, candidate_type=candidate_type, requested_k=requested_k, settings=len(pending_settings))
    # DB-cold runs are intentionally a distinct command/checkpoint in a later
    # runner phase: each observation requires an external PostgreSQL restart.
    # A warm-only result cannot be marked passed by finalize().


def run_kernel_cold(args: argparse.Namespace) -> None:
    """Collect 30 truly DB-cold exact/ANN pairs for warm-confirmed kernels."""

    _plan(args)
    _lock_execution_code_fingerprint(args, phase="kernel-cold")
    manifest = BenchmarkManifest.load(args.goal1_root / "phase2/scenario-manifest.json")
    warm_rows = [
        row
        for row in read_jsonl(args.output_root / "kernel-confirmation.jsonl")
        if bool(row.get("ann_candidate_retrieval_passed"))
    ]
    raw_path = args.output_root / "kernel-cold-raw.jsonl"
    result_path = args.output_root / "kernel-cold-confirmation.jsonl"
    existing = read_jsonl(raw_path)
    done = {
        _cold_key(row)
        for row in existing
        if row.get("phase") == "confirmation_cold" and bool(row.get("measured"))
    }
    for warm in warm_rows:
        candidate_type = str(warm["candidate_type"])
        scenario_id = str(warm["query_id"])
        cohort_size = int(warm["corpus_user_count"])
        requested_k = int(warm["requested_k"])
        hnsw = _hnsw_from_dict(warm["hnsw"])
        scenario = manifest.require_scenario(scenario_id)
        diagnostic = _existing_kernel_diagnostic(
            args=args,
            phase="confirmation_warm",
            scenario_id=scenario_id,
            cohort_size=cohort_size,
            requested_k=requested_k,
            hnsw=hnsw,
        )
        rows_for_summary = [
            row
            for row in existing
            if _cold_target_key(row)
            == (candidate_type, scenario_id, cohort_size, requested_k, hnsw)
        ]
        for iteration in range(30):
            if (candidate_type, scenario_id, cohort_size, requested_k, hnsw, iteration) in done:
                continue
            attempts: dict[str, Mapping[str, Any]] = {}
            order = ("ann", "exact") if iteration % 2 else ("exact", "ann")
            for plan in order:
                _restart_postgres_and_wait(args.postgres_container)
                with _live_scope(args, cohort_size, manifest) as live:
                    attempts[plan] = _topk_attempt(
                        live=live,
                        scenario=scenario,
                        requested_k=requested_k,
                        hnsw=hnsw if plan == "ann" else None,
                    )
            row = _kernel_row(
                phase="confirmation_cold",
                measured=True,
                iteration=iteration,
                candidate_type=candidate_type,
                scenario=scenario,
                cohort_size=cohort_size,
                requested_k=requested_k,
                k_provenance=str(warm["k_provenance"]),
                hnsw=hnsw,
                exact=attempts["exact"],
                ann=attempts["ann"],
                diagnostic=diagnostic,
            )
            append_jsonl_once(raw_path, (row,), identity_fields=("measurement_id",))
            rows_for_summary.append(row)
            done.add((candidate_type, scenario_id, cohort_size, requested_k, hnsw, iteration))
            _progress("kernel_confirmation_cold_iteration", candidate_type=candidate_type, cohort_size=cohort_size, requested_k=requested_k, iteration=iteration + 1)
        if len(rows_for_summary) != 30:
            raise RuntimeError("DB-cold kernel confirmation is incomplete")
        summary = kernel_final_gate(summarize_kernel_rows(rows_for_summary)[0])
        summary = {**summary, "warm_confirmation_passed": True, "db_cold_confirmation_completed": True}
        append_jsonl_once(result_path, (summary,), identity_fields=("phase", "candidate_type", "query_id", "corpus_user_count", "requested_k", "hnsw"))
        _write_checkpoint(args, phase="kernel-cold-confirmation", cohort_size=cohort_size, candidate_type=candidate_type, query_id=scenario_id, requested_k=requested_k, raw_rows=rows_for_summary, results=(summary,))


def _cold_target_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, HnswSettings]:
    return (
        str(row["candidate_type"]),
        str(row["query_id"]),
        int(row["corpus_user_count"]),
        int(row["requested_k"]),
        _hnsw_from_dict(row["hnsw"]),
    )


def _cold_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, HnswSettings, int]:
    return (*_cold_target_key(row), int(row["iteration"]))


def _existing_kernel_diagnostic(*, args: argparse.Namespace, phase: str, scenario_id: str, cohort_size: int, requested_k: int, hnsw: HnswSettings) -> Mapping[str, Any]:
    path = args.output_root / "checkpoints" / "explain" / phase / f"n{cohort_size}" / f"{scenario_id}-k{requested_k}-ef{hnsw.ef_search}-{hnsw.iterative_scan}-scan{hnsw.max_scan_tuples}.json"
    if not path.is_file():
        raise RuntimeError(f"missing warm plan diagnostic for DB-cold cell: {path}")
    return {**read_json(path), "explain_path": str(path), "explain_sha256": sha256_file(path)}


def _restart_postgres_and_wait(container: str) -> None:
    subprocess.run(("docker", "restart", container), check=True, stdout=subprocess.DEVNULL)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        ready = subprocess.run(("docker", "exec", container, "pg_isready", "-q"), check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if ready.returncode == 0:
            return
        time.sleep(0.25)
    raise TimeoutError("PostgreSQL did not become ready within 60 seconds")


def run_part_b_preconfirmation(args: argparse.Namespace) -> None:
    """Measure the corrected, unchanged full-membership meaning.

    Candidate selection is a union: the old Goal 2 rows are hint-only and
    cannot supply a result, while Part A rows enter only when their own
    retrieval gate is close enough to be informative.
    """

    _plan(args)
    _lock_execution_code_fingerprint(args, phase="part-b-preconfirmation")
    manifest = BenchmarkManifest.load(args.goal1_root / "phase2/scenario-manifest.json")
    candidates = _build_part_b_candidates(args)
    write_immutable_json(
        args.output_root / "part-b-candidate-provenance.json",
        {
            "experiment_version": GOAL3_EXPERIMENT_VERSION,
            "candidate_count": len(candidates),
            "candidates": candidates,
        },
    )
    raw_path = args.output_root / "full-membership-raw.jsonl"
    result_path = args.output_root / "full-membership-results.jsonl"
    completed = {str(row["candidate_id"]) for row in read_jsonl(result_path) if row.get("phase") == "preconfirmation"}
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        if candidate_id in completed:
            continue
        cohort_size = int(candidate["corpus_user_count"])
        with _live_scope(args, cohort_size, manifest) as live:
            observations = _part_b_observations(
                live=live,
                scenario_id=str(candidate["tuning_scenario_id"]),
                requested_k=int(candidate["requested_k"]),
                hnsw=_hnsw_from_dict(candidate["hnsw"]),
                phase=BenchmarkPhase.TUNING,
                cache_mode=CacheMode.WARM,
                warmups=5,
                repetitions=30,
                diagnostics_dir=args.output_root / "checkpoints" / "part-b-explain" / candidate_id,
            )
        raw_rows = [{**item.to_dict(), "candidate_id": candidate_id, "part_b_phase": "preconfirmation"} for item in observations]
        append_jsonl_once(raw_path, raw_rows, identity_fields=("candidate_id", "plan", "measured", "duration_ms"))
        result = _part_b_result(candidate=candidate, observations=observations, phase="preconfirmation", final=False)
        append_jsonl_once(result_path, (result,), identity_fields=("candidate_id", "phase"))
        _write_checkpoint(args, phase="part-b-preconfirmation", cohort_size=cohort_size, candidate_type=str(candidate["candidate_type"]), query_id=str(candidate["tuning_scenario_id"]), requested_k=int(candidate["requested_k"]), raw_rows=raw_rows, results=(result,))
        _progress("part_b_preconfirmation_checkpoint", candidate_id=candidate_id, cohort_size=cohort_size, requested_k=candidate["requested_k"], passed=result["preconfirmation_passed"])


def run_part_b_confirmation(args: argparse.Namespace) -> None:
    _plan(args)
    _lock_execution_code_fingerprint(args, phase="part-b-confirmation")
    manifest = BenchmarkManifest.load(args.goal1_root / "phase2/scenario-manifest.json")
    provenance = read_json(args.output_root / "part-b-candidate-provenance.json")
    candidates = provenance.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("Part B candidate provenance is required")
    pre_results = {str(row["candidate_id"]): row for row in read_jsonl(args.output_root / "full-membership-results.jsonl") if row.get("phase") == "preconfirmation"}
    raw_path = args.output_root / "full-membership-raw.jsonl"
    confirmation_path = args.output_root / "full-membership-confirmation.jsonl"
    completed = {str(row["candidate_id"]) for row in read_jsonl(confirmation_path) if row.get("phase") == "confirmation_warm"}
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise ValueError("invalid Part B candidate")
        candidate_id = str(candidate["candidate_id"])
        pre = pre_results.get(candidate_id)
        if not pre or not bool(pre.get("preconfirmation_passed")):
            continue
        if candidate.get("confirmation_scenario_id") is None:
            continue
        if candidate_id in completed:
            continue
        cohort_size = int(candidate["corpus_user_count"])
        with _live_scope(args, cohort_size, manifest) as live:
            observations = _part_b_observations(
                live=live,
                scenario_id=str(candidate["confirmation_scenario_id"]),
                requested_k=int(candidate["requested_k"]),
                hnsw=_hnsw_from_dict(candidate["hnsw"]),
                phase=BenchmarkPhase.CONFIRMATION,
                cache_mode=CacheMode.WARM,
                warmups=10,
                repetitions=300,
                diagnostics_dir=args.output_root / "checkpoints" / "part-b-explain" / candidate_id,
            )
        raw_rows = [{**item.to_dict(), "candidate_id": candidate_id, "part_b_phase": "confirmation_warm"} for item in observations]
        append_jsonl_once(raw_path, raw_rows, identity_fields=("candidate_id", "plan", "measured", "duration_ms"))
        result = _part_b_result(candidate=candidate, observations=observations, phase="confirmation_warm", final=True)
        # DB-cold is intentionally not faked: finalization requires separately
        # registered cold rows and therefore keeps this warm-only record from
        # becoming a passed full-membership conclusion.
        result = {**result, "db_cold_confirmation_completed": False, "ann_full_membership_passed": False, "ann_policy_candidate": False}
        append_jsonl_once(confirmation_path, (result,), identity_fields=("candidate_id", "phase"))
        _write_checkpoint(args, phase="part-b-confirmation-warm", cohort_size=cohort_size, candidate_type=str(candidate["candidate_type"]), query_id=str(candidate["confirmation_scenario_id"]), requested_k=int(candidate["requested_k"]), raw_rows=raw_rows, results=(result,))
        _progress("part_b_confirmation_warm_checkpoint", candidate_id=candidate_id, cohort_size=cohort_size, requested_k=candidate["requested_k"], warm_gate=result.get("ann_full_membership_passed"))


def _build_part_b_candidates(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    splits = {item.candidate_type: item for item in scenario_splits(read_json(args.goal1_root / "phase2/scenario-manifest.json"))}
    candidates: dict[tuple[Any, ...], dict[str, Any]] = {}
    def include(*, cohort_size: int, candidate_type: str, requested_k: int, hnsw: Mapping[str, Any], source: str, reason: str, part_a: Mapping[str, Any] | None = None) -> None:
        split = splits[candidate_type]
        key = (cohort_size, split.tuning_scenario_id, requested_k, int(hnsw["ef_search"]), str(hnsw["iterative_scan"]), int(hnsw["max_scan_tuples"]))
        item = candidates.setdefault(key, {"experiment_version": GOAL3_EXPERIMENT_VERSION, "candidate_id": canonical_json_sha256(list(key)), "corpus_user_count": cohort_size, "scenario_id": split.tuning_scenario_id, "tuning_scenario_id": split.tuning_scenario_id, "confirmation_scenario_id": split.confirmation_scenario_id if split.independent_confirmation_query else None, "candidate_type": candidate_type, "requested_k": requested_k, "hnsw": dict(hnsw), "provenance": [], "selection_reasons": [], "part_a_links": []})
        item["provenance"].append(source)
        item["selection_reasons"].append(reason)
        if part_a is not None:
            item["part_a_links"].append({"query_id": part_a["query_id"], "phase": part_a["phase"], "k_over_n": part_a["k_over_n"], "ann_p95_ratio": part_a["ann_p95_ratio"]})
    # Mandatory diagnostic, required even if Goal 2 had no positive cell.
    include(cohort_size=100_000, candidate_type="funnel_recovery", requested_k=1_000, hnsw={"ef_search": 50, "iterative_scan": "relaxed_order", "max_scan_tuples": 20_000}, source="mandatory_diagnostic", reason="specified corrected Goal 2 diagnostic")
    # Goal 2 rows are candidate selection hints only.  Preserve the closest two
    # per actual N/candidate-type rather than treating their invalid precision
    # or latency as evidence.
    goal2_rows = read_jsonl(args.goal1_root / "goal2/hnsw-tuning-results.jsonl")
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in goal2_rows:
        grouped[(int(row["corpus_user_count"]), str(row["candidate_type"]))].append(row)
    for (cohort_size, candidate_type), rows in grouped.items():
        for row in sorted(rows, key=lambda item: abs(float(item.get("exact_all_p95_ratio", 9.0)) - 0.8))[:2]:
            hnsw = row.get("hnsw")
            if isinstance(hnsw, Mapping):
                include(cohort_size=cohort_size, candidate_type=candidate_type, requested_k=int(row["requested_k"]), hnsw=hnsw, source="goal2_near_cell", reason="invalidated Goal 2 proximity hint")
    part_a_rows = [
        row
        for row in read_jsonl(args.output_root / "kernel-results.jsonl")
        if row.get("phase") == RSS_CORRECTED_SCREENING_PHASE
        and (bool(row.get("screening_gate_passed")) or bool(row.get("boundary_near")))
    ]
    for row in part_a_rows:
        hnsw = row.get("hnsw")
        if isinstance(hnsw, Mapping):
            include(cohort_size=int(row["corpus_user_count"]), candidate_type=str(row["candidate_type"]), requested_k=int(row["requested_k"]), hnsw=hnsw, source=("part_a_passed" if bool(row.get("screening_gate_passed")) else "part_a_boundary_near"), reason="Part A retrieval candidate", part_a=row)
    return sorted(candidates.values(), key=lambda item: (item["corpus_user_count"], item["candidate_type"], item["requested_k"], item["candidate_id"]))


def _part_b_observations(*, live: _LiveScope, scenario_id: str, requested_k: int, hnsw: HnswSettings, phase: BenchmarkPhase, cache_mode: CacheMode, warmups: int, repetitions: int, diagnostics_dir: Path) -> Sequence[Any]:
    assert live.benchmark is not None
    scenario = live.manifest.require_scenario(scenario_id)
    return live.benchmark.run_scenario(manifest=live.manifest, scenario=scenario, config=CellRunConfig(phase=phase, cache_mode=cache_mode, warmups=warmups, repetitions=repetitions, sample_sizes=(min(50_000, live.cohort_size),), hnsw_settings=(hnsw,), requested_k_values=(requested_k,), plans=(SearchPlan.CURRENT_RUNTIME, SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT, SearchPlan.ANN_FIRST)), diagnostics_dir=diagnostics_dir)


def _part_b_result(*, candidate: Mapping[str, Any], observations: Sequence[Any], phase: str, final: bool) -> Mapping[str, Any]:
    summaries = summarize_observations(observations, phase=(BenchmarkPhase.CONFIRMATION if final else BenchmarkPhase.TUNING))
    by_plan = {item.plan.value: item.to_dict() for item in summaries}
    required = {"current_runtime", "exact_all", "filter_first_exact", "ann_first"}
    if set(by_plan) != required:
        raise RuntimeError("Part B matched baseline coverage is incomplete")
    gates = part_b_gate(ann=by_plan["ann_first"], exact_all=by_plan["exact_all"], filter_first=by_plan["filter_first_exact"], final=final, current_runtime=by_plan["current_runtime"])
    return {"experiment_version": GOAL3_EXPERIMENT_VERSION, "candidate_id": candidate["candidate_id"], "phase": phase, "candidate_type": candidate["candidate_type"], "corpus_user_count": candidate["corpus_user_count"], "scenario_id": (candidate["confirmation_scenario_id"] if final else candidate["tuning_scenario_id"]), "requested_k": candidate["requested_k"], "hnsw": candidate["hnsw"], "current_runtime": by_plan["current_runtime"], "exact_all": by_plan["exact_all"], "filter_first_exact": by_plan["filter_first_exact"], "ann_corrected": by_plan["ann_first"], **gates}


def _hnsw_from_dict(payload: object) -> HnswSettings:
    if not isinstance(payload, Mapping):
        raise ValueError("HNSW payload is invalid")
    return HnswSettings(int(payload["ef_search"]), str(payload["iterative_scan"]), int(payload["max_scan_tuples"]))


def finalize(args: argparse.Namespace) -> None:
    plan = _plan(args)
    _lock_execution_code_fingerprint(args, phase="finalization")
    splits = scenario_splits(read_json(args.goal1_root / "phase2/scenario-manifest.json"))
    warm_confirmations = read_jsonl(args.output_root / "kernel-confirmation.jsonl")
    cold_confirmations = read_jsonl(args.output_root / "kernel-cold-confirmation.jsonl")
    if not warm_confirmations:
        raise RuntimeError("warm kernel confirmation is required before finalization")
    combined_confirmations = combine_kernel_confirmations(
        warm_confirmations, cold_confirmations
    )
    final_kernel_path = args.output_root / "candidate-retrieval-final-confirmation.jsonl"
    append_jsonl_once(
        final_kernel_path,
        combined_confirmations,
        identity_fields=(
            "candidate_type",
            "query_id",
            "corpus_user_count",
            "requested_k",
            "hnsw",
        ),
    )
    crossover = crossover_report(combined_confirmations, splits=splits)
    write_immutable_json(args.output_root / "candidate-retrieval-crossover.json", crossover)

    provenance = read_json(args.output_root / "part-b-candidate-provenance.json")
    candidates = provenance.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("Part B candidate provenance is invalid")
    pre_results = [
        row
        for row in read_jsonl(args.output_root / "full-membership-results.jsonl")
        if row.get("phase") == "preconfirmation"
    ]
    _require_part_b_preconfirmation_coverage(candidates, pre_results)
    confirmation_path = args.output_root / "full-membership-confirmation.jsonl"
    # An empty JSONL file is an intentional, auditable result when every
    # candidate fails the pre-confirmation gate; it is not a missing artifact.
    append_jsonl_once(confirmation_path, (), identity_fields=("candidate_id", "phase"))
    full_confirmations = read_jsonl(confirmation_path)
    _require_part_b_confirmation_coverage(candidates, pre_results, full_confirmations)

    write_immutable_json(
        args.output_root / "part-b-candidate-evidence.json",
        _part_b_candidate_evidence(candidates, pre_results),
    )
    write_immutable_json(
        args.output_root / "part-b-provenance-validation.json",
        _part_b_provenance_validation(args, candidates),
    )
    write_immutable_json(
        args.output_root / "stage-cost-breakdown.json",
        _stage_cost_breakdown(candidates, pre_results),
    )
    write_immutable_json(
        args.output_root / "unvalidated-fallbacks.json",
        _unvalidated_fallbacks(
            kernel_screen_rows=[
                row
                for row in read_jsonl(args.output_root / "kernel-results.jsonl")
                if row.get("phase") == RSS_CORRECTED_SCREENING_PHASE
            ],
            kernel_confirmations=combined_confirmations,
            candidates=candidates,
            pre_results=pre_results,
            full_confirmations=full_confirmations,
        ),
    )
    write_immutable_json(
        args.output_root / "raw-result-reconciliation.json",
        _raw_result_reconciliation(
            kernel_screen_rows=[
                row
                for row in read_jsonl(args.output_root / "kernel-results.jsonl")
                if row.get("phase") == RSS_CORRECTED_SCREENING_PHASE
            ],
            candidates=candidates,
            pre_results=pre_results,
            full_raw_rows=read_jsonl(args.output_root / "full-membership-raw.jsonl"),
        ),
    )

    candidate_passed = any(
        bool(row["ann_candidate_retrieval_passed"])
        for row in combined_confirmations
    )
    passed_candidate_types = sorted(
        {
            str(row["candidate_type"])
            for row in combined_confirmations
            if bool(row["ann_candidate_retrieval_passed"])
        }
    )
    full_passed = any(
        bool(row.get("ann_full_membership_passed"))
        and bool(row.get("db_cold_confirmation_completed"))
        for row in full_confirmations
    )
    policy_candidate = any(
        bool(row.get("ann_policy_candidate"))
        and bool(row.get("db_cold_confirmation_completed"))
        for row in full_confirmations
    )
    fallback_candidate_types = sorted(
        {item.candidate_type for item in splits} - set(passed_candidate_types)
    )
    preconfirmed = [
        row for row in pre_results if bool(row.get("preconfirmation_passed"))
    ]
    conclusion = {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "ann_candidate_retrieval_passed": candidate_passed,
        "ann_candidate_retrieval_min_users_by_k": crossover["ann_candidate_retrieval_min_users_by_k"],
        "ann_candidate_retrieval_valid_region": crossover["ann_candidate_retrieval_valid_region"],
        "query_specific_kernel_winner": candidate_passed,
        "deployable_common_setting": False,
        "ann_full_membership_passed": full_passed,
        "ann_full_membership_min_users": None,
        "ann_policy_candidate": policy_candidate,
        "candidate_types_passed": passed_candidate_types,
        "candidate_types_fallback": fallback_candidate_types,
        "measurement_scope": {
            "plan_sha256": plan["experiment_plan_sha256"],
            "kernel_warm_confirmation_rows": len(warm_confirmations),
            "kernel_cold_confirmation_rows": len(cold_confirmations),
            "kernel_final_confirmed_rows": len(combined_confirmations),
            "kernel_db_cold_completed_rows": sum(
                bool(row["db_cold_confirmation_completed"])
                for row in combined_confirmations
            ),
            "part_b_candidate_count": len(candidates),
            "part_b_preconfirmation_rows": len(pre_results),
            "part_b_preconfirmation_passed_rows": len(preconfirmed),
            "part_b_final_confirmation_rows": len(full_confirmations),
            "part_b_completed": True,
        },
        "claim_text": _claim_text(
            candidate_passed=candidate_passed,
            full_passed=full_passed,
            preconfirmed_count=len(preconfirmed),
        ),
        "claim_limitations": [
            "Warm kernel confirmation alone cannot pass the final gate; DB-cold evidence is joined per HNSW cell.",
            "Candidate retrieval does not imply a full-membership or product-policy winner.",
            "general_destination_explorer has no distinct actual confirmation query vector.",
            "No query-independent runtime schedule is promoted from candidate-type-specific evidence.",
        ],
    }
    write_immutable_json(args.output_root / "conclusion.json", conclusion)
    _write_immutable_text(args.output_root / "goal3-summary.md", render_summary(conclusion))
    write_goal3_figures(args.output_root)
    graph_validation = validate_graph_sources(args.output_root)
    integrity = {
        **build_integrity_report(args.output_root, finalization_required=True),
        "graph_source_validation": graph_validation,
    }
    integrity["passed"] = bool(integrity["passed"] and graph_validation["passed"])
    write_immutable_json(args.output_root / "integrity-report.json", integrity)
    _progress(
        "goal3_finalized",
        kernel_rows=len(combined_confirmations),
        part_b_rows=len(pre_results),
        integrity_passed=integrity["passed"],
    )


def _require_part_b_preconfirmation_coverage(
    candidates: Sequence[Mapping[str, Any]],
    pre_results: Sequence[Mapping[str, Any]],
) -> None:
    expected = {str(item["candidate_id"]) for item in candidates}
    actual = {str(item["candidate_id"]) for item in pre_results}
    if actual != expected:
        raise RuntimeError(
            "Part B pre-confirmation is incomplete: "
            f"missing={sorted(expected - actual)}, unexpected={sorted(actual - expected)}"
        )


def _require_part_b_confirmation_coverage(
    candidates: Sequence[Mapping[str, Any]],
    pre_results: Sequence[Mapping[str, Any]],
    final_results: Sequence[Mapping[str, Any]],
) -> None:
    pre_by_id = {str(item["candidate_id"]): item for item in pre_results}
    required = {
        str(candidate["candidate_id"])
        for candidate in candidates
        if bool(pre_by_id[str(candidate["candidate_id"])].get("preconfirmation_passed"))
        and candidate.get("confirmation_scenario_id") is not None
    }
    actual = {
        str(item["candidate_id"])
        for item in final_results
        if item.get("phase") == "confirmation_final"
    }
    if actual != required:
        raise RuntimeError(
            "Part B final confirmation is incomplete: "
            f"missing={sorted(required - actual)}, unexpected={sorted(actual - required)}"
        )


def _part_b_candidate_evidence(
    candidates: Sequence[Mapping[str, Any]],
    pre_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    pre_by_id = {str(row["candidate_id"]): row for row in pre_results}
    rows = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        result = pre_by_id[candidate_id]
        exact = result["exact_all"]
        ann = result["ann_corrected"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate["candidate_type"],
                "part_b_scenario": candidate["tuning_scenario_id"],
                "corpus_user_count": candidate["corpus_user_count"],
                "requested_k": candidate["requested_k"],
                "hnsw": candidate["hnsw"],
                "provenance": candidate["provenance"],
                "selection_reasons": candidate["selection_reasons"],
                "part_a_links": candidate["part_a_links"],
                "expected_exact_positive_count": exact.get("exact_positive_count"),
                "H_over_N": exact.get("hard_match_ratio"),
                "E_over_N": exact.get("expected_member_ratio"),
                "ann_final_member_count": ann.get("final_user_count"),
                "preconfirmation_passed": result.get("preconfirmation_passed"),
            }
        )
    return {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "rows": rows,
        "note": "Candidate provenance is immutable; observed full-membership selectivity is recorded in this companion evidence file.",
    }


def _part_b_provenance_validation(
    args: argparse.Namespace,
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    allowed_sources = {
        "goal2_near_cell",
        "part_a_passed",
        "part_a_boundary_near",
        "mandatory_diagnostic",
    }
    selected: set[tuple[Any, ...]] = set()
    invalid_sources: list[str] = []
    for candidate in candidates:
        hnsw = candidate["hnsw"]
        selected.add(
            (
                int(candidate["corpus_user_count"]),
                str(candidate["candidate_type"]),
                int(candidate["requested_k"]),
                int(hnsw["ef_search"]),
                str(hnsw["iterative_scan"]),
                int(hnsw["max_scan_tuples"]),
            )
        )
        invalid_sources.extend(
            str(source)
            for source in candidate.get("provenance", [])
            if source not in allowed_sources
        )
    expected_part_a = set()
    for row in read_jsonl(args.output_root / "kernel-results.jsonl"):
        if row.get("phase") != RSS_CORRECTED_SCREENING_PHASE:
            continue
        if not (bool(row.get("screening_gate_passed")) or bool(row.get("boundary_near"))):
            continue
        hnsw = row["hnsw"]
        expected_part_a.add(
            (
                int(row["corpus_user_count"]),
                str(row["candidate_type"]),
                int(row["requested_k"]),
                int(hnsw["ef_search"]),
                str(hnsw["iterative_scan"]),
                int(hnsw["max_scan_tuples"]),
            )
        )
    mandatory = (100_000, "funnel_recovery", 1_000, 50, "relaxed_order", 20_000)
    missing_part_a = sorted(expected_part_a - selected)
    return {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "passed": not missing_part_a and not invalid_sources and mandatory in selected,
        "selected_candidate_count": len(candidates),
        "expected_part_a_candidate_count": len(expected_part_a),
        "missing_part_a_candidates": missing_part_a,
        "unexpected_provenance_sources": sorted(set(invalid_sources)),
        "mandatory_diagnostic_present": mandatory in selected,
        "source_union": sorted(
            {source for candidate in candidates for source in candidate.get("provenance", [])}
        ),
    }


def _part_b_failure_reasons(result: Mapping[str, Any]) -> list[str]:
    ann = result["ann_corrected"]
    reasons = []
    if float(result["precision"]) != 1.0:
        reasons.append("precision_not_one")
    if not bool(result["final_set_subset_invariant_passed"]):
        reasons.append("final_set_subset_invariant_failed")
    if float(result["worst_run_recall"]) < 0.95:
        reasons.append("worst_run_recall_below_0_95")
    if float(result["wilson_lower_bound"]) < 0.95:
        reasons.append("wilson_lower_bound_below_0_95")
    if float(result["exact_all_p95_ratio"]) > 0.80:
        reasons.append("ann_p95_not_within_preconfirmation_ratio")
    if not bool(ann["index_used"]):
        reasons.append("hnsw_index_not_used")
    if bool(ann["temp_spill"]):
        reasons.append("temp_spill")
    if bool(ann["oom"]):
        reasons.append("oom")
    return reasons or ["final_confirmation_required"]


def _stage_cost_breakdown(
    candidates: Sequence[Mapping[str, Any]],
    pre_results: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    pre_by_id = {str(row["candidate_id"]): row for row in pre_results}
    rows = []
    for candidate in candidates:
        links = candidate.get("part_a_links", [])
        if not links:
            continue
        result = pre_by_id[str(candidate["candidate_id"])]
        current = result["current_runtime"]
        stages = current.get("stage_p95_ms", {})
        if not isinstance(stages, Mapping):
            stages = {}
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "candidate_type": candidate["candidate_type"],
                "corpus_user_count": candidate["corpus_user_count"],
                "requested_k": candidate["requested_k"],
                "hnsw": candidate["hnsw"],
                "part_a_links": links,
                "part_b_preconfirmation_passed": result["preconfirmation_passed"],
                "part_b_failure_reasons": _part_b_failure_reasons(result),
                "stage_ms": dict(stages),
                "current_runtime_total_p95_ms": current.get("p95_ms"),
                "ann_corrected_total_p95_ms": result["ann_corrected"].get("p95_ms"),
                "exact_all_total_p95_ms": result["exact_all"].get("p95_ms"),
                "filter_first_exact_total_p95_ms": result["filter_first_exact"].get("p95_ms"),
                "unobserved_stages": [
                    "query_preparation",
                    "result_materialization",
                    "audit",
                ],
                "note": "Only directly instrumented current-runtime stage p95 values are decomposed; unobserved stages are not estimated.",
            }
        )
    return {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "rows": rows,
        "stage_measurement_source": "current_runtime.stage_p95_ms from the matched Part B observation",
    }


def _unvalidated_fallbacks(
    *,
    kernel_screen_rows: Sequence[Mapping[str, Any]],
    kernel_confirmations: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    pre_results: Sequence[Mapping[str, Any]],
    full_confirmations: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    confirmed_keys = {
        _fallback_setting_key(row) for row in kernel_confirmations
    }
    kernel_rows = []
    for row in kernel_screen_rows:
        key = _fallback_setting_key(row)
        if key in confirmed_keys:
            continue
        kernel_rows.append(
            {
                "candidate_type": row["candidate_type"],
                "query_id": row["query_id"],
                "corpus_user_count": row["corpus_user_count"],
                "requested_k": row["requested_k"],
                "hnsw": row["hnsw"],
                "fallback": "exact_top_k",
                "reason": "not_selected_for_final_confirmation",
            }
        )
    for row in kernel_confirmations:
        if bool(row["ann_candidate_retrieval_passed"]):
            continue
        kernel_rows.append(
            {
                "candidate_type": row["candidate_type"],
                "query_id": row["query_id"],
                "corpus_user_count": row["corpus_user_count"],
                "requested_k": row["requested_k"],
                "hnsw": row["hnsw"],
                "fallback": "exact_top_k",
                "reason": row["cold_confirmation_status"],
            }
        )
    pre_by_id = {str(row["candidate_id"]): row for row in pre_results}
    final_by_id = {str(row["candidate_id"]): row for row in full_confirmations}
    full_rows = []
    for candidate in candidates:
        candidate_id = str(candidate["candidate_id"])
        pre = pre_by_id[candidate_id]
        final = final_by_id.get(candidate_id)
        if final is not None and bool(final.get("ann_full_membership_passed")):
            continue
        full_rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": candidate["candidate_type"],
                "corpus_user_count": candidate["corpus_user_count"],
                "requested_k": candidate["requested_k"],
                "hnsw": candidate["hnsw"],
                "fallback": "exact_all_or_current_runtime_exact_fallback",
                "reason": (
                    _part_b_failure_reasons(pre)
                    if not bool(pre.get("preconfirmation_passed"))
                    else ["final_confirmation_not_passing"]
                ),
            }
        )
    return {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "candidate_retrieval_exact_fallbacks": kernel_rows,
        "full_membership_exact_fallbacks": full_rows,
        "global_policy": "Exact fallback remains required where a final gate is absent or fails.",
    }


def _raw_result_reconciliation(
    *,
    kernel_screen_rows: Sequence[Mapping[str, Any]],
    candidates: Sequence[Mapping[str, Any]],
    pre_results: Sequence[Mapping[str, Any]],
    full_raw_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in full_raw_rows:
        if row.get("part_b_phase") == "preconfirmation":
            by_candidate[str(row["candidate_id"])].append(row)
    expected_ids = {str(candidate["candidate_id"]) for candidate in candidates}
    result_ids = {str(row["candidate_id"]) for row in pre_results}
    violations = []
    for candidate_id in sorted(expected_ids):
        rows = by_candidate.get(candidate_id, [])
        by_plan: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            by_plan[str(row["plan"])].append(row)
        if set(by_plan) != {
            SearchPlan.CURRENT_RUNTIME.value,
            SearchPlan.EXACT_ALL.value,
            SearchPlan.FILTER_FIRST_EXACT.value,
            SearchPlan.ANN_FIRST.value,
        }:
            violations.append(f"{candidate_id}:matched_plan_coverage")
            continue
        for plan, plan_rows in by_plan.items():
            measured = sum(bool(row.get("measured")) for row in plan_rows)
            warmups = len(plan_rows) - measured
            if measured != 30 or warmups != 5:
                violations.append(
                    f"{candidate_id}:{plan}:warmups={warmups}:measured={measured}"
                )
    return {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "passed": not violations and result_ids == expected_ids,
        "kernel_screen_summary_rows": len(kernel_screen_rows),
        "part_b_expected_candidates": len(expected_ids),
        "part_b_preconfirmation_results": len(result_ids),
        "part_b_preconfirmation_raw_rows": sum(len(rows) for rows in by_candidate.values()),
        "violations": violations,
        "missing_result_candidates": sorted(expected_ids - result_ids),
        "unexpected_result_candidates": sorted(result_ids - expected_ids),
    }


def repair_fallback_audit(args: argparse.Namespace) -> None:
    """Repair the one historical Goal 3 finalization that emitted JSON null.

    This command is deliberately narrower than finalization: it will only
    replace an exact ``null`` placeholder whose hash is already recorded by
    the existing integrity report.  Goal 1, Goal 2, raw observations, and all
    performance conclusions remain untouched.
    """

    fallback_path = args.output_root / "unvalidated-fallbacks.json"
    integrity_path = args.output_root / "integrity-report.json"
    if fallback_path.read_text(encoding="utf-8") != "null\n":
        raise RuntimeError("fallback audit repair requires the exact historical null placeholder")
    old_integrity = read_json(integrity_path)
    recorded_hashes = old_integrity.get("artifact_sha256")
    if not isinstance(recorded_hashes, Mapping):
        raise RuntimeError("existing Goal 3 integrity hashes are missing")
    actual_null_hash = sha256_file(fallback_path)
    if recorded_hashes.get("unvalidated-fallbacks.json") != actual_null_hash:
        raise RuntimeError("fallback placeholder does not match the recorded integrity hash")

    _lock_execution_code_fingerprint(args, phase="fallback-audit-repair")
    kernel_screen_rows = [
        row
        for row in read_jsonl(args.output_root / "kernel-results.jsonl")
        if row.get("phase") == RSS_CORRECTED_SCREENING_PHASE
    ]
    payload = dict(
        _unvalidated_fallbacks(
            kernel_screen_rows=kernel_screen_rows,
            kernel_confirmations=read_jsonl(
                args.output_root / "candidate-retrieval-final-confirmation.jsonl"
            ),
            candidates=read_json(
                args.output_root / "part-b-candidate-provenance.json"
            )["candidates"],
            pre_results=read_jsonl(args.output_root / "full-membership-results.jsonl"),
            full_confirmations=read_jsonl(
                args.output_root / "full-membership-confirmation.jsonl"
            ),
        )
    )
    payload["repair_provenance"] = {
        "reason": "finalizer control-flow indentation defect emitted JSON null",
        "replaced_artifact_sha256": actual_null_hash,
        "measurement_results_changed": False,
    }
    fallback_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    graph_validation = validate_graph_sources(args.output_root)
    integrity = {
        **build_integrity_report(args.output_root, finalization_required=True),
        "graph_source_validation": graph_validation,
    }
    integrity["passed"] = bool(integrity["passed"] and graph_validation["passed"])
    integrity["repair_provenance"] = payload["repair_provenance"]
    integrity_path.write_text(
        json.dumps(integrity, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _progress(
        "goal3_fallback_audit_repaired",
        candidate_retrieval_fallbacks=len(payload["candidate_retrieval_exact_fallbacks"]),
        full_membership_fallbacks=len(payload["full_membership_exact_fallbacks"]),
        integrity_passed=integrity["passed"],
    )


def _claim_text(*, candidate_passed: bool, full_passed: bool, preconfirmed_count: int) -> str:
    if full_passed:
        return "A corrected full-membership ANN result passed its independent warm and DB-cold confirmations."
    if candidate_passed:
        return (
            "HNSW passed the candidate-retrieval kernel gate for query-specific cells, "
            "but no corrected full-membership policy winner was established. "
            f"Part B pre-confirmation winners: {preconfirmed_count}."
        )
    return "No verified ANN efficiency region or corrected full-membership policy winner was established."


def _write_immutable_text(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise FileExistsError(f"refusing to overwrite immutable artifact: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class _LiveScope:
    def __init__(self, args: argparse.Namespace, cohort_size: int, manifest: BenchmarkManifest) -> None:
        self.args = args
        self.cohort_size = cohort_size
        self.manifest = manifest
        self.connection: Any | None = None
        self.clickhouse: Any | None = None
        self.pg: PsycopgPostgresExecutor | None = None
        self.benchmark: LiveAnnSearchBenchmark | None = None
        self.context: Any | None = None
        self._previous_rss_container: str | None = None

    def __enter__(self) -> "_LiveScope":
        environment = _environment(self.args, self.cohort_size)
        # _BackendRssSampler reads the container name from the benchmark
        # process environment.  ``load_settings(environment)`` deliberately
        # does not mutate that environment, so bind the disposable PostgreSQL
        # container for the lifetime of this live scope.  This affects only
        # offline measurement instrumentation, never PostgreSQL settings.
        self._previous_rss_container = os.environ.get("ANN_POSTGRES_CONTAINER")
        os.environ["ANN_POSTGRES_CONTAINER"] = str(
            environment["ANN_POSTGRES_CONTAINER"]
        )
        settings = load_settings(environment)
        self.connection = create_postgres_connection(settings)
        self.clickhouse = create_clickhouse_client(settings)
        self.pg = PsycopgPostgresExecutor(self.connection)
        self.benchmark = LiveAnnSearchBenchmark(
            postgres_connection=self.connection,
            clickhouse=self.clickhouse,
            scale_cohort_scope=ScaleCohortScope(
                scale_series_id=self.args.scale_series_id,
                cohort_size=self.cohort_size,
                reference_sample_seed=self.args.reference_sample_seed,
            ),
        )
        preflight = self.benchmark.preflight(self.manifest)
        if int(preflight["corpus_user_count"]) != self.cohort_size:
            raise RuntimeError("live cohort does not match frozen N")
        self.context = self.benchmark._load_context(self.manifest)
        return self

    def __exit__(self, *_exc: object) -> None:
        if self.connection is not None:
            self.connection.close()
        close = getattr(self.clickhouse, "close", None)
        if callable(close):
            close()
        if self._previous_rss_container is None:
            os.environ.pop("ANN_POSTGRES_CONTAINER", None)
        else:
            os.environ["ANN_POSTGRES_CONTAINER"] = self._previous_rss_container


def _live_scope(args: argparse.Namespace, cohort_size: int, manifest: BenchmarkManifest) -> _LiveScope:
    return _LiveScope(args, cohort_size, manifest)


def _run_kernel_scope(
    *,
    live: _LiveScope,
    scenario: Any,
    candidate_type: str,
    requested_k: int,
    settings: Sequence[HnswSettings],
    phase: str,
    warmups: int,
    measured: int,
    k_provenance: str,
    interleaved: bool,
    diagnostics: bool,
    diagnostics_root: Path | None = None,
) -> list[Mapping[str, Any]]:
    assert live.pg is not None and live.connection is not None and live.context is not None
    if requested_k > live.cohort_size:
        raise ValueError("K exceeds the frozen corpus")
    diagnostics_by_setting: dict[HnswSettings, Mapping[str, Any]] = {}
    if diagnostics:
        for setting in settings:
            diagnostics_by_setting[setting] = _kernel_diagnostic(
                live=live,
                scenario=scenario,
                requested_k=requested_k,
                hnsw=setting,
                output_root=diagnostics_root,
                phase=phase,
            )
    rng = random.Random(_seed_for(phase, candidate_type, scenario.scenario_id, live.cohort_size, requested_k))
    exact_samples: list[Mapping[str, Any]] = []
    rows: list[Mapping[str, Any]] = []
    if not interleaved:
        for iteration in range(warmups + measured):
            exact_samples.append(_topk_attempt(live=live, scenario=scenario, requested_k=requested_k, hnsw=None))
        for setting in settings:
            for iteration in range(warmups + measured):
                ann = _topk_attempt(live=live, scenario=scenario, requested_k=requested_k, hnsw=setting)
                exact = exact_samples[iteration]
                rows.append(_kernel_row(phase=phase, measured=iteration >= warmups, iteration=iteration, candidate_type=candidate_type, scenario=scenario, cohort_size=live.cohort_size, requested_k=requested_k, k_provenance=k_provenance, hnsw=setting, exact=exact, ann=ann, diagnostic=diagnostics_by_setting.get(setting)))
        return rows
    for setting in settings:
        for iteration in range(warmups + measured):
            if rng.choice((True, False)):
                ann = _topk_attempt(live=live, scenario=scenario, requested_k=requested_k, hnsw=setting)
                exact = _topk_attempt(live=live, scenario=scenario, requested_k=requested_k, hnsw=None)
            else:
                exact = _topk_attempt(live=live, scenario=scenario, requested_k=requested_k, hnsw=None)
                ann = _topk_attempt(live=live, scenario=scenario, requested_k=requested_k, hnsw=setting)
            rows.append(_kernel_row(phase=phase, measured=iteration >= warmups, iteration=iteration, candidate_type=candidate_type, scenario=scenario, cohort_size=live.cohort_size, requested_k=requested_k, k_provenance=k_provenance, hnsw=setting, exact=exact, ann=ann, diagnostic=diagnostics_by_setting.get(setting)))
    return rows


def _topk_attempt(*, live: _LiveScope, scenario: Any, requested_k: int, hnsw: HnswSettings | None) -> Mapping[str, Any]:
    assert live.pg is not None and live.connection is not None and live.context is not None
    with live.connection.transaction():
        if hnsw is None:
            live.pg.execute("SET LOCAL enable_indexscan = off")
            live.pg.execute("SET LOCAL enable_bitmapscan = off")
            live.pg.execute("SET LOCAL enable_indexonlyscan = off")
        else:
            _set_hnsw(live.pg, hnsw)
        with _BackendRssSampler.from_connection(live.connection) as sampler:
            started = time.perf_counter_ns()
            result = live.pg.fetchall(
                _ann_sql(explain=False, exclude_promotion_users=False),
                _ann_params(manifest=live.manifest, scenario=scenario, context=live.context, requested_k=requested_k, exclusion_promotion_id=None),
            )
            duration_ms = (time.perf_counter_ns() - started) / 1_000_000
    return {"duration_ms": duration_ms, "user_ids": [str(row["user_id"]) for row in result], "peak_rss_bytes": sampler.peak_rss_bytes}


def _kernel_diagnostic(*, live: _LiveScope, scenario: Any, requested_k: int, hnsw: HnswSettings, output_root: Path | None, phase: str) -> Mapping[str, Any]:
    assert live.pg is not None and live.connection is not None and live.context is not None
    explain_rows: dict[str, Any] = {}
    for name, setting in (("exact", None), ("hnsw", hnsw)):
        with live.connection.transaction():
            if setting is None:
                live.pg.execute("SET LOCAL enable_indexscan = off")
                live.pg.execute("SET LOCAL enable_bitmapscan = off")
                live.pg.execute("SET LOCAL enable_indexonlyscan = off")
            else:
                _set_hnsw(live.pg, setting)
            row = live.pg.fetchone(_ann_sql(explain=True, exclude_promotion_users=False), _ann_params(manifest=live.manifest, scenario=scenario, context=live.context, requested_k=requested_k, exclusion_promotion_id=None))
        explain = next(iter(row.values())) if row else None
        explain_rows[name] = explain
    ann_indices = _find_plan_values(explain_rows["hnsw"], "Index Name")
    exact_indices = _find_plan_values(explain_rows["exact"], "Index Name")
    spills = sum(int(value or 0) for explain in explain_rows.values() for key in ("Temp Read Blocks", "Temp Written Blocks") for value in _find_plan_values(explain, key))
    payload = {"experiment_version": GOAL3_EXPERIMENT_VERSION, "phase": phase, "scenario_id": scenario.scenario_id, "corpus_user_count": live.cohort_size, "requested_k": requested_k, "hnsw": hnsw.to_dict(), "exact_explain": explain_rows["exact"], "hnsw_explain": explain_rows["hnsw"], "hnsw_index_used": HNSW_INDEX_NAME in ann_indices, "exact_uses_hnsw": HNSW_INDEX_NAME in exact_indices, "temp_spill": spills > 0}
    if output_root is not None:
        path = output_root / phase / f"n{live.cohort_size}" / f"{scenario.scenario_id}-k{requested_k}-ef{hnsw.ef_search}-{hnsw.iterative_scan}-scan{hnsw.max_scan_tuples}.json"
        # A previously interrupted confirmation can have written this
        # immutable plan diagnostic before its timed observations completed.
        # Reuse that first actual plan evidence rather than trying to replace
        # it with a second EXPLAIN of the identical N/query/K/HNSW cell.
        if path.is_file():
            payload = dict(read_json(path))
        else:
            write_immutable_json(path, payload)
        payload = {**payload, "explain_path": str(path), "explain_sha256": sha256_file(path)}
    return payload


def _kernel_row(*, phase: str, measured: bool, iteration: int, candidate_type: str, scenario: Any, cohort_size: int, requested_k: int, k_provenance: str, hnsw: HnswSettings, exact: Mapping[str, Any], ann: Mapping[str, Any], diagnostic: Mapping[str, Any] | None) -> Mapping[str, Any]:
    exact_ids = set(exact["user_ids"])
    ann_ids = set(ann["user_ids"])
    h = hashlib_sha256_ids(exact_ids)
    identity = {"phase": phase, "measured": measured, "iteration": iteration, "candidate_type": candidate_type, "query_id": scenario.scenario_id, "corpus_user_count": cohort_size, "requested_k": requested_k, "hnsw": hnsw.to_dict()}
    return {"experiment_version": GOAL3_EXPERIMENT_VERSION, **identity, "measurement_id": canonical_json_sha256(identity), "scenario_id": scenario.scenario_id, "candidate_type_label": "candidate_retrieval_kernel", "k_over_n": requested_k / cohort_size, "k_provenance": k_provenance, "exact_duration_ms": exact["duration_ms"], "ann_duration_ms": ann["duration_ms"], "exact_top_k_sha256": h, "ann_top_k_sha256": hashlib_sha256_ids(ann_ids), "intersection_count": len(exact_ids & ann_ids), "exact_count": len(exact_ids), "ann_count": len(ann_ids), "exact_peak_rss_bytes": exact.get("peak_rss_bytes"), "ann_peak_rss_bytes": ann.get("peak_rss_bytes"), "hnsw_index_used": bool(diagnostic and diagnostic.get("hnsw_index_used")), "exact_uses_hnsw": bool(diagnostic and diagnostic.get("exact_uses_hnsw")), "temp_spill": bool(diagnostic and diagnostic.get("temp_spill")), "oom": False, "measurement_fingerprint": "recorded in experiment-fingerprint.json"}


def _set_hnsw(pg: PsycopgPostgresExecutor, hnsw: HnswSettings) -> None:
    for name, value in (("hnsw.ef_search", str(hnsw.ef_search)), ("hnsw.iterative_scan", hnsw.iterative_scan), ("hnsw.max_scan_tuples", str(hnsw.max_scan_tuples))):
        pg.execute("SELECT set_config(%s, %s, true)", (name, value))


def hashlib_sha256_ids(user_ids: Iterable[str]) -> str:
    import hashlib
    digest = hashlib.sha256()
    for user_id in sorted(user_ids):
        digest.update(user_id.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_checkpoint(args: argparse.Namespace, *, phase: str, cohort_size: int, candidate_type: str, query_id: str, requested_k: int, raw_rows: Sequence[Mapping[str, Any]], results: Sequence[Mapping[str, Any]]) -> None:
    payload = {"experiment_version": GOAL3_EXPERIMENT_VERSION, "status": "complete", "phase": phase, "corpus_user_count": cohort_size, "candidate_type": candidate_type, "query_id": query_id, "requested_k": requested_k, "raw_row_count": len(raw_rows), "result_row_count": len(results), "raw_sha256": canonical_json_sha256(list(raw_rows)), "result_sha256": canonical_json_sha256(list(results)), "completed_at": datetime.now(UTC).isoformat()}
    # Several HNSW settings can be confirmed for the same N/query/K.  The
    # payload hashes make each immutable completed batch addressable without
    # overwriting a neighbouring setting's checkpoint.
    checkpoint_id = canonical_json_sha256(
        {"raw_sha256": payload["raw_sha256"], "result_sha256": payload["result_sha256"]}
    )[:16]
    filename = (
        f"{phase}-n{cohort_size}-{candidate_type}-{query_id}-k{requested_k}"
        f"-{checkpoint_id}.json"
    ).replace("/", "_")
    write_immutable_json(args.output_root / "checkpoints" / filename, payload)


def _plan(args: argparse.Namespace) -> Mapping[str, Any]:
    path = args.output_root / "experiment-plan.json"
    if not path.is_file():
        raise RuntimeError("Goal 3 plan is required before live execution")
    return read_json(path)


def _lock_execution_code_fingerprint(args: argparse.Namespace, *, phase: str) -> None:
    """Capture the exact runner revision at the first live checkpoint.

    The immutable planning fingerprint records plan construction; this sibling
    lock keeps a later runner-only safety correction auditable without ever
    rewriting the plan or its original fingerprint.
    """

    code_paths = (
        ROOT / "offline_evaluation/ann_search_benchmark.py",
        ROOT / "offline_evaluation/ann_search_goal3.py",
        ROOT / "scripts/run_ann_scale_goal3.py",
    )
    # The first calibration was started before phase-specific locks existed;
    # retain that immutable generic lock as its historical code fingerprint.
    if phase == "calibration" and (args.output_root / "execution-code-fingerprint.json").exists():
        return
    write_immutable_json(
        args.output_root / f"execution-code-fingerprint-{phase}.json",
        measurement_fingerprint(goal1_root=args.goal1_root, code_paths=code_paths),
    )


def _k_provenance_map(args: argparse.Namespace) -> Mapping[int, str]:
    payload = read_json(args.output_root / "k-provenance.json")
    rows = payload.get("required_k_values")
    if not isinstance(rows, list):
        raise ValueError("Goal 3 K provenance is invalid")
    result = {int(row["requested_k"]): str(row["provenance"]) for row in rows if isinstance(row, Mapping)}
    if set(result) != set(REQUIRED_K_VALUES):
        raise ValueError("Goal 3 K provenance coverage is incomplete")
    return result


def _calibration_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (str(row["candidate_type"]), str(row["query_id"]), int(row["corpus_user_count"]), int(row["requested_k"]))


def _confirmation_key(row: Mapping[str, Any]) -> tuple[str, str, int, int, int, str, int]:
    hnsw = row.get("hnsw")
    if not isinstance(hnsw, Mapping):
        raise ValueError("confirmation row HNSW settings are invalid")
    return (
        str(row["candidate_type"]),
        str(row["query_id"]),
        int(row["corpus_user_count"]),
        int(row["requested_k"]),
        int(hnsw["ef_search"]),
        str(hnsw["iterative_scan"]),
        int(hnsw["max_scan_tuples"]),
    )


def _fallback_setting_key(row: Mapping[str, Any]) -> tuple[str, int, int, int, str, int]:
    """Match tuning and independent-confirmation rows by runtime setting."""

    hnsw = row.get("hnsw")
    if not isinstance(hnsw, Mapping):
        raise ValueError("fallback row HNSW settings are invalid")
    return (
        str(row["candidate_type"]),
        int(row["corpus_user_count"]),
        int(row["requested_k"]),
        int(hnsw["ef_search"]),
        str(hnsw["iterative_scan"]),
        int(hnsw["max_scan_tuples"]),
    )


def _environment(args: argparse.Namespace, cohort_size: int) -> Mapping[str, str]:
    values = dotenv_values(args.env_file)
    missing = [key for key, value in values.items() if value is None]
    if missing:
        raise RuntimeError("environment file has empty values")
    return {**{str(key): str(value) for key, value in values.items() if value is not None}, **os.environ, "LOOPAD_AURORA_DATABASE": cohort_database_name(args.postgres_prefix, cohort_size), "LOOPAD_CLICKHOUSE_DATABASE": args.clickhouse_database, "ANN_POSTGRES_CONTAINER": args.postgres_container}


def _positive_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(item) for item in value.split(",") if item.strip())
    if not parsed or any(item <= 0 for item in parsed) or len(set(parsed)) != len(parsed):
        raise argparse.ArgumentTypeError("cohort sizes must be unique positive integers")
    return parsed


def _require_full_scale(args: argparse.Namespace) -> None:
    if tuple(args.cohort_sizes) != COHORT_SIZES:
        raise ValueError("Goal 3 final runner requires every frozen actual cohort")


def _seed_for(*values: object) -> int:
    return int(canonical_json_sha256(list(values))[:16], 16)


def _progress(event: str, **fields: Any) -> None:
    print(json.dumps({"event": event, **fields}, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
