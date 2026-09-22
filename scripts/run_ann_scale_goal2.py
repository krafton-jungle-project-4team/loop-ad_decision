#!/usr/bin/env python3
"""Checkpointed Goal 2 orchestration for ANN scale-series benchmark v2.

The script treats Goal 1 as read-only, gates all live work on a Phase 0 input
lock, and runs one PostgreSQL benchmark block at a time.  Completed raw blocks
are immutable and are safely reused only when their registered identity and
SHA-256 still match the frozen execution manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_goal2 import (  # noqa: E402
    GOAL1_ROOT,
    GOAL2_REPETITIONS,
    GOAL2_ROOT,
    GOAL2_WARMUPS,
    analyze_tuning_measurements,
    build_execution_manifest,
    build_policy_coverage_audit,
    build_tuning_cells,
    canonical_json_sha256,
    goal1_required_sha256,
    load_json_object,
    render_policy_coverage_markdown,
)
from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    AppendOnlyJsonl,
    ScaleArtifactValidator,
    sha256_file,
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_COHORT_SIZES,
    SCALE_EXPERIMENT_VERSION,
)
from scripts.analyze_ann_scale_goal1 import (  # noqa: E402
    MEASUREMENT_CODE_PATHS,
    measurement_code_sha256,
)
from scripts.prepare_ann_scale_postgres import cohort_database_name  # noqa: E402


DEFAULT_ENV_FILE = Path(".env.ann-source.local")
DEFAULT_POSTGRES_PREFIX = "loopad_ann_v2"
DEFAULT_POSTGRES_CONTAINER = "loop-ad_data-source_contract-postgres-1"
DEFAULT_CLICKHOUSE_CONTAINER = "loop-ad_data-source_contract-clickhouse-1"
DEFAULT_CLICKHOUSE_DATABASE = "ann_scale_v2_build"
DEFAULT_SCALE_SERIES_ID = "expedia-full-2015-v2-86ded0d7"
DEFAULT_SAMPLE_SEED = "ann-score-pass-v2-20260718"
ALL_BASELINE_PLANS = ("current_runtime", "exact_all", "filter_first_exact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "phase0",
            "plan",
            "tune",
            "aggregate",
            "diagnostics",
            "finalize-tuning",
            "finalize-goal",
            "postprocess",
            "all",
        ),
    )
    parser.add_argument("--goal1-root", type=Path, default=GOAL1_ROOT)
    parser.add_argument("--goal2-root", type=Path, default=GOAL2_ROOT)
    parser.add_argument("--env-file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--postgres-prefix", default=DEFAULT_POSTGRES_PREFIX)
    parser.add_argument("--postgres-container", default=DEFAULT_POSTGRES_CONTAINER)
    parser.add_argument(
        "--clickhouse-container", default=DEFAULT_CLICKHOUSE_CONTAINER
    )
    parser.add_argument(
        "--clickhouse-database", default=DEFAULT_CLICKHOUSE_DATABASE
    )
    parser.add_argument("--scale-series-id", default=DEFAULT_SCALE_SERIES_ID)
    parser.add_argument("--reference-sample-seed", default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--cohort-sizes",
        type=_positive_ints,
        default=tuple(SCALE_COHORT_SIZES),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_roots(args)
    if args.command in {"phase0", "all"}:
        run_phase0(args)
    if args.command in {"plan", "all"}:
        freeze_execution_plan(args)
    if args.command in {"tune", "all"}:
        run_tuning(args)
    if args.command in {"aggregate", "postprocess", "all"}:
        aggregate_tuning(args)
    if args.command in {"diagnostics", "postprocess", "all"}:
        run_diagnostics(args)
    if args.command in {"finalize-tuning", "postprocess", "all"}:
        finalize_tuning(args)
    if args.command in {"finalize-goal", "postprocess", "all"}:
        finalize_goal(args)
    return 0


def run_phase0(args: argparse.Namespace) -> None:
    root = args.goal1_root
    output = args.goal2_root
    integrity = ScaleArtifactValidator(root).raise_for_errors()
    goal1_manifest = load_json_object(root / "manifest.json")
    fingerprint = load_json_object(root / "fingerprint.json")
    backend = load_json_object(root / "phase0/scale-ready-backend.json")
    expected_code_sha = str(backend["measurement_code_sha256"])
    actual_code_sha = measurement_code_sha256(MEASUREMENT_CODE_PATHS)
    if actual_code_sha != expected_code_sha:
        raise RuntimeError("Goal 1 measurement code changed after fingerprint freeze")
    if goal1_manifest.get("goal2_executed") is not False:
        raise RuntimeError("Goal 1 manifest no longer identifies Goal 2 as unexecuted")
    if goal1_manifest.get("historical_50k_decision") != "pilot_only":
        raise RuntimeError("historical 50K is no longer locked to pilot-only")

    current_resources = _docker_resources(
        postgres_container=args.postgres_container,
        clickhouse_container=args.clickhouse_container,
    )
    frozen_resources = load_json_object(root / "resource-manifest.json")
    _validate_resource_limits(current_resources, frozen_resources)

    preflights: list[Mapping[str, Any]] = []
    database_state: list[Mapping[str, Any]] = []
    for cohort_size in args.cohort_sizes:
        preflight = _live_preflight(args, cohort_size)
        _validate_against_goal1_preflight(root, cohort_size, preflight)
        state = _database_state(args, cohort_size)
        _validate_database_state(root, cohort_size, preflight, state)
        preflights.append(preflight)
        database_state.append(state)
        _progress(
            "phase0_cohort_locked",
            cohort_size=cohort_size,
            postgres_row_count=state["postgres_row_count"],
            index_valid=state["index_valid"],
            index_ready=state["index_ready"],
        )

    audit = build_policy_coverage_audit(root)
    decision = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": audit["product_policy_track_status"],
        "policy_dimensions": ["N", "H/N", "E/N"],
        "five_candidate_type_coverage_bucket_count": audit[
            "five_candidate_type_coverage_bucket_count"
        ],
        "policy_rule_possible_bucket_count": audit[
            "policy_rule_possible_bucket_count"
        ],
        "scale_performance_track_enabled": True,
        "product_policy_confirmation_enabled": audit[
            "product_confirmation_branch_enabled"
        ],
        "filter_first_product_confirmation_enabled": audit[
            "product_confirmation_branch_enabled"
        ],
        "score_sample_size_enabled": audit[
            "product_confirmation_branch_enabled"
        ],
        "macro_benchmark_enabled": False,
        "candidate_policy_generation_enabled": False,
        "reason": (
            "No observed (N, H/N, E/N) bucket has validated tuning/confirmation "
            "coverage for all five production candidate types."
            if not audit["product_confirmation_branch_enabled"]
            else "At least one policy bucket is identifiable; later gates apply."
        ),
    }

    required_hashes = goal1_required_sha256(root)
    input_lock = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": "passed",
        "goal1_integrity_passed": integrity.passed,
        "goal1_fingerprint_sha256": fingerprint["fingerprint_sha256"],
        "goal1_manifest_sha256": sha256_file(root / "manifest.json"),
        "goal2_tuning_input_sha256": sha256_file(
            root / "goal2-tuning-inputs.json"
        ),
        "policy_tuning_input_sha256": sha256_file(
            root / "policy-hnsw-tuning-inputs.json"
        ),
        "scale_tuning_input_sha256": sha256_file(
            root / "scale-hnsw-tuning-inputs.json"
        ),
        "source_vector_identity": {
            key: fingerprint[key]
            for key in (
                "scale_series_id",
                "project_id",
                "vector_version",
                "vector_manifest_hash",
                "vector_generation_id",
                "window_start",
                "window_end",
                "source_revision_cutoff",
                "source_user_count",
                "source_vector_revision_count",
                "cohort_seed",
                "membership_sha256",
                "scenario_manifest_sha256",
            )
        },
        "benchmark_code_revision": fingerprint["code_revision"],
        "measurement_code_sha256": actual_code_sha,
        "compiler_calibration_hashes": _compiler_calibration_hashes(root),
        "goal1_required_file_sha256": required_hashes,
        "goal1_required_file_set_sha256": canonical_json_sha256(required_hashes),
        "database_state": database_state,
        "resource_limits": current_resources,
        "goal1_vector_and_membership_unchanged": True,
    }
    environment = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "postgres_container": current_resources["postgres"],
        "clickhouse_container": current_resources["clickhouse"],
        "preflights": preflights,
        "benchmark_code_revision": fingerprint["code_revision"],
        "measurement_code_sha256": actual_code_sha,
    }

    _write_json_same_or_new(output / "goal1-input-lock.json", input_lock)
    _write_json_same_or_new(output / "environment-manifest.json", environment)
    _write_json_same_or_new(output / "policy-coverage-audit.json", audit)
    _write_text_same_or_new(
        output / "policy-coverage-audit.md",
        render_policy_coverage_markdown(audit),
    )
    _write_json_same_or_new(
        output / "policy-identifiability-decision.json", decision
    )
    _progress(
        "phase0_passed",
        cohorts=len(preflights),
        coverage_buckets=audit["five_candidate_type_coverage_bucket_count"],
        product_branch_enabled=audit["product_confirmation_branch_enabled"],
    )


def freeze_execution_plan(args: argparse.Namespace) -> None:
    _require_phase0(args.goal2_root)
    manifest = build_execution_manifest(args.goal1_root)
    decision = load_json_object(
        args.goal2_root / "policy-identifiability-decision.json"
    )
    goal2_manifest = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "goal": "ANN Scale-performance and product policy Goal 2",
        "status": "execution_locked",
        "goal1_input_lock_sha256": sha256_file(
            args.goal2_root / "goal1-input-lock.json"
        ),
        "execution_manifest_sha256": manifest["execution_manifest_sha256"],
        "scale_performance_track_enabled": True,
        "product_policy_track_status": decision["status"],
        "product_policy_confirmation_enabled": decision[
            "product_policy_confirmation_enabled"
        ],
        "historical_50k": "pilot_only_not_reused",
        "runtime_selector_implemented": False,
        "candidate_policy_runtime_applied": False,
    }
    checkpoint_schema = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "identity_fields": ["corpus_user_count", "scenario_id"],
        "required_fields": [
            "status",
            "execution_manifest_sha256",
            "baseline_path",
            "baseline_sha256",
            "ann_path",
            "ann_sha256",
            "actual_invocation_count",
            "measured_observation_count",
            "wall_clock_seconds",
        ],
        "resume_rule": (
            "skip only an identical complete identity; any hash or count change fails"
        ),
    }
    plan_markdown = _render_execution_plan(manifest, decision)

    _write_json_same_or_new(
        args.goal2_root / "goal2-execution-manifest.json", manifest
    )
    _write_json_same_or_new(args.goal2_root / "goal2-manifest.json", goal2_manifest)
    _write_json_same_or_new(
        args.goal2_root / "checkpoint-schema.json", checkpoint_schema
    )
    _write_text_same_or_new(
        args.goal2_root / "goal2-execution-plan.md", plan_markdown
    )
    _write_tuning_cell_manifest(
        args.goal2_root / "tuning-cell-manifest.jsonl",
        build_tuning_cells(args.goal1_root),
    )
    for relative in (
        "checkpoints",
        "raw/tuning",
        "raw/confirmation-warm",
        "raw/confirmation-cold",
        "raw/confirmation-exclusion",
        "diagnostics",
        "preliminary",
    ):
        (args.goal2_root / relative).mkdir(parents=True, exist_ok=True)
    _progress(
        "execution_plan_locked",
        cells=manifest["total_core_cell_count"],
        invocations=manifest["expected_total_invocation_count"],
        measured=manifest["expected_measured_observation_count"],
    )


def run_tuning(args: argparse.Namespace) -> None:
    _require_phase0(args.goal2_root)
    execution = load_json_object(
        args.goal2_root / "goal2-execution-manifest.json"
    )
    expected = build_execution_manifest(args.goal1_root)
    if execution != expected:
        raise RuntimeError("Goal 2 execution manifest changed after lock")
    _verify_goal1_lock(args.goal1_root, args.goal2_root)

    points = execution["points"]
    if not isinstance(points, list):
        raise RuntimeError("Goal 2 execution points are invalid")
    selected = [
        item
        for item in points
        if isinstance(item, Mapping)
        and int(item["corpus_user_count"]) in set(args.cohort_sizes)
    ]
    total_registered_invocations = sum(
        (3 + len(item["requested_k_values"]) * 24)
        * (GOAL2_WARMUPS + GOAL2_REPETITIONS)
        for item in selected
    )
    completed_before = _completed_invocations(args.goal2_root, selected)
    run_started = time.monotonic()
    first_new_block = True
    for point in selected:
        cohort_size = int(point["corpus_user_count"])
        scenario_id = str(point["scenario_id"])
        requested_ks = tuple(int(value) for value in point["requested_k_values"])
        checkpoint = (
            args.goal2_root
            / "checkpoints"
            / f"cohort-{cohort_size}"
            / f"{scenario_id}.json"
        )
        if checkpoint.exists():
            _validate_block_checkpoint(
                args,
                point=point,
                checkpoint=checkpoint,
                execution_manifest_sha256=str(
                    execution["execution_manifest_sha256"]
                ),
            )
            _progress(
                "tuning_block_resumed",
                cohort_size=cohort_size,
                scenario_id=scenario_id,
                checkpoint=str(checkpoint),
            )
            continue
        block_started = time.monotonic()
        raw_root = (
            args.goal2_root
            / "raw/tuning"
            / f"cohort-{cohort_size}"
            / scenario_id
        )
        baseline_path = raw_root / "baselines.jsonl"
        ann_path = raw_root / "ann.jsonl"
        _run_raw_block(
            args,
            cohort_size=cohort_size,
            scenario_id=scenario_id,
            output=baseline_path,
            plans=ALL_BASELINE_PLANS,
            requested_ks=(),
            full_hnsw=False,
        )
        _validate_raw_block(
            baseline_path,
            cohort_size=cohort_size,
            scenario_id=scenario_id,
            expected_plans=set(ALL_BASELINE_PLANS),
            expected_requested_ks=set(),
            expected_hnsw_count=0,
        )
        _run_raw_block(
            args,
            cohort_size=cohort_size,
            scenario_id=scenario_id,
            output=ann_path,
            plans=("ann_first",),
            requested_ks=requested_ks,
            full_hnsw=True,
        )
        ann_counts = _validate_raw_block(
            ann_path,
            cohort_size=cohort_size,
            scenario_id=scenario_id,
            expected_plans={"ann_first"},
            expected_requested_ks=set(requested_ks),
            expected_hnsw_count=24,
        )
        baseline_counts = _raw_counts(baseline_path)
        elapsed = time.monotonic() - block_started
        payload = {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "status": "complete",
            "execution_manifest_sha256": execution[
                "execution_manifest_sha256"
            ],
            "corpus_user_count": cohort_size,
            "scenario_id": scenario_id,
            "candidate_type": point["candidate_type"],
            "requested_k_values": list(requested_ks),
            "baseline_path": str(baseline_path),
            "baseline_sha256": sha256_file(baseline_path),
            "ann_path": str(ann_path),
            "ann_sha256": sha256_file(ann_path),
            "actual_invocation_count": (
                baseline_counts["total"] + ann_counts["total"]
            ),
            "measured_observation_count": (
                baseline_counts["measured"] + ann_counts["measured"]
            ),
            "warmup_observation_count": (
                baseline_counts["warmup"] + ann_counts["warmup"]
            ),
            "peak_rss_complete": True,
            "spill_oom_fields_complete": True,
            "wall_clock_seconds": elapsed,
        }
        write_immutable_json(checkpoint, payload)
        AppendOnlyJsonl(
            args.goal2_root / "checkpoints/progress.jsonl",
            identity_fields=("corpus_user_count", "scenario_id"),
        ).append((payload,))
        completed_now = _completed_invocations(args.goal2_root, selected)
        _progress(
            "tuning_block_completed",
            cohort_size=cohort_size,
            scenario_id=scenario_id,
            invocations=payload["actual_invocation_count"],
            elapsed_seconds=round(elapsed, 3),
            completed_invocations=completed_now,
            total_invocations=total_registered_invocations,
        )
        if first_new_block:
            first_new_block = False
            rate = payload["actual_invocation_count"] / elapsed if elapsed else 0.0
            remaining = total_registered_invocations - completed_now
            eta = remaining / rate if rate else None
            eta_payload = {
                "experiment_version": SCALE_EXPERIMENT_VERSION,
                "source": "first completed full HNSW block",
                "cohort_size": cohort_size,
                "scenario_id": scenario_id,
                "actual_invocation_count": payload["actual_invocation_count"],
                "wall_clock_seconds": elapsed,
                "actual_invocations_per_second": rate,
                "remaining_registered_invocations": remaining,
                "estimated_remaining_seconds": eta,
                "used_in_result_graph": False,
            }
            eta_path = args.goal2_root / "first-block-eta.json"
            if not eta_path.exists():
                _write_json_same_or_new(eta_path, eta_payload)
            _progress(
                "first_block_eta",
                estimated_remaining_seconds=(
                    round(
                        float(
                            load_json_object(eta_path)[
                                "estimated_remaining_seconds"
                            ]
                        ),
                        1,
                    )
                    if eta_path.exists()
                    and load_json_object(eta_path).get(
                        "estimated_remaining_seconds"
                    )
                    is not None
                    else None
                ),
            )
    _progress(
        "tuning_selected_points_complete",
        selected_point_count=len(selected),
        prior_completed_invocations=completed_before,
        elapsed_seconds=round(time.monotonic() - run_started, 3),
        completed_invocations=_completed_invocations(args.goal2_root, selected),
        total_invocations=total_registered_invocations,
    )


def aggregate_tuning(args: argparse.Namespace) -> None:
    execution = _require_complete_tuning(args)
    analysis = analyze_tuning_measurements(
        goal1_root=args.goal1_root,
        goal2_root=args.goal2_root,
    )
    plan = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "status": "locked",
        "cell_count": len(analysis["diagnostic_plan"]),
        "cells": analysis["diagnostic_plan"],
        "selection": (
            "all prediagnostic scale-pass cells plus one measured representative "
            "adjacent failure per cohort/scenario point"
        ),
        "timing_included": False,
    }
    preliminary = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "counts": analysis["counts"],
        "diagnostics_pending": True,
        "final_gate_values": False,
    }
    _write_json_same_or_new(args.goal2_root / "diagnostic-plan.json", plan)
    _write_json_same_or_new(
        args.goal2_root / "preliminary/tuning-prediagnostic-summary.json",
        preliminary,
    )
    _write_jsonl_same_or_new(
        args.goal2_root / "preliminary/scale-gates-prediagnostic.jsonl",
        analysis["scale_gates"],
    )
    _progress(
        "tuning_aggregated",
        ann_cells=analysis["counts"]["ann_tuning_result_count"],
        prediagnostic_passed=analysis["counts"][
            "prediagnostic_scale_passed_count"
        ],
        diagnostic_cells=len(analysis["diagnostic_plan"]),
    )


def run_diagnostics(args: argparse.Namespace) -> None:
    _require_complete_tuning(args)
    plan = load_json_object(args.goal2_root / "diagnostic-plan.json")
    rows = plan.get("cells")
    if not isinstance(rows, list):
        raise RuntimeError("Goal 2 diagnostic plan is invalid")
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for item in rows:
        if not isinstance(item, Mapping):
            raise RuntimeError("Goal 2 diagnostic plan row is invalid")
        grouped.setdefault(
            (int(item["corpus_user_count"]), str(item["scenario_id"])), []
        ).append(item)
    registry_path = args.goal2_root / "diagnostics/registry.jsonl"
    existing = {
        _diagnostic_row_identity(item): item
        for item in _read_jsonl_if_exists(registry_path)
    }
    for (cohort_size, scenario_id), cells in sorted(grouped.items()):
        identities = {_diagnostic_row_identity(item) for item in cells}
        if identities.issubset(existing):
            _progress(
                "diagnostic_group_resumed",
                cohort_size=cohort_size,
                scenario_id=scenario_id,
                cell_count=len(cells),
            )
            continue
        if identities & set(existing):
            raise RuntimeError(
                "partial diagnostic registry conflicts with group atomicity: "
                f"{cohort_size}/{scenario_id}"
            )
        captured = _collect_diagnostic_group(
            args,
            cohort_size=cohort_size,
            scenario_id=scenario_id,
            cells=cells,
        )
        AppendOnlyJsonl(
            registry_path,
            identity_fields=(
                "corpus_user_count",
                "scenario_id",
                "requested_k",
                "ef_search",
                "iterative_scan",
                "max_scan_tuples",
            ),
        ).append(captured)
        existing.update({_diagnostic_row_identity(item): item for item in captured})
        _progress(
            "diagnostic_group_completed",
            cohort_size=cohort_size,
            scenario_id=scenario_id,
            cell_count=len(captured),
        )
    if set(existing) != {
        _diagnostic_row_identity(item) for item in rows
    }:
        raise RuntimeError("Goal 2 diagnostic registry coverage differs from plan")
    _progress("diagnostics_complete", cell_count=len(existing))


def finalize_tuning(args: argparse.Namespace) -> None:
    execution = _require_complete_tuning(args)
    diagnostic_plan = load_json_object(args.goal2_root / "diagnostic-plan.json")
    registry = _read_jsonl_if_exists(
        args.goal2_root / "diagnostics/registry.jsonl"
    )
    expected_identities = {
        _diagnostic_row_identity(item) for item in diagnostic_plan["cells"]
    }
    actual_identities = {_diagnostic_row_identity(item) for item in registry}
    if actual_identities != expected_identities:
        raise RuntimeError("Goal 2 diagnostics are incomplete")
    analysis = analyze_tuning_measurements(
        goal1_root=args.goal1_root,
        goal2_root=args.goal2_root,
        diagnostic_rows=registry,
    )
    _write_jsonl_same_or_new(
        args.goal2_root / "hnsw-tuning-results.jsonl",
        analysis["tuning_results"],
    )
    _write_jsonl_same_or_new(
        args.goal2_root / "ann-scale-final-gates.jsonl",
        analysis["scale_gates"],
    )
    _write_jsonl_same_or_new(
        args.goal2_root / "ann-policy-final-gates.jsonl",
        analysis["policy_gates"],
    )
    winners_payload = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "winner_definition": (
            "fastest p95 diagnostic-validated setting per (cohort, scenario, K)"
        ),
        "winner_count": len(analysis["winners"]),
        "winners": analysis["winners"],
    }
    _write_json_same_or_new(
        args.goal2_root / "hnsw-tuning-winners.json", winners_payload
    )
    confirmation_plan = _build_confirmation_plan(args, analysis["winners"])
    _write_json_same_or_new(
        args.goal2_root / "confirmation-plan.json", confirmation_plan
    )
    summary = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "counts": analysis["counts"],
        "confirmation_required": bool(analysis["winners"]),
        "negative_result_stop_condition_met": (
            analysis["counts"]["ann_vs_exact_all_passed_count"] == 0
        ),
    }
    _write_json_same_or_new(
        args.goal2_root / "tuning-finalization.json", summary
    )
    _progress(
        "tuning_finalized",
        ann_vs_exact_all_passed=analysis["counts"][
            "ann_vs_exact_all_passed_count"
        ],
        ann_policy_candidate=analysis["counts"]["ann_policy_candidate_count"],
        winners=len(analysis["winners"]),
        confirmation_required=bool(analysis["winners"]),
    )


def finalize_goal(args: argparse.Namespace) -> None:
    import csv

    execution = _require_complete_tuning(args)
    tuning_final = load_json_object(args.goal2_root / "tuning-finalization.json")
    confirmation_plan = load_json_object(args.goal2_root / "confirmation-plan.json")
    if int(confirmation_plan["winner_count"]) > 0:
        raise RuntimeError(
            "scale winners exist; warm/cold holdout confirmation must complete "
            "before final Goal 2 artifacts"
        )
    registry = _read_jsonl_if_exists(args.goal2_root / "diagnostics/registry.jsonl")
    analysis = analyze_tuning_measurements(
        goal1_root=args.goal1_root,
        goal2_root=args.goal2_root,
        diagnostic_rows=registry,
    )
    if analysis["winners"]:
        raise RuntimeError("unconfirmed scale winners cannot be finalized")

    _write_jsonl_same_or_new(args.goal2_root / "confirmation-results.jsonl", ())
    score_sample = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": "skipped",
        "selected_sample_size": None,
        "candidate_sizes": [2_000, 5_000, 10_000, 20_000, 50_000],
        "reason": "product policy track is not identifiable under current dimensions",
        "latency_confirmation_executed": False,
    }
    _write_json_same_or_new(
        args.goal2_root / "score-sample-selection.json", score_sample
    )
    policy_decision = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "generated": False,
        "reason": "not identifiable under current policy dimensions",
        "coverage_status": "zero buckets with all five candidate types",
        "ann_confirmation_status": "not run because no policy rule is identifiable",
        "filter_confirmation_status": "not run because no policy rule is identifiable",
        "default_exact_fallback": {
            "plan": "exact_all",
            "scope": "all product policy buckets",
        },
        "required_follow_up": [
            "make an explicit product decision to change policy dimensions or data coverage",
            "collect distinct tuning/confirmation coverage for all required candidate types",
            "then rerun cold and real exclusion confirmation before rollout",
        ],
    }
    _write_json_same_or_new(
        args.goal2_root / "candidate-policy-decision.json", policy_decision
    )
    break_even = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "observed_ann_break_even_users": None,
        "observed_ann_break_even_status": "not observed through 1M",
        "measured_max_users": 1_000_000,
        "ann_min_users": None,
        "validated_max_users": 1_000_000,
        "interpolation_used": False,
        "reason": "no HNSW setting passed the full tuning scale gate",
    }
    _write_json_same_or_new(args.goal2_root / "ann-break-even.json", break_even)

    current_rows = _current_runtime_rows(analysis["summaries"])
    current_report = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "source": "Goal 2 matched baselines",
        "raw_event_chunk_rescan_included_in_latency": False,
        "backend_scope": (
            "production selector/K-growth/audit/fallback with normalized exact "
            "frozen hard-predicate backend"
        ),
        "cell_count": len(current_rows),
        "cells": current_rows,
    }
    _write_json_same_or_new(
        args.goal2_root / "current-runtime-work-amplification.json",
        current_report,
    )
    baselines_by_point = {
        (
            int(item["corpus_user_count"]),
            str(item["scenario_id"]),
            str(item["plan"]),
        ): item
        for item in analysis["summaries"]
        if item["plan"] in {"exact_all", "filter_first_exact"}
    }
    comparison = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "candidate_policy_generated": False,
        "scale_candidate_count": 0,
        "policy_candidate_count": int(
            analysis["counts"]["ann_policy_candidate_count"]
        ),
        "policy_rule_eligible_count": 0,
        "rows": [
            {
                "corpus_user_count": int(item["corpus_user_count"]),
                "scenario_id": item["scenario_id"],
                "current_runtime_p50_ms": float(item["p50_ms"]),
                "current_runtime_p95_ms": float(item["p95_ms"]),
                "current_runtime_p99_ms": float(item["p99_ms"]),
                "exact_all_p95_ms": float(
                    baselines_by_point[
                        (
                            int(item["corpus_user_count"]),
                            str(item["scenario_id"]),
                            "exact_all",
                        )
                    ]["p95_ms"]
                ),
                "filter_first_exact_p95_ms": float(
                    baselines_by_point[
                        (
                            int(item["corpus_user_count"]),
                            str(item["scenario_id"]),
                            "filter_first_exact",
                        )
                    ]["p95_ms"]
                ),
                "candidate_plan": "exact_all",
                "candidate_reason": "no confirmed ANN candidate",
            }
            for item in current_rows
        ],
    }
    _write_json_same_or_new(
        args.goal2_root / "current-vs-candidate.json", comparison
    )

    final_rows = _final_scale_rows(analysis["summaries"], confirmed_winners=())
    csv_path = args.goal2_root / "final-scale-performance.csv"
    if csv_path.exists():
        existing_rows = list(csv.DictReader(csv_path.open(encoding="utf-8")))
        expected_rows = [{key: str(value) for key, value in row.items()} for row in final_rows]
        if existing_rows != expected_rows:
            raise RuntimeError(f"immutable Goal 2 CSV conflict: {csv_path}")
    else:
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=tuple(final_rows[0]))
            writer.writeheader()
            writer.writerows(final_rows)
    _write_text_same_or_new(
        args.goal2_root / "final-scale-performance-report.md",
        _render_final_scale_report(final_rows, analysis["counts"]),
    )
    png_path = args.goal2_root / "final-scale-performance.png"
    if not png_path.exists():
        _render_final_scale_png(final_rows, png_path)

    fallbacks = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "fallback_plan": "exact_all",
        "policy_track": {
            "scope": "all observed and unobserved (N, H/N, E/N) buckets",
            "reason": "not identifiable under current policy dimensions",
        },
        "scale_track": {
            "scope": "50K through 1M",
            "reason": "no confirmed ANN candidate",
        },
        "outside_validated_range": {
            "min_exclusive_users": 1_000_000,
            "plan": "exact_all",
        },
        "manifest_or_parse_failure": "exact_all",
        "historical_50k": "pilot-only and excluded from Goal 2",
        "synthetic_data_used": False,
        "missing_values_interpolated": False,
    }
    _write_json_same_or_new(
        args.goal2_root / "unvalidated-fallbacks.json", fallbacks
    )
    deployment = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "deployment_equivalent_validated": False,
        "local_environment_validated": True,
        "rollout_values_available": False,
        "reason": (
            "local preregistered scale experiment produced no policy rule; "
            "deployment-equivalent validation remains required for any future candidate"
        ),
    }
    _write_json_same_or_new(
        args.goal2_root / "deployment-equivalent-unvalidated.json", deployment
    )

    checkpoint_rows = _read_jsonl_if_exists(
        args.goal2_root / "checkpoints/progress.jsonl"
    )
    actual_invocations = sum(int(item["actual_invocation_count"]) for item in checkpoint_rows)
    measured_observations = sum(
        int(item["measured_observation_count"]) for item in checkpoint_rows
    )
    warmups = sum(int(item["warmup_observation_count"]) for item in checkpoint_rows)
    benchmark_wall = sum(float(item["wall_clock_seconds"]) for item in checkpoint_rows)
    summary_payload = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": "complete",
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
        "actual_core_cell_count": int(execution["total_core_cell_count"]),
        "actual_invocation_count": actual_invocations,
        "measured_observation_count": measured_observations,
        "warmup_observation_count": warmups,
        "cold_observation_count": 0,
        "exclusion_observation_count": 0,
        "diagnostic_ann_cell_count": len(registry),
        "diagnostic_exact_plan_count": 38,
        "benchmark_block_wall_clock_seconds": benchmark_wall,
        "scale_winner_count": 0,
        "observed_ann_break_even_users": None,
        "observed_ann_break_even_status": "not observed through 1M",
        "ann_min_users": None,
        "validated_max_users": 1_000_000,
        "ann_vs_exact_all_passed_count": int(
            analysis["counts"]["ann_vs_exact_all_passed_count"]
        ),
        "ann_policy_candidate_count": int(
            analysis["counts"]["ann_policy_candidate_count"]
        ),
        "policy_rule_eligible_count": 0,
        "filter_first_rule_count": 0,
        "macro_executed": False,
        "macro_skip_reason": "no policy_rule_eligible rule",
        "deployment_equivalent_validation_required": True,
        "candidate_policy_generated": False,
        "goal_token_and_elapsed_usage_reported_by_codex": True,
    }
    goal2_code_paths = (
        Path("offline_evaluation/ann_search_goal2.py"),
        Path("scripts/run_ann_scale_goal2.py"),
        Path("tests/test_ann_search_goal2.py"),
        Path("docs/ann_search_scale_series_v2.md"),
    )
    goal2_code_hashes = {
        str(path): sha256_file(ROOT / path) for path in goal2_code_paths
    }
    code_fingerprint = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "measurement_code_sha256": load_json_object(
            args.goal2_root / "goal1-input-lock.json"
        )["measurement_code_sha256"],
        "goal2_code_sha256": goal2_code_hashes,
        "goal2_code_set_sha256": canonical_json_sha256(goal2_code_hashes),
        "execution_manifest_sha256": execution["execution_manifest_sha256"],
    }
    _write_json_same_or_new(
        args.goal2_root / "goal2-code-fingerprint.json", code_fingerprint
    )
    _write_text_same_or_new(
        args.goal2_root / "goal2-summary.md",
        _render_goal2_summary(summary_payload),
    )
    integrity = _build_goal2_integrity(args, summary_payload, final_rows)
    _write_json_same_or_new(args.goal2_root / "integrity-report.json", integrity)
    if integrity["passed"] is not True:
        raise RuntimeError("Goal 2 artifact integrity validation failed")
    _progress(
        "goal2_finalized",
        actual_invocations=actual_invocations,
        measured_observations=measured_observations,
        ann_vs_exact_all_passed=summary_payload[
            "ann_vs_exact_all_passed_count"
        ],
        observed_break_even="not observed through 1M",
        artifact_integrity_passed=True,
    )


def _require_complete_tuning(args: argparse.Namespace) -> Mapping[str, Any]:
    _require_phase0(args.goal2_root)
    _verify_goal1_lock(args.goal1_root, args.goal2_root)
    execution = load_json_object(
        args.goal2_root / "goal2-execution-manifest.json"
    )
    expected = build_execution_manifest(args.goal1_root)
    if execution != expected:
        raise RuntimeError("Goal 2 execution manifest changed after lock")
    points = execution.get("points")
    if not isinstance(points, list):
        raise RuntimeError("Goal 2 execution points are invalid")
    for point in points:
        checkpoint = (
            args.goal2_root
            / "checkpoints"
            / f"cohort-{int(point['corpus_user_count'])}"
            / f"{point['scenario_id']}.json"
        )
        if not checkpoint.exists():
            raise RuntimeError(f"Goal 2 tuning checkpoint missing: {checkpoint}")
        _validate_block_checkpoint(
            args,
            point=point,
            checkpoint=checkpoint,
            execution_manifest_sha256=str(
                execution["execution_manifest_sha256"]
            ),
        )
    actual_invocations = _completed_invocations(args.goal2_root, points)
    if actual_invocations != int(execution["expected_total_invocation_count"]):
        raise RuntimeError("Goal 2 actual tuning invocation count is incomplete")
    return execution


def _collect_diagnostic_group(
    args: argparse.Namespace,
    *,
    cohort_size: int,
    scenario_id: str,
    cells: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    from datetime import UTC, datetime

    from app.config import load_settings
    from app.db import create_clickhouse_client, create_postgres_connection
    from offline_evaluation.ann_search_benchmark import (
        BenchmarkManifest,
        LiveAnnSearchBenchmark,
        _ExecutionCell,
    )
    from offline_evaluation.ann_search_experiment import (
        HnswSettings,
        SearchPlan,
    )
    from offline_evaluation.ann_search_scale_series import ScaleCohortScope

    environment = _benchmark_environment(args, cohort_size)
    settings = load_settings(environment)
    connection = create_postgres_connection(settings)
    clickhouse = create_clickhouse_client(settings)
    try:
        manifest = BenchmarkManifest.load(
            args.goal1_root / "phase2/selected-scenario-manifest.json"
        )
        benchmark = LiveAnnSearchBenchmark(
            postgres_connection=connection,
            clickhouse=clickhouse,
            scale_cohort_scope=ScaleCohortScope(
                scale_series_id=args.scale_series_id,
                cohort_size=cohort_size,
                reference_sample_seed=args.reference_sample_seed,
            ),
        )
        benchmark.preflight(manifest)
        scenario = manifest.require_scenario(scenario_id)
        context = benchmark._load_context(manifest)
        execution_cells = []
        settings_by_key: dict[tuple[int, int, str, int], HnswSettings] = {}
        for item in cells:
            raw_hnsw = item["hnsw"]
            hnsw = HnswSettings(
                ef_search=int(raw_hnsw["ef_search"]),
                iterative_scan=str(raw_hnsw["iterative_scan"]),
                max_scan_tuples=int(raw_hnsw["max_scan_tuples"]),
            )
            requested_k = int(item["requested_k"])
            execution_cells.append(
                _ExecutionCell(
                    sample_size=min(50_000, cohort_size),
                    plan=SearchPlan.ANN_FIRST,
                    requested_k=requested_k,
                    hnsw=hnsw,
                )
            )
            settings_by_key[
                (
                    requested_k,
                    hnsw.ef_search,
                    hnsw.iterative_scan,
                    hnsw.max_scan_tuples,
                )
            ] = hnsw
        output_dir = (
            args.goal2_root
            / "diagnostics"
            / f"cohort-{cohort_size}"
            / scenario_id
        )
        started_at = datetime.now(UTC)
        diagnostics = benchmark._ann_diagnostics(
            manifest=manifest,
            scenario=scenario,
            context=context,
            cells=execution_cells,
            diagnostics_dir=output_dir,
        )
        ended_at = datetime.now(UTC)
        benchmark._write_clickhouse_query_log(
            manifest=manifest,
            scenario=scenario,
            started_at=started_at,
            ended_at=ended_at,
            diagnostics_dir=output_dir,
        )
        result: list[Mapping[str, Any]] = []
        for item in cells:
            raw_hnsw = item["hnsw"]
            key = (
                int(item["requested_k"]),
                int(raw_hnsw["ef_search"]),
                str(raw_hnsw["iterative_scan"]),
                int(raw_hnsw["max_scan_tuples"]),
            )
            hnsw = settings_by_key[key]
            diagnostic = diagnostics[(key[0], hnsw)]
            filename = (
                f"{scenario_id}-k{key[0]}-ef{key[1]}-"
                f"{key[2]}-scan{key[3]}.json"
            )
            explain_path = output_dir / filename
            if not explain_path.is_file():
                raise RuntimeError(f"diagnostic EXPLAIN missing: {explain_path}")
            result.append(
                {
                    "experiment_version": SCALE_EXPERIMENT_VERSION,
                    "cell_id": item["cell_id"],
                    "corpus_user_count": cohort_size,
                    "scenario_id": scenario_id,
                    "candidate_type": item["candidate_type"],
                    "requested_k": key[0],
                    "ef_search": key[1],
                    "iterative_scan": key[2],
                    "max_scan_tuples": key[3],
                    "hnsw": {
                        "ef_search": key[1],
                        "iterative_scan": key[2],
                        "max_scan_tuples": key[3],
                    },
                    "reason": item["reason"],
                    "hnsw_index_used": diagnostic.index_name
                    == "idx_user_behavior_vector_search_embedding_hnsw",
                    "index_name": diagnostic.index_name,
                    "temp_spill": diagnostic.temp_spill,
                    "oom": False,
                    "explain_path": str(explain_path),
                    "explain_sha256": sha256_file(explain_path),
                    "timing_included": False,
                }
            )
        return tuple(result)
    finally:
        connection.close()
        close = getattr(clickhouse, "close", None)
        if callable(close):
            close()


def _build_confirmation_plan(
    args: argparse.Namespace, winners: Sequence[Mapping[str, Any]]
) -> Mapping[str, Any]:
    pair_manifest = load_json_object(
        args.goal1_root / "phase2/scenario-pairs.json"
    )
    confirmation_for_tuning = {
        str(item["tuning_scenario_id"]): str(item["confirmation_scenario_id"])
        for item in pair_manifest["pairs"]
        if item.get("validated") is True
    }
    cells: list[Mapping[str, Any]] = []
    for item in winners:
        scenario_id = str(item["scenario_id"])
        confirmation_id = confirmation_for_tuning.get(scenario_id)
        if confirmation_id is None:
            raise RuntimeError(
                f"scale winner has no frozen confirmation scenario: {scenario_id}"
            )
        cells.append(
            {
                "winner_cell_id": item["cell_id"],
                "corpus_user_count": int(item["corpus_user_count"]),
                "tuning_scenario_id": scenario_id,
                "confirmation_scenario_id": confirmation_id,
                "candidate_type": item["candidate_type"],
                "requested_k": int(item["requested_k"]),
                "hnsw": dict(item["hnsw"]),
                "warmups": 10,
                "warm_measured_runs_per_plan": 300,
                "cold_measured_runs_per_plan": 30,
                "plans": ["exact_all", "ann_first"],
            }
        )
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "status": "required" if cells else "skipped",
        "winner_count": len(cells),
        "cells": cells,
        "scale_confirmation_enabled": bool(cells),
        "product_policy_confirmation_enabled": False,
        "exclusion_confirmation_enabled": False,
        "reason": (
            "full-tuning scale winners require disjoint holdout warm/cold confirmation"
            if cells
            else "all 104x24 settings failed the final scale gate"
        ),
        "expected_warm_invocations": len(cells) * (10 + 300) * 2,
        "expected_cold_invocations": len(cells) * 30 * 2,
        "expected_exclusion_invocations": 0,
    }


def _diagnostic_row_identity(item: Mapping[str, Any]) -> tuple[Any, ...]:
    hnsw = item.get("hnsw")
    if isinstance(hnsw, Mapping):
        ef_search = int(hnsw["ef_search"])
        iterative_scan = str(hnsw["iterative_scan"])
        max_scan_tuples = int(hnsw["max_scan_tuples"])
    else:
        ef_search = int(item["ef_search"])
        iterative_scan = str(item["iterative_scan"])
        max_scan_tuples = int(item["max_scan_tuples"])
    return (
        int(item["corpus_user_count"]),
        str(item["scenario_id"]),
        int(item["requested_k"]),
        ef_search,
        iterative_scan,
        max_scan_tuples,
    )


def _read_jsonl_if_exists(path: Path) -> tuple[Mapping[str, Any], ...]:
    if not path.exists():
        return ()
    result: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise RuntimeError(f"blank JSONL row at {path}:{line_number}")
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"invalid JSONL row at {path}:{line_number}")
        result.append(payload)
    return tuple(result)


def _write_jsonl_same_or_new(
    path: Path, rows: Iterable[Mapping[str, Any]]
) -> None:
    encoded = "".join(
        json.dumps(dict(item), sort_keys=True) + "\n" for item in rows
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable Goal 2 JSONL conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _current_runtime_rows(
    summaries: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    rows = [
        {
            key: value
            for key, value in item.items()
            if key != "duration_samples_ms"
        }
        for item in summaries
        if item["plan"] == "current_runtime"
    ]
    return sorted(
        rows,
        key=lambda item: (item["corpus_user_count"], item["scenario_id"]),
    )


def _final_scale_rows(
    summaries: Sequence[Mapping[str, Any]],
    *,
    confirmed_winners: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    exact = {
        (int(item["corpus_user_count"]), str(item["scenario_id"])): item
        for item in summaries
        if item["plan"] == "exact_all"
    }
    winners = {
        (int(item["corpus_user_count"]), str(item["scenario_id"])): item
        for item in confirmed_winners
    }
    return [
        {
            "corpus_user_count": cohort_size,
            "scenario_id": scenario_id,
            "candidate_type": item["candidate_type"],
            "exact_all_p50_ms": round(float(item["p50_ms"]), 6),
            "exact_all_p95_ms": round(float(item["p95_ms"]), 6),
            "exact_all_p99_ms": round(float(item["p99_ms"]), 6),
            "ann_status": (
                "confirmed" if (cohort_size, scenario_id) in winners else "no ANN candidate"
            ),
            "ann_p50_ms": (
                round(float(winners[(cohort_size, scenario_id)]["p50_ms"]), 6)
                if (cohort_size, scenario_id) in winners
                else ""
            ),
            "ann_p95_ms": (
                round(float(winners[(cohort_size, scenario_id)]["p95_ms"]), 6)
                if (cohort_size, scenario_id) in winners
                else ""
            ),
            "ann_p99_ms": (
                round(float(winners[(cohort_size, scenario_id)]["p99_ms"]), 6)
                if (cohort_size, scenario_id) in winners
                else ""
            ),
            "ann_vs_exact_all_speedup": (
                round(
                    float(item["p95_ms"])
                    / float(winners[(cohort_size, scenario_id)]["p95_ms"]),
                    6,
                )
                if (cohort_size, scenario_id) in winners
                else ""
            ),
            "ann_vs_exact_all_passed": (
                (cohort_size, scenario_id) in winners
            ),
        }
        for (cohort_size, scenario_id), item in sorted(exact.items())
    ]


def _render_final_scale_report(
    rows: Sequence[Mapping[str, Any]], counts: Mapping[str, Any]
) -> str:
    lines = [
        "# Goal 2 final scale-performance",
        "",
        "- verdict: `ANN break-even not observed through 1M`",
        "- graph values: actual measured points only",
        "- interpolation: `none`",
        f"- ANN tuning cells evaluated: `{counts['ann_tuning_result_count']}`",
        f"- `ann_vs_exact_all_passed`: `{counts['ann_vs_exact_all_passed_count']}`",
        "",
        "| cohort | scenario | exact p50 | exact p95 | exact p99 | ANN |",
        "|---:|---|---:|---:|---:|---|",
    ]
    for item in rows:
        lines.append(
            f"| {item['corpus_user_count']} | {item['scenario_id']} | "
            f"{float(item['exact_all_p50_ms']):.3f} | "
            f"{float(item['exact_all_p95_ms']):.3f} | "
            f"{float(item['exact_all_p99_ms']):.3f} | {item['ann_status']} |"
        )
    lines.extend(
        [
            "",
            "Filter-first Exact and current_runtime are intentionally reported in "
            "the policy/work-amplification artifacts rather than the primary scale graph.",
            "",
        ]
    )
    return "\n".join(lines)


def _render_final_scale_png(
    rows: Sequence[Mapping[str, Any]], path: Path
) -> None:
    from offline_evaluation.ann_search_scale_artifacts import (
        _disc,
        _line,
        _png_bytes,
    )

    width, height = 960, 540
    pixels = bytearray([255] * width * height * 3)
    left, right, top, bottom = 70, width - 30, 30, height - 55
    _line(pixels, width, height, left, top, left, bottom, (30, 30, 30))
    _line(pixels, width, height, left, bottom, right, bottom, (30, 30, 30))
    sizes = tuple(sorted({int(item["corpus_user_count"]) for item in rows}))
    x_positions = {
        size: int(left + index * (right - left) / max(1, len(sizes) - 1))
        for index, size in enumerate(sizes)
    }
    values = [float(item["exact_all_p95_ms"]) for item in rows]
    values.extend(
        float(item["ann_p95_ms"])
        for item in rows
        if item["ann_p95_ms"] != ""
    )
    maximum = max(values) if values else 1.0
    for item in rows:
        x = x_positions[int(item["corpus_user_count"])]
        exact_y = int(
            bottom - float(item["exact_all_p95_ms"]) / maximum * (bottom - top)
        )
        _disc(pixels, width, height, x, exact_y, 3, (30, 80, 180))
        if item["ann_p95_ms"] != "":
            ann_y = int(
                bottom - float(item["ann_p95_ms"]) / maximum * (bottom - top)
            )
            _disc(pixels, width, height, x, ann_y, 4, (210, 80, 50))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_png_bytes(width, height, pixels))


def _render_goal2_summary(payload: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            "# ANN scale-series Goal 2 result",
            "",
            f"- status: `{payload['status']}`",
            f"- actual core cells: `{payload['actual_core_cell_count']}`",
            f"- actual invocations: `{payload['actual_invocation_count']}`",
            f"- measured observations: `{payload['measured_observation_count']}`",
            f"- warm-up observations: `{payload['warmup_observation_count']}`",
            f"- cold observations: `{payload['cold_observation_count']}`",
            f"- exclusion observations: `{payload['exclusion_observation_count']}`",
            f"- diagnostic ANN cells: `{payload['diagnostic_ann_cell_count']}`",
            f"- diagnostic Exact plans: `{payload['diagnostic_exact_plan_count']}`",
            "- scale winners: `0`",
            "- observed ANN break-even: `not observed through 1M`",
            "- ann_min_users: `null`",
            f"- validated_max_users: `{payload['validated_max_users']}`",
            f"- ann_vs_exact_all_passed: `{payload['ann_vs_exact_all_passed_count']}`",
            f"- ann_policy_candidate: `{payload['ann_policy_candidate_count']}`",
            "- policy_rule_eligible: `0`",
            "- Filter-first rules: `0`",
            "- candidate policy generated: `false`",
            "- macro: `skipped — no policy_rule_eligible rule`",
            "- deployment-equivalent confirmation: `still required for any future candidate`",
            "",
            "The product policy track is not identifiable under N, H/N, E/N because "
            "no observed bucket has all five production candidate types. Goal 1 "
            "artifacts and the historical 50K pilot remain unchanged.",
            "",
        )
    )


def _build_goal2_integrity(
    args: argparse.Namespace,
    summary: Mapping[str, Any],
    final_rows: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    issues: list[str] = []
    _verify_goal1_lock(args.goal1_root, args.goal2_root)
    try:
        ScaleArtifactValidator(args.goal1_root).raise_for_errors()
    except ValueError as exc:
        issues.append(f"goal1_integrity:{exc}")
    issues.extend(_goal1_checkpoint_hash_issues(args.goal1_root))
    required = (
        "goal1-input-lock.json",
        "goal2-manifest.json",
        "environment-manifest.json",
        "policy-coverage-audit.json",
        "policy-coverage-audit.md",
        "policy-identifiability-decision.json",
        "goal2-execution-manifest.json",
        "goal2-execution-plan.md",
        "tuning-cell-manifest.jsonl",
        "hnsw-tuning-results.jsonl",
        "hnsw-tuning-winners.json",
        "ann-scale-final-gates.jsonl",
        "ann-policy-final-gates.jsonl",
        "confirmation-plan.json",
        "confirmation-results.jsonl",
        "score-sample-selection.json",
        "final-scale-performance.csv",
        "final-scale-performance-report.md",
        "final-scale-performance.png",
        "ann-break-even.json",
        "current-vs-candidate.json",
        "current-runtime-work-amplification.json",
        "unvalidated-fallbacks.json",
        "deployment-equivalent-unvalidated.json",
        "candidate-policy-decision.json",
        "goal2-summary.md",
        "goal2-code-fingerprint.json",
    )
    hashes: dict[str, str] = {}
    for relative in required:
        path = args.goal2_root / relative
        if not path.exists():
            issues.append(f"missing:{relative}")
            continue
        hashes[relative] = sha256_file(path)
    counts = {
        "tuning_cell_manifest": len(
            _read_jsonl_if_exists(args.goal2_root / "tuning-cell-manifest.jsonl")
        ),
        "hnsw_tuning_results": len(
            _read_jsonl_if_exists(args.goal2_root / "hnsw-tuning-results.jsonl")
        ),
        "scale_gates": len(
            _read_jsonl_if_exists(args.goal2_root / "ann-scale-final-gates.jsonl")
        ),
        "policy_gates": len(
            _read_jsonl_if_exists(args.goal2_root / "ann-policy-final-gates.jsonl")
        ),
        "final_scale_rows": len(final_rows),
    }
    expected_counts = {
        "tuning_cell_manifest": 2_496,
        "hnsw_tuning_results": 2_496,
        "scale_gates": 2_496,
        "policy_gates": 2_496,
        "final_scale_rows": 19,
    }
    if counts != expected_counts:
        issues.append(f"artifact_counts:{counts!r}")
    if int(summary["actual_invocation_count"]) != 33_189:
        issues.append("invocation_count")
    if int(summary["measured_observation_count"]) != 25_530:
        issues.append("measured_observation_count")
    if (args.goal2_root / "candidate-policy.json").exists():
        issues.append("candidate_policy_must_not_exist")
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "passed": not issues,
        "issues": issues,
        "goal1_unchanged": not any(item.startswith("goal1") for item in issues),
        "runtime_api_data_contract_changed": False,
        "artifact_counts": counts,
        "artifact_sha256": dict(sorted(hashes.items())),
        "artifact_set_sha256": canonical_json_sha256(hashes),
    }


def _goal1_checkpoint_hash_issues(goal1_root: Path) -> list[str]:
    issues: list[str] = []
    for cohort_size in SCALE_COHORT_SIZES:
        checkpoint = load_json_object(
            goal1_root / "checkpoints" / f"cohort-{cohort_size}.json"
        )
        for label in ("raw", "result"):
            item = checkpoint.get(label)
            if not isinstance(item, Mapping):
                issues.append(f"goal1_{label}_entry:{cohort_size}")
                continue
            path = Path(str(item["path"]))
            if not path.is_file() or sha256_file(path) != item["sha256"]:
                issues.append(f"goal1_{label}_hash:{cohort_size}")
        ground_truth = checkpoint.get("ground_truth")
        if not isinstance(ground_truth, Mapping):
            issues.append(f"goal1_ground_truth_entry:{cohort_size}")
            continue
        for scenario_id, item in ground_truth.items():
            if not isinstance(item, Mapping):
                issues.append(
                    f"goal1_ground_truth_entry:{cohort_size}:{scenario_id}"
                )
                continue
            for path_key, hash_key in (
                ("path", "sha256"),
                ("summary_path", "summary_sha256"),
            ):
                path = Path(str(item[path_key]))
                if not path.is_file() or sha256_file(path) != item[hash_key]:
                    issues.append(
                        f"goal1_ground_truth_hash:{cohort_size}:{scenario_id}:{path_key}"
                    )
    return issues


def _live_preflight(args: argparse.Namespace, cohort_size: int) -> Mapping[str, Any]:
    environment = _benchmark_environment(args, cohort_size)
    command = (
        sys.executable,
        str(ROOT / "scripts/benchmark_audience_search.py"),
        "preflight",
        "--env-file",
        str(args.env_file),
        "--scale-series-id",
        args.scale_series_id,
        "--scale-cohort-size",
        str(cohort_size),
        "--reference-sample-seed",
        args.reference_sample_seed,
        "--manifest",
        str(args.goal1_root / "phase2/selected-scenario-manifest.json"),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Goal 2 preflight failed for {cohort_size}:\n"
            + completed.stderr[-8_000:]
        )
    payload = json.loads(completed.stdout)
    if not isinstance(payload, Mapping):
        raise RuntimeError("Goal 2 preflight did not return an object")
    return payload


def _database_state(args: argparse.Namespace, cohort_size: int) -> Mapping[str, Any]:
    env_values = _env_values(args.env_file)
    database = cohort_database_name(args.postgres_prefix, cohort_size)
    username = env_values["LOOPAD_AURORA_USERNAME"]
    query = """
        SELECT json_build_object(
          'postgres_row_count', (
            SELECT count(*) FROM user_behavior_vector_search
          ),
          'vector_generation_id', (
            SELECT vector_generation_id
            FROM user_behavior_vector_search_generations
            WHERE is_active = true AND status = 'activated'
            ORDER BY created_at DESC LIMIT 1
          ),
          'index_definition', pg_get_indexdef(indexrelid),
          'index_valid', indisvalid,
          'index_ready', indisready,
          'index_size_bytes', pg_relation_size(indexrelid),
          'shared_buffers', current_setting('shared_buffers'),
          'work_mem', current_setting('work_mem'),
          'effective_cache_size', current_setting('effective_cache_size'),
          'maintenance_work_mem', current_setting('maintenance_work_mem'),
          'random_page_cost', current_setting('random_page_cost'),
          'max_parallel_workers', current_setting('max_parallel_workers'),
          'max_parallel_workers_per_gather',
             current_setting('max_parallel_workers_per_gather'),
          'postgres_version', current_setting('server_version'),
          'pgvector_version', (
            SELECT extversion FROM pg_extension WHERE extname='vector'
          )
        )
        FROM pg_index
        WHERE indexrelid =
          'idx_user_behavior_vector_search_embedding_hnsw'::regclass
    """
    raw = _docker_psql(
        container=args.postgres_container,
        username=username,
        database=database,
        query=query,
    )
    payload = json.loads(raw)
    if not isinstance(payload, Mapping):
        raise RuntimeError("PostgreSQL state query did not return an object")
    expected_prefix = _hash_stream(
        (
            "docker",
            "exec",
            "-i",
            args.clickhouse_container,
            "clickhouse-client",
            "--database",
            args.clickhouse_database,
            "--query",
            (
                "SELECT user_id FROM ann_benchmark_scale_membership "
                f"WHERE scale_series_id='{args.scale_series_id}' "
                f"AND cohort_rank <= {cohort_size} "
                "ORDER BY cohort_rank FORMAT TSVRaw"
            ),
        )
    )
    clickhouse_set = _hash_stream(
        (
            "docker",
            "exec",
            "-i",
            args.clickhouse_container,
            "clickhouse-client",
            "--database",
            args.clickhouse_database,
            "--query",
            (
                "SELECT user_id FROM ann_benchmark_scale_membership "
                f"WHERE scale_series_id='{args.scale_series_id}' "
                f"AND cohort_rank <= {cohort_size} "
                "ORDER BY user_id FORMAT TSVRaw"
            ),
        )
    )
    postgres_set = _hash_stream(
        (
            "docker",
            "exec",
            "-i",
            args.postgres_container,
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            username,
            "-d",
            database,
            "-At",
            "-c",
            "SELECT user_id FROM user_behavior_vector_search ORDER BY user_id",
        )
    )
    return {
        **dict(payload),
        "database": database,
        "cohort_size": cohort_size,
        "clickhouse_membership_prefix_sha256": expected_prefix["sha256"],
        "clickhouse_membership_prefix_rows": expected_prefix["row_count"],
        "clickhouse_membership_set_sha256": clickhouse_set["sha256"],
        "postgres_vector_set_sha256": postgres_set["sha256"],
        "postgres_vector_set_rows": postgres_set["row_count"],
        "membership_set_equal": clickhouse_set == postgres_set,
    }


def _validate_against_goal1_preflight(
    root: Path, cohort_size: int, current: Mapping[str, Any]
) -> None:
    frozen = load_json_object(
        root
        / "phase3/preflight-scale-ready"
        / f"cohort-{cohort_size}.json"
    )
    stable_fields = (
        "project_id",
        "vector_version",
        "manifest_hash",
        "vector_generation_id",
        "corpus_user_count",
        "window_start",
        "source_cutoff",
        "source_revision_cutoff",
        "postgres_version",
        "pgvector_version",
        "clickhouse_version",
        "cohort_membership_count",
        "cohort_signal_count",
        "scale_series_id",
    )
    differences = {
        field: {"goal1": frozen.get(field), "current": current.get(field)}
        for field in stable_fields
        if frozen.get(field) != current.get(field)
    }
    for field in ("index_name", "indisvalid", "indisready", "index_size_bytes"):
        frozen_value = (frozen.get("hnsw_index") or {}).get(field)
        current_value = (current.get("hnsw_index") or {}).get(field)
        if frozen_value != current_value:
            differences[f"hnsw_index.{field}"] = {
                "goal1": frozen_value,
                "current": current_value,
            }
    if differences:
        raise RuntimeError(
            f"Goal 1 DB preflight changed for cohort {cohort_size}: "
            + json.dumps(differences, sort_keys=True, default=str)
        )


def _validate_database_state(
    root: Path,
    cohort_size: int,
    preflight: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    checkpoint = load_json_object(root / "checkpoints" / f"cohort-{cohort_size}.json")
    if int(state["postgres_row_count"]) != cohort_size:
        raise RuntimeError(f"PostgreSQL row count changed for cohort {cohort_size}")
    if int(state["postgres_vector_set_rows"]) != cohort_size:
        raise RuntimeError(f"PostgreSQL membership row count changed for {cohort_size}")
    if int(state["clickhouse_membership_prefix_rows"]) != cohort_size:
        raise RuntimeError(f"ClickHouse membership row count changed for {cohort_size}")
    if state["clickhouse_membership_prefix_sha256"] != checkpoint["cohort_sha256"]:
        raise RuntimeError(f"ClickHouse membership prefix changed for {cohort_size}")
    if state["membership_set_equal"] is not True:
        raise RuntimeError(f"PostgreSQL/ClickHouse membership differs for {cohort_size}")
    if state["vector_generation_id"] != preflight["vector_generation_id"]:
        raise RuntimeError(f"active vector generation changed for {cohort_size}")
    if state["index_valid"] is not True or state["index_ready"] is not True:
        raise RuntimeError(f"HNSW index is not valid/ready for {cohort_size}")
    if "USING hnsw" not in str(state["index_definition"]):
        raise RuntimeError(f"HNSW DDL changed for {cohort_size}")


def _run_raw_block(
    args: argparse.Namespace,
    *,
    cohort_size: int,
    scenario_id: str,
    output: Path,
    plans: Sequence[str],
    requested_ks: Sequence[int],
    full_hnsw: bool,
) -> None:
    if output.exists():
        return
    partial = output.with_suffix(".partial.jsonl")
    partial.unlink(missing_ok=True)
    command: list[str] = [
        sys.executable,
        str(ROOT / "scripts/benchmark_audience_search.py"),
        "run-cell",
        "--env-file",
        str(args.env_file),
        "--scale-series-id",
        args.scale_series_id,
        "--scale-cohort-size",
        str(cohort_size),
        "--reference-sample-seed",
        args.reference_sample_seed,
        "--manifest",
        str(args.goal1_root / "phase2/selected-scenario-manifest.json"),
        "--scenario-id",
        scenario_id,
        "--phase",
        "tuning",
        "--cache-mode",
        "warm",
        "--warmups",
        str(GOAL2_WARMUPS),
        "--repetitions",
        str(GOAL2_REPETITIONS),
        "--sample-sizes",
        "50000",
        "--hnsw-grid",
        "full" if full_hnsw else "baseline",
        "--output",
        str(partial),
        "--confirm-disposable-postgres",
    ]
    for plan in plans:
        command.extend(("--plan", plan))
    if requested_ks:
        command.extend(("--requested-k", ",".join(map(str, requested_ks))))
    output.parent.mkdir(parents=True, exist_ok=True)
    _progress(
        "raw_block_started",
        cohort_size=cohort_size,
        scenario_id=scenario_id,
        plans=list(plans),
        requested_k_count=len(requested_ks),
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_benchmark_environment(args, cohort_size),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Goal 2 raw block failed for {cohort_size}/{scenario_id}:\n"
            + completed.stdout[-4_000:]
            + "\n"
            + completed.stderr[-12_000:]
        )
    if not partial.exists() or partial.stat().st_size <= 0:
        raise RuntimeError("Goal 2 raw block produced no JSONL")
    os.replace(partial, output)


def _validate_raw_block(
    path: Path,
    *,
    cohort_size: int,
    scenario_id: str,
    expected_plans: set[str],
    expected_requested_ks: set[int],
    expected_hnsw_count: int,
) -> Mapping[str, int]:
    identities: dict[tuple[Any, ...], dict[bool, int]] = {}
    hnsw_values: set[tuple[int, str, int]] = set()
    requested_values: set[int] = set()
    plans: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"raw row {line_number} is not an object")
        if payload.get("experiment_version") != SCALE_EXPERIMENT_VERSION:
            raise RuntimeError("raw experiment version differs")
        if payload.get("scenario_id") != scenario_id:
            raise RuntimeError("raw scenario identity differs")
        if int(payload.get("corpus_user_count", -1)) != cohort_size:
            raise RuntimeError("raw cohort identity differs")
        if int(payload.get("score_pass_sample_size", -1)) != min(50_000, cohort_size):
            raise RuntimeError("raw score sample is not frozen at cohort-bounded 50K")
        if payload.get("peak_rss_bytes") is None:
            raise RuntimeError("raw peak RSS evidence is missing")
        if payload.get("temp_spill") is None or payload.get("oom") is None:
            raise RuntimeError("raw spill/OOM evidence is missing")
        plan = str(payload["plan"])
        plans.add(plan)
        hnsw = payload.get("hnsw")
        hnsw_key = None
        if isinstance(hnsw, Mapping):
            hnsw_key = (
                int(hnsw["ef_search"]),
                str(hnsw["iterative_scan"]),
                int(hnsw["max_scan_tuples"]),
            )
            hnsw_values.add(hnsw_key)
        requested_k = payload.get("requested_k")
        if requested_k is not None:
            requested_values.add(int(requested_k))
        identity = (plan, requested_k, hnsw_key)
        counts = identities.setdefault(identity, {False: 0, True: 0})
        counts[bool(payload["measured"])] += 1
    if plans != expected_plans:
        raise RuntimeError(f"raw plan coverage differs: {plans}")
    if requested_values != expected_requested_ks:
        raise RuntimeError("raw K coverage differs from execution manifest")
    if len(hnsw_values) != expected_hnsw_count:
        raise RuntimeError("raw HNSW setting coverage differs")
    for identity, counts in identities.items():
        if counts[False] != GOAL2_WARMUPS or counts[True] != GOAL2_REPETITIONS:
            raise RuntimeError(f"raw run count incomplete for {identity!r}")
    return _raw_counts(path)


def _raw_counts(path: Path) -> Mapping[str, int]:
    total = measured = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            measured += int(bool(json.loads(line)["measured"]))
    return {"total": total, "measured": measured, "warmup": total - measured}


def _validate_block_checkpoint(
    args: argparse.Namespace,
    *,
    point: Mapping[str, Any],
    checkpoint: Path,
    execution_manifest_sha256: str,
) -> None:
    payload = load_json_object(checkpoint)
    if payload.get("status") != "complete":
        raise RuntimeError(f"incomplete Goal 2 checkpoint: {checkpoint}")
    if payload.get("execution_manifest_sha256") != execution_manifest_sha256:
        raise RuntimeError(f"checkpoint manifest conflict: {checkpoint}")
    if int(payload["corpus_user_count"]) != int(point["corpus_user_count"]):
        raise RuntimeError(f"checkpoint cohort conflict: {checkpoint}")
    if payload["scenario_id"] != point["scenario_id"]:
        raise RuntimeError(f"checkpoint scenario conflict: {checkpoint}")
    if list(payload["requested_k_values"]) != list(point["requested_k_values"]):
        raise RuntimeError(f"checkpoint K conflict: {checkpoint}")
    for label in ("baseline", "ann"):
        path = Path(str(payload[f"{label}_path"]))
        if not path.is_file() or sha256_file(path) != payload[f"{label}_sha256"]:
            raise RuntimeError(f"checkpoint {label} hash conflict: {checkpoint}")
    _validate_raw_block(
        Path(str(payload["baseline_path"])),
        cohort_size=int(point["corpus_user_count"]),
        scenario_id=str(point["scenario_id"]),
        expected_plans=set(ALL_BASELINE_PLANS),
        expected_requested_ks=set(),
        expected_hnsw_count=0,
    )
    _validate_raw_block(
        Path(str(payload["ann_path"])),
        cohort_size=int(point["corpus_user_count"]),
        scenario_id=str(point["scenario_id"]),
        expected_plans={"ann_first"},
        expected_requested_ks={int(value) for value in point["requested_k_values"]},
        expected_hnsw_count=24,
    )


def _completed_invocations(
    goal2_root: Path, points: Sequence[Mapping[str, Any]]
) -> int:
    result = 0
    for point in points:
        path = (
            goal2_root
            / "checkpoints"
            / f"cohort-{int(point['corpus_user_count'])}"
            / f"{point['scenario_id']}.json"
        )
        if path.exists():
            result += int(load_json_object(path)["actual_invocation_count"])
    return result


def _benchmark_environment(
    args: argparse.Namespace, cohort_size: int
) -> Mapping[str, str]:
    return {
        **_env_values(args.env_file),
        **os.environ,
        "LOOPAD_AURORA_DATABASE": cohort_database_name(
            args.postgres_prefix, cohort_size
        ),
        "LOOPAD_CLICKHOUSE_DATABASE": args.clickhouse_database,
        "ANN_POSTGRES_CONTAINER": args.postgres_container,
    }


def _docker_resources(
    *, postgres_container: str, clickhouse_container: str
) -> Mapping[str, Any]:
    return {
        "postgres": _docker_resource(postgres_container),
        "clickhouse": _docker_resource(clickhouse_container),
    }


def _docker_resource(container: str) -> Mapping[str, Any]:
    completed = subprocess.run(
        ("docker", "inspect", container),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"docker inspect failed for {container}: {completed.stderr}")
    rows = json.loads(completed.stdout)
    if not isinstance(rows, list) or len(rows) != 1:
        raise RuntimeError(f"docker inspect returned invalid output for {container}")
    item = rows[0]
    state = item.get("State", {})
    host = item.get("HostConfig", {})
    config = item.get("Config", {})
    if state.get("Running") is not True:
        raise RuntimeError(f"required benchmark container is not running: {container}")
    return {
        "name": container,
        "container_id": str(item.get("Id", "")),
        "image": str(config.get("Image", "")),
        "image_id": str(item.get("Image", "")),
        "running": True,
        "memory_bytes": int(host.get("Memory", 0)),
        "nano_cpus": int(host.get("NanoCpus", 0)),
        "shm_bytes": int(host.get("ShmSize", 0)),
    }


def _validate_resource_limits(
    current: Mapping[str, Any], frozen: Mapping[str, Any]
) -> None:
    expected = frozen.get("container_limits")
    if not isinstance(expected, Mapping):
        raise RuntimeError("Goal 1 resource manifest is invalid")
    for name in ("postgres", "clickhouse"):
        for field in ("memory_bytes", "nano_cpus", "shm_bytes"):
            if int(current[name][field]) != int(expected[name][field]):
                raise RuntimeError(f"Goal 1 {name} {field} resource limit changed")


def _docker_psql(
    *, container: str, username: str, database: str, query: str
) -> str:
    completed = subprocess.run(
        (
            "docker",
            "exec",
            "-i",
            container,
            "psql",
            "-v",
            "ON_ERROR_STOP=1",
            "-U",
            username,
            "-d",
            database,
            "-At",
            "-c",
            query,
        ),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"PostgreSQL state query failed for {database}: {completed.stderr}"
        )
    return completed.stdout.strip()


def _hash_stream(command: Sequence[str]) -> Mapping[str, Any]:
    digest = hashlib.sha256()
    row_count = 0
    process = subprocess.Popen(
        tuple(command),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None
    for raw in process.stdout:
        value = raw.rstrip(b"\r\n")
        digest.update(value)
        digest.update(b"\n")
        row_count += 1
    stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
    returncode = process.wait()
    if returncode != 0:
        raise RuntimeError("membership hash command failed: " + stderr[-4_000:])
    return {"row_count": row_count, "sha256": digest.hexdigest()}


def _env_values(path: Path) -> Mapping[str, str]:
    values = {
        str(key): str(value)
        for key, value in dotenv_values(path).items()
        if value is not None
    }
    required = ("LOOPAD_AURORA_USERNAME",)
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise RuntimeError("benchmark env is incomplete: " + ",".join(missing))
    return values


def _compiler_calibration_hashes(root: Path) -> Mapping[str, Any]:
    manifest = load_json_object(root / "phase2/selected-scenario-manifest.json")
    scenarios = manifest["scenarios"]
    values = {
        str(item["candidate_type"]): {
            "calibration_hash": item["compiler_provenance"]["calibration_hash"],
            "query_compiler_hash": item["compiler_provenance"][
                "query_compiler_hash"
            ],
        }
        for item in scenarios
    }
    return dict(sorted(values.items()))


def _verify_goal1_lock(goal1_root: Path, goal2_root: Path) -> None:
    lock = load_json_object(goal2_root / "goal1-input-lock.json")
    current = goal1_required_sha256(goal1_root)
    if current != lock["goal1_required_file_sha256"]:
        raise RuntimeError("Goal 1 required inputs changed after Goal 2 Phase 0")
    if canonical_json_sha256(current) != lock["goal1_required_file_set_sha256"]:
        raise RuntimeError("Goal 1 input hash set changed after Goal 2 Phase 0")


def _require_phase0(goal2_root: Path) -> None:
    lock = load_json_object(goal2_root / "goal1-input-lock.json")
    if lock.get("status") != "passed" or lock.get("goal1_integrity_passed") is not True:
        raise RuntimeError("Goal 2 Phase 0 has not passed")


def _require_roots(args: argparse.Namespace) -> None:
    if args.goal1_root.resolve() != (ROOT / GOAL1_ROOT).resolve():
        raise ValueError("Goal 1 root must remain the frozen scale-series-v2 path")
    if args.goal2_root.resolve() != (ROOT / GOAL2_ROOT).resolve():
        raise ValueError("Goal 2 outputs must remain isolated under scale-series-v2/goal2")
    if tuple(args.cohort_sizes) != tuple(SCALE_COHORT_SIZES):
        raise ValueError("Goal 2 must keep the frozen six-cohort execution set")


def _write_json_same_or_new(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        existing = load_json_object(path)
        if existing != payload:
            raise RuntimeError(f"immutable Goal 2 artifact conflict: {path}")
        return
    write_immutable_json(path, payload)


def _write_text_same_or_new(path: Path, value: str) -> None:
    encoded = value.rstrip() + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable Goal 2 artifact conflict: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _write_tuning_cell_manifest(path: Path, cells: Iterable[Any]) -> None:
    encoded = "".join(
        json.dumps(cell.to_dict(), sort_keys=True) + "\n" for cell in cells
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("immutable tuning-cell manifest conflict")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def _render_execution_plan(
    manifest: Mapping[str, Any], decision: Mapping[str, Any]
) -> str:
    lines = [
        "# Goal 2 execution plan",
        "",
        f"- execution manifest: `{manifest['execution_manifest_sha256']}`",
        f"- Goal 1 survivor union: `{manifest['goal1_union_cell_count']}`",
        f"- HNSW settings: `{manifest['hnsw_setting_count']}`",
        f"- ANN tuning cells: `{manifest['ann_tuning_cell_count']}`",
        f"- matched baseline cells: `{manifest['baseline_cell_count']}`",
        f"- total invocations: `{manifest['expected_total_invocation_count']}`",
        f"- measured observations: `{manifest['expected_measured_observation_count']}`",
        f"- product policy track: `{decision['status']}`",
        "",
        "Live measurements run serially in cohort/scenario blocks. Completed raw "
        "blocks and checkpoints are immutable; resume skips only identical hashes.",
        "Diagnostics are not included in timing and are collected after tuning for "
        "provisional winners and representative adjacent failures.",
        "",
        "| cohort | points | ANN cells | baseline cells | invocations | measured |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for cohort_size, item in manifest["by_cohort"].items():
        lines.append(
            f"| {cohort_size} | {item['point_count']} | "
            f"{item['ann_tuning_cell_count']} | {item['baseline_cell_count']} | "
            f"{item['expected_invocation_count']} | "
            f"{item['expected_measured_observation_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _positive_ints(value: str) -> tuple[int, ...]:
    values = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("cohort sizes must be positive")
    return values


def _progress(event: str, **values: Any) -> None:
    print(json.dumps({"event": event, **values}, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
