#!/usr/bin/env python3
"""Reanalyze the existing 750K/1M ANN DB-cold measurements.

This tool does not run a benchmark. It reads the immutable Goal 3 artifacts,
strictly selects one fixed HNSW cell, validates its 60 stored observations,
cross-checks the published summaries, and writes small derived artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_experiment import percentile  # noqa: E402


DEFAULT_CHECKPOINT_SUMMARY = (
    ROOT
    / "performance-tests"
    / "ann-search"
    / "evidence"
    / "experiments"
    / "ann-efficiency-recovery-v1"
    / "summary.json"
)
INPUT_FILES = (
    "kernel-cold-raw.jsonl",
    "kernel-cold-confirmation.jsonl",
    "kernel-confirmation.jsonl",
    "candidate-retrieval-final-confirmation.jsonl",
    "conclusion.json",
)
TARGET_COHORTS = (750_000, 1_000_000)
EXPECTED_RUNS_PER_COHORT = 30
TARGET_SELECTION: Mapping[str, object] = {
    "candidate_type": "funnel_recovery",
    "requested_k": 500,
    "ef_search": 200,
    "iterative_scan": "relaxed_order",
    "max_scan_tuples": 20_000,
    "corpus_user_count": list(TARGET_COHORTS),
    "phase": "confirmation_cold",
    "measured": True,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from error
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(row)
    return rows


def is_target_row(row: Mapping[str, Any]) -> bool:
    hnsw = row.get("hnsw")
    return bool(
        isinstance(hnsw, Mapping)
        and row.get("candidate_type") == TARGET_SELECTION["candidate_type"]
        and row.get("requested_k") == TARGET_SELECTION["requested_k"]
        and hnsw.get("ef_search") == TARGET_SELECTION["ef_search"]
        and hnsw.get("iterative_scan") == TARGET_SELECTION["iterative_scan"]
        and hnsw.get("max_scan_tuples") == TARGET_SELECTION["max_scan_tuples"]
        and row.get("corpus_user_count") in TARGET_COHORTS
        and row.get("phase") == TARGET_SELECTION["phase"]
        and row.get("measured") is True
    )


def select_and_validate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_runs_per_cohort: int = EXPECTED_RUNS_PER_COHORT,
) -> list[dict[str, Any]]:
    selected = [dict(row) for row in rows if is_target_row(row)]
    expected_iterations = set(range(expected_runs_per_cohort))
    all_measurement_ids: set[str] = set()
    identities: dict[int, tuple[str, str]] = {}

    for cohort in TARGET_COHORTS:
        cohort_rows = [row for row in selected if row["corpus_user_count"] == cohort]
        if len(cohort_rows) != expected_runs_per_cohort:
            raise ValueError(
                f"{cohort}: expected {expected_runs_per_cohort} target rows, "
                f"found {len(cohort_rows)}"
            )

        iterations = [row.get("iteration") for row in cohort_rows]
        if len(set(iterations)) != len(iterations):
            raise ValueError(f"{cohort}: duplicate iteration")
        if set(iterations) != expected_iterations:
            raise ValueError(
                f"{cohort}: iterations must be 0..{expected_runs_per_cohort - 1}"
            )

        query_pairs = {
            (str(row.get("query_id")), str(row.get("scenario_id")))
            for row in cohort_rows
        }
        if len(query_pairs) != 1:
            raise ValueError(f"{cohort}: mixed query_id/scenario_id values")
        identities[cohort] = next(iter(query_pairs))

        for row in cohort_rows:
            measurement_id = row.get("measurement_id")
            if not isinstance(measurement_id, str) or not measurement_id:
                raise ValueError(f"{cohort}: missing measurement_id")
            if measurement_id in all_measurement_ids:
                raise ValueError(f"duplicate measurement_id: {measurement_id}")
            all_measurement_ids.add(measurement_id)

            for field in ("exact_duration_ms", "ann_duration_ms"):
                value = row.get(field)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) <= 0
                ):
                    raise ValueError(
                        f"{cohort} iteration {row['iteration']}: invalid {field}"
                    )

            if row.get("hnsw_index_used") is not True:
                raise ValueError(f"{cohort} iteration {row['iteration']}: HNSW index not used")
            if row.get("exact_uses_hnsw") is not False:
                raise ValueError(f"{cohort} iteration {row['iteration']}: Exact used HNSW")
            if row.get("temp_spill") is not False or row.get("oom") is not False:
                raise ValueError(
                    f"{cohort} iteration {row['iteration']}: spill or OOM flag set"
                )

    if len(set(identities.values())) != 1:
        raise ValueError("750K and 1M do not use the same confirmation query")

    return sorted(selected, key=lambda row: (row["corpus_user_count"], row["iteration"]))


def latency_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    exact = [float(row["exact_duration_ms"]) for row in rows]
    ann = [float(row["ann_duration_ms"]) for row in rows]
    exact_p95 = percentile(exact, 0.95)
    ann_p95 = percentile(ann, 0.95)
    return {
        "run_count": len(rows),
        "exact_median_ms": percentile(exact, 0.50),
        "exact_p95_ms": exact_p95,
        "exact_max_ms": max(exact),
        "ann_median_ms": percentile(ann, 0.50),
        "ann_p95_ms": ann_p95,
        "ann_max_ms": max(ann),
        "ann_to_exact_p95_ratio": ann_p95 / exact_p95,
        "ann_exceeds_cohort_exact_p95_count": sum(value > exact_p95 for value in ann),
        "ann_slower_than_paired_exact_count": sum(
            float(row["ann_duration_ms"]) > float(row["exact_duration_ms"])
            for row in rows
        ),
    }


def _same_float(actual: Any, expected: float, *, tolerance: float = 1e-9) -> bool:
    return isinstance(actual, (int, float)) and math.isclose(
        float(actual), expected, rel_tol=1e-12, abs_tol=tolerance
    )


def _target_summary_rows(path: Path, *, phase: str) -> dict[int, dict[str, Any]]:
    matches: dict[int, dict[str, Any]] = {}
    for row in load_jsonl(path):
        hnsw = row.get("hnsw")
        if not isinstance(hnsw, Mapping):
            continue
        if (
            row.get("candidate_type") == "funnel_recovery"
            and row.get("requested_k") == 500
            and hnsw.get("ef_search") == 200
            and hnsw.get("iterative_scan") == "relaxed_order"
            and hnsw.get("max_scan_tuples") == 20_000
            and row.get("corpus_user_count") in TARGET_COHORTS
            and row.get("phase") == phase
        ):
            cohort = int(row["corpus_user_count"])
            if cohort in matches:
                raise ValueError(f"{path}: duplicate target summary for {cohort}")
            matches[cohort] = row
    if set(matches) != set(TARGET_COHORTS):
        raise ValueError(f"{path}: missing target summary; found {sorted(matches)}")
    return matches


def cross_check_existing_artifacts(
    *,
    input_dir: Path,
    checkpoint_summary_path: Path,
    selected: Sequence[Mapping[str, Any]],
    summaries: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    cold = _target_summary_rows(
        input_dir / "kernel-cold-confirmation.jsonl", phase="confirmation_cold"
    )
    warm = _target_summary_rows(
        input_dir / "kernel-confirmation.jsonl", phase="confirmation_warm"
    )
    final = _target_summary_rows(
        input_dir / "candidate-retrieval-final-confirmation.jsonl",
        phase="confirmation_warm",
    )

    for cohort in TARGET_COHORTS:
        cohort_rows = [row for row in selected if row["corpus_user_count"] == cohort]
        exact_samples = [float(row["exact_duration_ms"]) for row in cohort_rows]
        ann_samples = [float(row["ann_duration_ms"]) for row in cohort_rows]
        existing = cold[cohort]
        derived = summaries[cohort]
        comparisons = {
            "run_count": int(existing.get("run_count", -1)) == len(cohort_rows),
            "exact_p50_ms": _same_float(existing.get("exact_p50_ms"), derived["exact_median_ms"]),
            "exact_p95_ms": _same_float(existing.get("exact_p95_ms"), derived["exact_p95_ms"]),
            "ann_p50_ms": _same_float(existing.get("ann_p50_ms"), derived["ann_median_ms"]),
            "ann_p95_ms": _same_float(existing.get("ann_p95_ms"), derived["ann_p95_ms"]),
            "ann_p95_ratio": _same_float(
                existing.get("ann_p95_ratio"), derived["ann_to_exact_p95_ratio"]
            ),
            "exact_samples_ms": existing.get("exact_samples_ms") == exact_samples,
            "ann_samples_ms": existing.get("ann_samples_ms") == ann_samples,
        }
        failed = [name for name, passed in comparisons.items() if not passed]
        if failed:
            raise ValueError(f"{cohort}: raw/cold summary mismatch: {', '.join(failed)}")

        expected_cold_status = "passed" if derived["ann_p95_ms"] <= derived["exact_p95_ms"] else "failed"
        if final[cohort].get("cold_confirmation_status") != expected_cold_status:
            raise ValueError(f"{cohort}: final cold status does not match raw p95 values")
        for field, value in (
            ("cold_exact_p95_ms", derived["exact_p95_ms"]),
            ("cold_ann_p95_ms", derived["ann_p95_ms"]),
            ("cold_ann_p95_ratio", derived["ann_to_exact_p95_ratio"]),
        ):
            if not _same_float(final[cohort].get(field), value):
                raise ValueError(f"{cohort}: final confirmation mismatch for {field}")
        if int(warm[cohort].get("run_count", -1)) != 300:
            raise ValueError(f"{cohort}: warm confirmation is not the recorded 300 runs")

    conclusion = load_json(input_dir / "conclusion.json")
    required_conclusion = {
        "ann_candidate_retrieval_passed": True,
        "ann_full_membership_passed": False,
        "deployable_common_setting": False,
        "ann_policy_candidate": False,
    }
    for field, expected in required_conclusion.items():
        if conclusion.get(field) is not expected:
            raise ValueError(f"conclusion changed: {field}")

    checkpoint = load_json(checkpoint_summary_path)
    representative = checkpoint.get("representativeResult")
    if not isinstance(representative, Mapping):
        raise ValueError("checkpoint representativeResult is invalid")
    checkpoint_warm = representative.get("warm")
    checkpoint_cold = representative.get("databaseCold")
    if not isinstance(checkpoint_warm, Mapping) or not isinstance(checkpoint_cold, Mapping):
        raise ValueError("checkpoint representative warm/cold result is invalid")
    if not _same_float(checkpoint_warm.get("exactP95Ms"), float(warm[1_000_000]["exact_p95_ms"]), tolerance=0.01):
        raise ValueError("checkpoint warm Exact p95 differs from source summary")
    if not _same_float(checkpoint_warm.get("annP95Ms"), float(warm[1_000_000]["ann_p95_ms"]), tolerance=0.01):
        raise ValueError("checkpoint warm ANN p95 differs from source summary")
    improvement = 1.0 - float(summaries[1_000_000]["ann_to_exact_p95_ratio"])
    if not _same_float(checkpoint_cold.get("p95ImprovementFraction"), improvement, tolerance=0.0005):
        raise ValueError("checkpoint DB-cold improvement differs from raw recomputation")

    return {
        "raw_matches_kernel_cold_confirmation": True,
        "raw_matches_final_confirmation": True,
        "warm_summary_matches_final_confirmation": True,
        "checkpoint_summary_matches_source": True,
        "conclusion_preserved": required_conclusion,
    }


def write_csv(
    output_path: Path,
    selected: Sequence[Mapping[str, Any]],
    summaries: Mapping[int, Mapping[str, Any]],
) -> None:
    fieldnames = (
        "corpus_user_count",
        "iteration",
        "measurement_id",
        "query_id",
        "scenario_id",
        "exact_duration_ms",
        "ann_duration_ms",
        "ann_minus_exact_ms",
        "cohort_exact_p95_ms",
        "ann_exceeds_cohort_exact_p95",
        "ann_slower_than_paired_exact",
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in selected:
            cohort = int(row["corpus_user_count"])
            exact = float(row["exact_duration_ms"])
            ann = float(row["ann_duration_ms"])
            exact_p95 = float(summaries[cohort]["exact_p95_ms"])
            writer.writerow(
                {
                    "corpus_user_count": cohort,
                    "iteration": row["iteration"],
                    "measurement_id": row["measurement_id"],
                    "query_id": row["query_id"],
                    "scenario_id": row["scenario_id"],
                    "exact_duration_ms": f"{exact:.6f}",
                    "ann_duration_ms": f"{ann:.6f}",
                    "ann_minus_exact_ms": f"{ann - exact:.6f}",
                    "cohort_exact_p95_ms": f"{exact_p95:.6f}",
                    "ann_exceeds_cohort_exact_p95": str(ann > exact_p95).lower(),
                    "ann_slower_than_paired_exact": str(ann > exact).lower(),
                }
            )


def write_plot(
    *,
    output_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    summaries: Mapping[int, Mapping[str, Any]],
) -> None:
    cache_root = Path(tempfile.gettempdir()) / "loopad-ann-matplotlib-cache"
    (cache_root / "matplotlib").mkdir(parents=True, exist_ok=True)
    (cache_root / "xdg").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    matplotlib.rcParams["svg.hashsalt"] = "ann-efficiency-recovery-cold-tail-v1"
    all_values = [
        float(row[field])
        for row in selected
        for field in ("exact_duration_ms", "ann_duration_ms")
    ]
    y_max = max(all_values) * 1.10
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), sharex=True, sharey=True)

    for ax, cohort in zip(axes, TARGET_COHORTS, strict=True):
        rows = [row for row in selected if row["corpus_user_count"] == cohort]
        iterations = [int(row["iteration"]) + 1 for row in rows]
        exact = [float(row["exact_duration_ms"]) for row in rows]
        ann = [float(row["ann_duration_ms"]) for row in rows]
        stats = summaries[cohort]

        ax.plot(iterations, exact, color="#2774AE", marker="o", markersize=4, linewidth=1.0, alpha=0.82)
        ax.plot(iterations, ann, color="#D55E00", marker="o", markersize=4, linewidth=1.0, alpha=0.88)
        ax.axhline(stats["exact_median_ms"], color="#2774AE", linestyle="--", linewidth=1.2)
        ax.axhline(stats["ann_median_ms"], color="#D55E00", linestyle="--", linewidth=1.2)
        ax.axhline(stats["exact_p95_ms"], color="#2774AE", linestyle=":", linewidth=1.8)
        ax.axhline(stats["ann_p95_ms"], color="#D55E00", linestyle=":", linewidth=1.8)

        exceed_x = [x for x, value in zip(iterations, ann, strict=True) if value > stats["exact_p95_ms"]]
        exceed_y = [value for value in ann if value > stats["exact_p95_ms"]]
        if exceed_x:
            ax.scatter(exceed_x, exceed_y, s=68, facecolors="none", edgecolors="#7A0019", linewidths=1.6, zorder=5)

        ax.set_title(f"{cohort // 1000:,}K users · DB-cold (n=30)")
        ax.set_xlabel("Stored run number")
        ax.set_xlim(0, 31)
        ax.set_ylim(0, y_max)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        ax.text(
            0.03,
            0.97,
            (
                f"ANN median {stats['ann_median_ms']:.2f} ms\n"
                f"ANN p95 {stats['ann_p95_ms']:.2f} ms\n"
                f"Exact p95 {stats['exact_p95_ms']:.2f} ms\n"
                f"ANN > Exact p95: {stats['ann_exceeds_cohort_exact_p95_count']}/30"
            ),
            transform=ax.transAxes,
            va="top",
            fontsize=9.2,
            bbox={"boxstyle": "round,pad=0.4", "facecolor": "white", "alpha": 0.9, "edgecolor": "#BBBBBB"},
        )

    axes[0].set_ylabel("Latency (ms), shared scale")
    legend = [
        Line2D([0], [0], color="#2774AE", marker="o", label="Exact observations"),
        Line2D([0], [0], color="#D55E00", marker="o", label="ANN observations"),
        Line2D([0], [0], color="#555555", linestyle="--", label="Median"),
        Line2D([0], [0], color="#555555", linestyle=":", linewidth=1.8, label="p95"),
        Line2D([0], [0], color="#7A0019", marker="o", markerfacecolor="none", linestyle="None", label="ANN above cohort Exact p95"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Existing ANN experiment: per-run DB-cold latency distribution", fontsize=14, fontweight="bold")
    fig.text(0.5, 0.925, "Fixed funnel_recovery setting · all 30 stored observations per cohort · no outlier removal", ha="center", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 0.90))
    fig.savefig(
        output_dir / "cold-tail-distribution.png",
        dpi=180,
        bbox_inches="tight",
        metadata={"Software": "matplotlib"},
    )
    fig.savefig(
        output_dir / "cold-tail-distribution.svg",
        bbox_inches="tight",
        metadata={"Date": None},
    )
    plt.close(fig)
    svg_path = output_dir / "cold-tail-distribution.svg"
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def write_markdown(output_path: Path, summaries: Mapping[int, Mapping[str, Any]]) -> None:
    s750 = summaries[750_000]
    s1m = summaries[1_000_000]
    output_path.write_text(
        f"""# DB-cold tail latency reanalysis

This is a local reanalysis of the existing measurements, not a new benchmark.
It uses only the fixed `funnel_recovery`, K=500, ef=200, relaxed-order,
20K-scan confirmation cell and retains every one of the 30 stored DB-cold runs
at both 750K and 1M users.

![Per-run DB-cold latency distribution](cold-tail-distribution.png)

| Cohort | ANN median | ANN p95 | ANN max | Exact p95 | ANN / Exact p95 | ANN above cohort Exact p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 750K | {s750['ann_median_ms']:.2f} ms | {s750['ann_p95_ms']:.2f} ms | {s750['ann_max_ms']:.2f} ms | {s750['exact_p95_ms']:.2f} ms | {s750['ann_to_exact_p95_ratio']:.3f} | {s750['ann_exceeds_cohort_exact_p95_count']}/30 |
| 1M | {s1m['ann_median_ms']:.2f} ms | {s1m['ann_p95_ms']:.2f} ms | {s1m['ann_max_ms']:.2f} ms | {s1m['exact_p95_ms']:.2f} ms | {s1m['ann_to_exact_p95_ratio']:.3f} | {s1m['ann_exceeds_cohort_exact_p95_count']}/30 |

At 750K, three stored ANN runs exceeded that cohort's Exact p95; those observed
slow runs lift ANN p95 above Exact p95 and connect directly to the DB-cold gate
failure. At 1M, no ANN run exceeded that cohort's Exact p95. This does not show
that scale itself made ANN faster, nor does it identify whether cache state,
memory, index placement, execution order, or another system effect caused the
750K slow runs.

The separately calculated same-iteration comparison is also 3/30 at 750K and
0/30 at 1M in these observations. It remains a different metric: it compares
each ANN run with its paired Exact run, whereas the count above compares every
ANN run with the cohort-level Exact p95 threshold.

The measurement-time warm gate and DB-cold gate are distinct. The warm final
gate requires quality and operational checks, ANN/Exact p95 and bootstrap upper
bounds at or below 0.70, and ANN p99 no slower than Exact p99. The DB-cold join
requires 30 runs, ANN p95 no slower than Exact p95, HNSW index use, Exact not
using HNSW, and no spill or OOM. Here, “DB-cold” means the existing runner's
PostgreSQL restart procedure; it does not claim that OS or disk caches were
cleared.

The previously reported 16.3x result is the 1M **warm** candidate-retrieval p95
speedup. The figure above instead shows the 30-run **DB-cold** distributions.
For the measured cohorts and this fixed setting, 1M was the first point to pass
both the warm and DB-cold final criteria. These 30-run p95 and maximum values do
not establish general service tail latency, and the full-membership failure and
Exact fallback policy remain unchanged.
""",
        encoding="utf-8",
    )


def analyze(*, input_dir: Path, output_dir: Path, checkpoint_summary_path: Path) -> dict[str, Any]:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    checkpoint_summary_path = checkpoint_summary_path.resolve()
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("output directory must be separate from the read-only input tree")
    for name in INPUT_FILES:
        if not (input_dir / name).is_file():
            raise ValueError(f"missing required input: {input_dir / name}")
    if not checkpoint_summary_path.is_file():
        raise ValueError(f"missing checkpoint summary: {checkpoint_summary_path}")

    selected = select_and_validate_rows(load_jsonl(input_dir / "kernel-cold-raw.jsonl"))
    summaries = {
        cohort: latency_summary(
            [row for row in selected if row["corpus_user_count"] == cohort]
        )
        for cohort in TARGET_COHORTS
    }
    cross_checks = cross_check_existing_artifacts(
        input_dir=input_dir,
        checkpoint_summary_path=checkpoint_summary_path,
        selected=selected,
        summaries=summaries,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "cold-tail-runs.csv", selected, summaries)
    write_plot(output_dir=output_dir, selected=selected, summaries=summaries)
    write_markdown(output_dir / "cold-tail-analysis.md", summaries)

    query_id = str(selected[0]["query_id"])
    scenario_id = str(selected[0]["scenario_id"])
    payload: dict[str, Any] = {
        "schema_version": 1,
        "analysis_type": "existing_measurement_reanalysis",
        "new_benchmark_executed": False,
        "description": "Per-run DB-cold distribution for the fixed 750K/1M Goal 3 confirmation cell.",
        "selection": dict(TARGET_SELECTION),
        "validated_identity": {
            "query_id": query_id,
            "scenario_id": scenario_id,
            "same_query_at_both_cohorts": True,
        },
        "calculation": {
            "percentile_function": "offline_evaluation.ann_search_experiment.percentile",
            "percentile_method": "linear interpolation at (n - 1) * quantile",
            "observations_per_cohort": EXPECTED_RUNS_PER_COHORT,
            "outlier_removal": False,
            "ann_exceeds_exact_p95_definition": "ANN duration > same cohort Exact p95",
            "paired_slower_definition": "ANN duration > Exact duration at the same iteration",
        },
        "input_files": [
            {
                "name": name,
                "bytes": (input_dir / name).stat().st_size,
                "sha256": sha256_file(input_dir / name),
            }
            for name in INPUT_FILES
        ],
        "checkpoint_summary": {
            "path": str(checkpoint_summary_path.relative_to(ROOT)),
            "sha256": sha256_file(checkpoint_summary_path),
        },
        "cross_checks": cross_checks,
        "cohorts": {str(cohort): summary for cohort, summary in summaries.items()},
        "claim_boundary": {
            "db_cold_definition": "Existing runner PostgreSQL restart procedure; OS/disk cache clearing is not claimed.",
            "cause_of_slow_runs": "Not identifiable from these stored measurements alone.",
            "general_service_tail_latency_established": False,
            "existing_full_membership_and_fallback_conclusions_changed": False,
        },
    }
    (output_dir / "cold-tail-summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path, help="read-only source artifact directory")
    parser.add_argument("--output-dir", required=True, type=Path, help="separate directory for derived artifacts")
    parser.add_argument(
        "--checkpoint-summary",
        type=Path,
        default=DEFAULT_CHECKPOINT_SUMMARY,
        help="committed checkpoint summary used for rounded-value cross-checks",
    )
    args = parser.parse_args()
    payload = analyze(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        checkpoint_summary_path=args.checkpoint_summary,
    )
    for cohort in TARGET_COHORTS:
        stats = payload["cohorts"][str(cohort)]
        print(
            f"{cohort}: ANN median={stats['ann_median_ms']:.2f} ms, "
            f"p95={stats['ann_p95_ms']:.2f} ms, max={stats['ann_max_ms']:.2f} ms, "
            f"ANN>Exact p95={stats['ann_exceeds_cohort_exact_p95_count']}/30"
        )
    print(f"wrote derived artifacts to {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
