"""Actual-value-only tables and figures for Goal 3 ANN evidence.

This module never estimates a missing cell.  Every plotted point comes from a
persisted JSONL observation/result, and a companion manifest binds each figure
to the SHA-256 hashes of its source files.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from offline_evaluation.ann_search_goal3 import (
    GOAL3_EXPERIMENT_VERSION,
    REQUIRED_K_VALUES,
    canonical_json_sha256,
    read_jsonl,
)
from offline_evaluation.ann_search_scale_artifacts import sha256_file


SCREENING_PHASE = "screening_rss_corrected"


def write_goal3_figures(root: Path) -> Mapping[str, Any]:
    """Write compact, actual-only graphs and their reproducibility manifest."""

    kernel_path = root / "kernel-results.jsonl"
    full_path = root / "full-membership-results.jsonl"
    confirmation_path = root / "full-membership-confirmation.jsonl"
    stage_path = root / "stage-cost-breakdown.json"
    kernel_rows = [
        row for row in read_jsonl(kernel_path) if row.get("phase") == SCREENING_PHASE
    ]
    full_rows = read_jsonl(full_path)
    confirmation_rows = read_jsonl(confirmation_path)
    stage_rows = _read_stage_rows(stage_path)

    csv_path = root / "candidate-retrieval-efficiency.csv"
    _write_candidate_csv(csv_path, kernel_rows)
    retrieval_graph = root / "candidate-retrieval-scale.png"
    heatmap_graph = root / "candidate-retrieval-heatmap.png"
    full_graph = root / "full-membership-scale.png"
    stage_graph = root / "stage-cost-breakdown.png"
    _candidate_retrieval_scale_graph(retrieval_graph, kernel_rows)
    _candidate_heatmap(heatmap_graph, kernel_rows)
    _full_membership_scale_graph(full_graph, full_rows, confirmation_rows)
    _stage_cost_graph(stage_graph, stage_rows)

    source_paths = [
        path
        for path in (kernel_path, full_path, confirmation_path, stage_path)
        if path.is_file()
    ]
    outputs = [csv_path, retrieval_graph, heatmap_graph, full_graph, stage_graph]
    payload = {
        "experiment_version": GOAL3_EXPERIMENT_VERSION,
        "actual_values_only": True,
        "screening_phase": SCREENING_PHASE,
        "source_sha256": {str(path.relative_to(root)): sha256_file(path) for path in source_paths},
        "output_sha256": {str(path.relative_to(root)): sha256_file(path) for path in outputs},
        "source_row_counts": {
            "candidate_retrieval": len(kernel_rows),
            "full_membership_preconfirmation": len(full_rows),
            "full_membership_confirmation": len(confirmation_rows),
            "stage_breakdown": len(stage_rows),
        },
    }
    manifest_path = root / "graph-artifacts.json"
    manifest_path.write_text(
        __import__("json").dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return payload


def validate_graph_sources(root: Path) -> Mapping[str, Any]:
    """Confirm graphs are still bound to the raw/result files that produced them."""

    import json

    path = root / "graph-artifacts.json"
    if not path.is_file():
        return {"passed": False, "reason": "graph-artifacts.json is missing"}
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected = payload.get("source_sha256", {})
    actual: dict[str, str | None] = {}
    for relative, digest in expected.items():
        source = root / relative
        actual[relative] = sha256_file(source) if source.is_file() else None
    mismatches = [
        relative for relative, digest in expected.items() if actual.get(relative) != digest
    ]
    return {
        "passed": not mismatches,
        "source_hash_mismatches": mismatches,
        "recorded_source_sha256": expected,
        "actual_source_sha256": actual,
    }


def _write_candidate_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = [
        "candidate_type",
        "query_id",
        "corpus_user_count",
        "requested_k",
        "k_over_n",
        "ef_search",
        "iterative_scan",
        "max_scan_tuples",
        "exact_p50_ms",
        "exact_p95_ms",
        "ann_p50_ms",
        "ann_p95_ms",
        "ann_p95_ratio",
        "per_query_recall_at_k",
        "aggregate_wilson_lower_bound",
        "quality_gate_passed",
        "operational_gate_passed",
        "screening_gate_passed",
        "boundary_near",
        "rss_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(
            rows,
            key=lambda item: (
                str(item["candidate_type"]),
                int(item["requested_k"]),
                int(item["corpus_user_count"]),
                canonical_json_sha256(item["hnsw"]),
            ),
        ):
            hnsw = row["hnsw"]
            assert isinstance(hnsw, Mapping)
            writer.writerow(
                {
                    "candidate_type": row["candidate_type"],
                    "query_id": row["query_id"],
                    "corpus_user_count": row["corpus_user_count"],
                    "requested_k": row["requested_k"],
                    "k_over_n": row["k_over_n"],
                    "ef_search": hnsw["ef_search"],
                    "iterative_scan": hnsw["iterative_scan"],
                    "max_scan_tuples": hnsw["max_scan_tuples"],
                    **{name: row.get(name) for name in fieldnames[8:]},
                }
            )


def _candidate_retrieval_scale_graph(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for axis, requested_k in zip(axes.flat, REQUIRED_K_VALUES, strict=False):
        subset = [row for row in rows if int(row["requested_k"]) == requested_k]
        _plot_kernel_lines(axis, subset)
        axis.set_title(f"K={requested_k:,}")
        axis.set_xscale("log")
        axis.set_xlabel("N (actual cohort)")
        axis.set_ylabel("p95 latency (ms)")
        axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figure.suptitle("Candidate retrieval: exact top-K vs HNSW top-K (actual values)")
    handles = [
        plt.Line2D([], [], color="#222222", marker="o", label="Exact top-K"),
        plt.Line2D([], [], color="#0072B2", marker="s", label="HNSW top-K"),
    ]
    figure.legend(handles=handles, loc="lower center", ncol=2)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_kernel_lines(axis: Any, rows: Sequence[Mapping[str, Any]]) -> None:
    by_candidate: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_candidate[str(row["candidate_type"])].append(row)
    for candidate_type, group in sorted(by_candidate.items()):
        chosen = _best_kernel_by_n(group)
        xs = [int(item["corpus_user_count"]) for item in chosen]
        exact = [float(item["exact_p95_ms"]) for item in chosen]
        ann = [float(item["ann_p95_ms"]) for item in chosen]
        axis.plot(xs, exact, color="#222222", alpha=0.22, marker="o", linewidth=1)
        axis.plot(xs, ann, color="#0072B2", alpha=0.45, marker="s", linewidth=1)
        for item in chosen:
            if bool(item.get("screening_gate_passed")):
                axis.scatter(
                    [int(item["corpus_user_count"])],
                    [float(item["ann_p95_ms"])],
                    color="#009E73",
                    zorder=4,
                )


def _best_kernel_by_n(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["corpus_user_count"])].append(row)
    return [
        min(items, key=lambda item: float(item["ann_p95_ms"]))
        for _, items in sorted(grouped.items())
    ]


def _candidate_heatmap(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    sizes = sorted({int(row["corpus_user_count"]) for row in rows})
    figure, axes = plt.subplots(1, 5, figsize=(20, 4.4), constrained_layout=True)
    colors = {
        "verified speedup": 3,
        "quality fallback": 2,
        "exact faster": 1,
        "unvalidated": 0,
    }
    for axis, candidate_type in zip(
        axes, sorted({str(row["candidate_type"]) for row in rows}), strict=False
    ):
        matrix: list[list[int]] = []
        for requested_k in REQUIRED_K_VALUES:
            cells = [
                row
                for row in rows
                if str(row["candidate_type"]) == candidate_type
                and int(row["requested_k"]) == requested_k
            ]
            cells_by_n: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
            for cell in cells:
                cells_by_n[int(cell["corpus_user_count"])].append(cell)
            matrix.append(
                [
                    colors[_kernel_status(cells_by_n.get(size, []))]
                    for size in sizes
                ]
            )
        image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=0, vmax=3)
        axis.set_title(candidate_type)
        axis.set_xticks(range(len(sizes)), [f"{size // 1000}K" for size in sizes], rotation=45)
        axis.set_yticks(range(len(REQUIRED_K_VALUES)), [str(value) for value in REQUIRED_K_VALUES])
        axis.set_xlabel("N")
        if axis is axes[0]:
            axis.set_ylabel("K")
    figure.suptitle("Candidate retrieval decision map (actual screened cells)")
    figure.legend(
        handles=[Patch(color=plt.get_cmap("viridis")(value / 3), label=label) for label, value in colors.items()],
        loc="lower center",
        ncol=4,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _kernel_status(cells: Sequence[Mapping[str, Any]]) -> str:
    if not cells:
        return "unvalidated"
    if any(bool(cell.get("screening_gate_passed")) for cell in cells):
        return "verified speedup"
    if all(not bool(cell.get("quality_gate_passed")) for cell in cells):
        return "quality fallback"
    if all(float(cell["ann_p95_ratio"]) >= 1 for cell in cells):
        return "exact faster"
    return "unvalidated"


def _full_membership_scale_graph(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    confirmation_rows: Sequence[Mapping[str, Any]],
) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    points = _membership_plan_points(rows)
    for plan, values in points.items():
        axis.scatter(
            [item[0] for item in values],
            [item[1] for item in values],
            label=plan,
            alpha=0.75,
        )
    failures = [
        row
        for row in confirmation_rows
        if bool(row.get("part_a_candidate_retrieval_passed"))
        and not bool(row.get("ann_full_membership_passed"))
    ]
    if failures:
        axis.scatter(
            [int(row["corpus_user_count"]) for row in failures],
            [
                float(row.get("ann_corrected", row.get("ann_first", {}))["p95_ms"])
                for row in failures
            ],
            marker="x",
            s=80,
            color="#D55E00",
            label="Part A pass, Part B fail",
        )
    axis.set_xscale("log")
    axis.set_xlabel("N (actual cohort)")
    axis.set_ylabel("full membership p95 latency (ms)")
    axis.set_title("Full membership: exact_all vs filter_first_exact vs ann_corrected")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _membership_plan_points(rows: Sequence[Mapping[str, Any]]) -> Mapping[str, list[tuple[int, float]]]:
    result: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for row in rows:
        for key, label in (
            ("exact_all", "exact_all"),
            ("filter_first_exact", "filter_first_exact"),
            ("ann_corrected", "ann_corrected"),
        ):
            detail = row.get(key)
            if isinstance(detail, Mapping) and detail.get("p95_ms") is not None:
                result[label].append((int(row["corpus_user_count"]), float(detail["p95_ms"])))
    return result


def _read_stage_rows(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        return []
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("rows", []) if isinstance(payload, Mapping) else []
    return [item for item in rows if isinstance(item, Mapping)]


def _stage_cost_graph(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    if not rows:
        axis.text(0.5, 0.5, "No eligible Part B stage measurements", ha="center", va="center")
        axis.set_axis_off()
    else:
        stages = sorted(
            {
                stage
                for row in rows
                for stage in row.get("stage_ms", {})
                if stage != "search_total"
            }
        )
        labels = [str(row["candidate_id"])[:10] for row in rows]
        bottom = [0.0] * len(rows)
        for stage in stages:
            values = [float(row.get("stage_ms", {}).get(stage, 0.0)) for row in rows]
            axis.bar(labels, values, bottom=bottom, label=stage)
            bottom = [left + right for left, right in zip(bottom, values, strict=False)]
        axis.set_ylabel("stage p95 duration (ms)")
        axis.set_title("Full-membership stage cost breakdown (observed stages)")
        axis.legend(fontsize=7)
        axis.tick_params(axis="x", rotation=45)
    figure.savefig(path, dpi=160)
    plt.close(figure)
