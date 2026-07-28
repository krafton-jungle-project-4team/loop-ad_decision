#!/usr/bin/env python3
"""Finalize Goal 1 artifacts after every benchmark-v2 cohort checkpoint exists."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    ActualScalePoint,
    ScaleArtifactValidator,
    write_actual_only_reports,
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_COHORT_SIZES,
    SCALE_EXPERIMENT_VERSION,
    SCALE_OUTPUT_ROOT,
    ExpectedMemberBucket,
    HardMatchBucket,
    ScenarioSet,
    ScreeningCandidate,
    expected_member_bucket,
    hard_match_bucket,
    scale_manifest_template,
    select_survivors,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=SCALE_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.root
    if root.resolve() != (ROOT / SCALE_OUTPUT_ROOT).resolve():
        raise ValueError("Goal 1 finalization must stay under scale-series-v2")
    results = load_complete_results(root, SCALE_COHORT_SIZES)
    candidates = candidates_from_results(results)
    survivors = select_survivors(candidates)
    write_survivor_inputs(root, survivors)
    points = actual_scale_points(results)
    write_actual_points(root, points)
    write_actual_only_reports(
        points=points,
        expected_cohort_sizes=SCALE_COHORT_SIZES,
        csv_path=root / "reports/preliminary-scale-points.csv",
        markdown_path=root / "reports/preliminary-scale-report.md",
        png_path=root / "reports/preliminary-scale-report.png",
    )
    write_current_runtime_report(root, results)
    write_unvalidated_fallbacks(root, results, survivors.scale)
    write_goal1_summary(root, results, survivors)
    write_final_manifest(root, results, survivors, points)
    report = ScaleArtifactValidator(root).raise_for_errors()
    write_immutable_json(root / "integrity-report.json", report.to_dict())
    print(
        json.dumps(
            {
                "completed_cohort_sizes": list(SCALE_COHORT_SIZES),
                "scale_survivors": len(survivors.scale),
                "policy_survivors": len(survivors.policy),
                "goal2_union": len(survivors.goal2_union),
                "actual_points": len(points),
                "artifact_validation_passed": report.passed,
            },
            sort_keys=True,
        )
    )
    return 0


def load_complete_results(
    root: Path,
    cohort_sizes: Sequence[int],
) -> tuple[Mapping[str, Any], ...]:
    results: list[Mapping[str, Any]] = []
    for size in cohort_sizes:
        checkpoint = object_at(root / "checkpoints" / f"cohort-{size}.json")
        required_true = (
            "prefix_validated",
            "ground_truth_complete",
            "baseline_complete",
            "rss_complete",
            "spill_oom_complete",
        )
        if checkpoint.get("status") != "complete" or not all(
            checkpoint.get(field) is True for field in required_true
        ):
            raise RuntimeError(f"cohort {size} checkpoint is incomplete")
        result = object_at(root / "phase5" / f"cohort-{size}" / "screening-result.json")
        if int(result.get("cohort_size", -1)) != size:
            raise RuntimeError(f"cohort {size} screening result identity differs")
        results.append(result)
    return tuple(results)


def candidates_from_results(
    results: Sequence[Mapping[str, Any]],
) -> tuple[ScreeningCandidate, ...]:
    values: list[ScreeningCandidate] = []
    for result in results:
        for row in rows_at(result, "screening_candidates"):
            values.append(candidate_from_row(row))
    identities = [item.identity for item in values]
    if len(identities) != len(set(identities)):
        raise RuntimeError("cross-cohort screening candidates are not unique")
    return tuple(values)


def candidate_from_row(row: Mapping[str, Any]) -> ScreeningCandidate:
    return ScreeningCandidate(
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


def write_survivor_inputs(root: Path, survivors: Any) -> None:
    for filename, values in (
        ("scale-hnsw-tuning-inputs.json", survivors.scale),
        ("policy-hnsw-tuning-inputs.json", survivors.policy),
        ("goal2-tuning-inputs.json", survivors.goal2_union),
    ):
        write_immutable_json(
            root / filename,
            {
                "experiment_version": SCALE_EXPERIMENT_VERSION,
                "selection_source": "tuning_screening_only",
                "confirmation_used": False,
                "cells": [item.to_dict() for item in values],
            },
        )


def actual_scale_points(
    results: Sequence[Mapping[str, Any]],
) -> tuple[ActualScalePoint, ...]:
    points: list[ActualScalePoint] = []
    for result in results:
        size = int(result["cohort_size"])
        summaries = rows_at(result, "summaries")
        tuning = [row for row in summaries if row["scenario_set"] == "tuning"]
        by_scenario: dict[str, list[Mapping[str, Any]]] = {}
        for row in tuning:
            by_scenario.setdefault(str(row["scenario_id"]), []).append(row)
        evidence_sha = hashlib.sha256(
            json.dumps(
                result,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        for scenario_id, rows in sorted(by_scenario.items()):
            exact = unique_plan(rows, "exact_all")
            filtered = unique_plan(rows, "filter_first_exact")
            bucket_id = bucket_for_summary(exact)
            for plan, row in (("exact_all", exact), ("filter_first_exact", filtered)):
                points.append(
                    ActualScalePoint(
                        cohort_size=size,
                        scenario_id=scenario_id,
                        bucket_id=bucket_id,
                        plan=plan,
                        p95_ms=float(row["p95_ms"]),
                        measured_run_count=int(row["run_count"]),
                        quality_passed=(
                            plan == "exact_all" or exact_results_equal(exact, filtered)
                        ),
                        evidence_sha256=evidence_sha,
                    )
                )
            ann_rows = [row for row in rows if row["plan"] == "ann_first"]
            if not ann_rows:
                raise RuntimeError(f"{size}/{scenario_id} has no measured ANN point")
            valid_ann = [row for row in ann_rows if ann_row_is_scale_survivor(row, exact)]
            chosen = min(valid_ann or ann_rows, key=lambda row: float(row["p95_ms"]))
            points.append(
                ActualScalePoint(
                    cohort_size=size,
                    scenario_id=scenario_id,
                    bucket_id=bucket_id,
                    plan="ann_first",
                    p95_ms=float(chosen["p95_ms"]),
                    measured_run_count=int(chosen["run_count"]),
                    quality_passed=bool(valid_ann),
                    evidence_sha256=evidence_sha,
                )
            )
    return tuple(points)


def write_actual_points(root: Path, points: Sequence[ActualScalePoint]) -> None:
    path = root / "reports/preliminary-scale-points.jsonl"
    encoded = "".join(
        json.dumps(item.to_dict(), sort_keys=True) + "\n"
        for item in sorted(
            points,
            key=lambda item: (
                item.cohort_size,
                item.scenario_id,
                item.bucket_id,
                item.plan,
            ),
        )
    )
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError("actual-only point file changed after finalization")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(encoded, encoding="utf-8")


def write_current_runtime_report(
    root: Path,
    results: Sequence[Mapping[str, Any]],
) -> None:
    cells: list[Mapping[str, Any]] = []
    for result in results:
        for row in rows_at(result, "summaries"):
            if row["plan"] != "current_runtime":
                continue
            cells.append(
                {
                    "cohort_size": int(result["cohort_size"]),
                    "scenario_id": row["scenario_id"],
                    "scenario_set": row["scenario_set"],
                    "p50_ms": row["p50_ms"],
                    "p95_ms": row["p95_ms"],
                    "p99_ms": row["p99_ms"],
                    "max_ann_attempt_count": row["max_ann_attempt_count"],
                    "requested_k_histories": row["requested_k_histories"],
                    "max_audited_user_count": row["max_audited_user_count"],
                    "exact_fallback_rate": row["exact_fallback_rate"],
                    "stage_p95_ms": row["stage_p95_ms"],
                    "peak_rss_bytes": row["peak_rss_bytes"],
                    "temp_spill": row["temp_spill"],
                    "oom": row["oom"],
                }
            )
    write_immutable_json(
        root / "current-runtime-work-amplification.json",
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "backend_scope": "normalized_exact_frozen_hard_predicate_backend",
            "raw_event_chunk_rescan_included_in_latency": False,
            "cells": cells,
        },
    )


def write_unvalidated_fallbacks(
    root: Path,
    results: Sequence[Mapping[str, Any]],
    scale_survivors: Sequence[ScreeningCandidate],
) -> None:
    discovery = object_at(root / "phase2/unvalidated-scenario-fallbacks.json")
    survivor_keys = {
        (item.corpus_user_count, item.scenario_id) for item in scale_survivors
    }
    screening: list[Mapping[str, Any]] = []
    for result in results:
        size = int(result["cohort_size"])
        exact_rows = [
            row
            for row in rows_at(result, "summaries")
            if row["scenario_set"] == "tuning" and row["plan"] == "exact_all"
        ]
        for exact in exact_rows:
            key = (size, str(exact["scenario_id"]))
            if key in survivor_keys:
                continue
            reason = (
                "exact positive count below 100"
                if int(exact["exact_positive_count"]) < 100
                else "no baseline-HNSW K passed the Goal 1 scale screening gate"
            )
            screening.append(
                {
                    "cohort_size": size,
                    "scenario_id": exact["scenario_id"],
                    "candidate_type": exact["candidate_type"],
                    "bucket_id": bucket_for_summary(exact),
                    "fallback_plan": "exact_all",
                    "reason": reason,
                }
            )
    write_immutable_json(
        root / "unvalidated-fallbacks.json",
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "fallback_plan": "exact_all",
            "scenario_discovery": discovery["fallbacks"],
            "screening_cells": screening,
            "outside_validated_range": {
                "min_exclusive_users": max(SCALE_COHORT_SIZES),
                "reason": "Goal 1 validated no corpus above the largest cohort",
            },
            "excluded_measurement_backend": {
                "raw_event_chunk_rescan_included_in_latency": False,
                "reason": (
                    "the production repository's repeated 1000-candidate raw-event "
                    "rescans were excluded after the scale precheck; policy work "
                    "amplification was measured on the exact frozen signal backend"
                ),
            },
        },
    )


def write_goal1_summary(root: Path, results: Sequence[Mapping[str, Any]], survivors: Any) -> None:
    lines = [
        "# ANN scale-series Goal 1 result",
        "",
        f"- experiment: `{SCALE_EXPERIMENT_VERSION}`",
        f"- completed cohorts: `{', '.join(str(value) for value in SCALE_COHORT_SIZES)}`",
        f"- measured observations: `{sum(int(item['observation_count']) for item in results)}`",
        f"- scale survivors: `{len(survivors.scale)}`",
        f"- policy survivors: `{len(survivors.policy)}`",
        f"- Goal 2 deduplicated union: `{len(survivors.goal2_union)}`",
        "- historical 50K: `pilot-only`, no latency or ground truth reused",
        "- candidate policy generated: `false`",
        "- Goal 2 HNSW tuning executed: `false`",
        "",
        "Survivors are screening inputs only. They are not runtime policy values "
        "and have not passed Goal 2 tuning or confirmation gates.",
        "",
    ]
    path = root / "reports/goal1-summary.md"
    encoded = "\n".join(lines)
    if path.exists() and path.read_text(encoding="utf-8") != encoded:
        raise RuntimeError("Goal 1 summary changed after finalization")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(encoded, encoding="utf-8")


def write_final_manifest(
    root: Path,
    results: Sequence[Mapping[str, Any]],
    survivors: Any,
    points: Sequence[ActualScalePoint],
) -> None:
    source = object_at(root / "phase1/source-preflight.json")
    vector = object_at(root / "phase1/vector-snapshot.json")
    scenario_pairs = object_at(root / "phase2/scenario-pairs.json")
    payload = {
        **scale_manifest_template(),
        "status": "complete",
        "completed_cohort_sizes": list(SCALE_COHORT_SIZES),
        "validated_max_users": max(SCALE_COHORT_SIZES),
        "source_user_count": int(source["source_user_count"]),
        "source_vector_revision_count": int(vector["processed_user_count"]),
        "scenario_count": int(results[0]["scenario_count"]),
        "validated_scenario_pair_count": int(scenario_pairs["validated_pair_count"]),
        "unvalidated_scenario_pair_count": int(scenario_pairs["fallback_count"]),
        "scale_survivor_count": len(survivors.scale),
        "policy_survivor_count": len(survivors.policy),
        "goal2_union_count": len(survivors.goal2_union),
        "actual_only_point_count": len(points),
        "historical_50k_decision": "pilot_only",
        "candidate_policy_generated": False,
        "goal2_executed": False,
    }
    write_immutable_json(root / "manifest.json", payload)


def ann_row_is_scale_survivor(
    ann: Mapping[str, Any],
    exact: Mapping[str, Any],
) -> bool:
    return (
        int(ann["exact_positive_count"]) >= 100
        and float(ann["recall"]) >= 0.90
        and float(ann["p95_ms"]) <= float(exact["p95_ms"]) * 1.20
        and bool(ann["index_used"])
        and not bool(ann["temp_spill"])
        and not bool(ann["oom"])
    )


def exact_results_equal(
    exact: Mapping[str, Any],
    filtered: Mapping[str, Any],
) -> bool:
    return (
        int(filtered["final_user_count"]) == int(exact["final_user_count"])
        and int(filtered["intersection_count"]) == int(exact["intersection_count"])
        and int(filtered["final_user_count"]) == int(exact["exact_positive_count"])
    )


def bucket_for_summary(summary: Mapping[str, Any]) -> str:
    hard: HardMatchBucket = hard_match_bucket(float(summary["hard_match_ratio"]))
    expected: ExpectedMemberBucket = expected_member_bucket(
        float(summary["expected_member_ratio"])
    )
    return f"{hard.value}__{expected.value}"


def unique_plan(rows: Sequence[Mapping[str, Any]], plan: str) -> Mapping[str, Any]:
    values = [row for row in rows if row["plan"] == plan]
    if len(values) != 1:
        raise RuntimeError(f"expected one {plan} summary, observed {len(values)}")
    return values[0]


def rows_at(payload: Mapping[str, Any], key: str) -> list[Mapping[str, Any]]:
    value = payload.get(key)
    if not isinstance(value, list) or any(not isinstance(item, Mapping) for item in value):
        raise ValueError(f"{key} must be an object array")
    return value


def object_at(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
