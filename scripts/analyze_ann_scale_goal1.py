#!/usr/bin/env python3
"""Analyze benchmark v2 measurements and write immutable cohort checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_benchmark import (  # noqa: E402
    BenchmarkManifest,
    enumerate_unique_candidate_counts,
)
from offline_evaluation.ann_search_experiment import (  # noqa: E402
    BenchmarkObservation,
    BenchmarkPhase,
    SearchPlan,
    load_observations,
    summarize_observations,
)
from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    checkpoint_payload,
    sha256_file,
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_COHORT_SIZES,
    SCALE_EXPERIMENT_VERSION,
    SCALE_OUTPUT_ROOT,
    ScenarioSet,
    ScreeningCandidate,
    build_fingerprint,
    confirmation_baseline_k,
    select_survivors,
)


MEASUREMENT_CODE_PATHS = (
    Path("offline_evaluation/ann_search_benchmark.py"),
    Path("offline_evaluation/ann_search_experiment.py"),
    Path("offline_evaluation/ann_search_scale_series.py"),
    Path("offline_evaluation/ann_search_scale_artifacts.py"),
    Path("scripts/benchmark_audience_search.py"),
    Path("scripts/run_ann_scale_goal1_measurements.py"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=SCALE_OUTPUT_ROOT)
    parser.add_argument(
        "--cohort-sizes", type=positive_ints, required=True
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output_root.resolve() != (ROOT / SCALE_OUTPUT_ROOT).resolve():
        raise ValueError("analysis output must remain under scale-series-v2")
    fingerprint = ensure_fingerprint(args.output_root)
    manifest = BenchmarkManifest.load(
        args.output_root / "phase2/selected-scenario-manifest.json"
    )
    for size in args.cohort_sizes:
        analyze_cohort(
            root=args.output_root,
            cohort_size=size,
            manifest=manifest,
            fingerprint_sha256=str(fingerprint["fingerprint_sha256"]),
        )
    return 0


def ensure_fingerprint(root: Path) -> Mapping[str, Any]:
    source = object_at(root / "phase1/source-preflight.json")
    vector = object_at(root / "phase1/vector-snapshot.json")
    membership = object_at(root / "phase1/membership-manifest.json")
    scenario_path = root / "phase2/selected-scenario-manifest.json"
    environment_path = root / "environment.json"
    resource_path = root / "resource-manifest.json"
    code_sha = measurement_code_sha256(MEASUREMENT_CODE_PATHS)
    backend = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "backend": "frozen_cohort_user_signals",
        "relation": "ann_benchmark_user_signals",
        "membership_relation": "ann_benchmark_scale_membership",
        "semantic_source": "production hard-predicate formulas over frozen raw window",
        "current_runtime_scope": (
            "production selector/K-growth/audit/fallback with normalized exact "
            "frozen hard-predicate backend"
        ),
        "excluded_from_latency": (
            "current repository 1000-candidate raw-event chunk rescans; observed "
            "during aborted scale precheck"
        ),
        "measurement_code_sha256": code_sha,
        "measurement_code_paths": [str(path) for path in MEASUREMENT_CODE_PATHS],
    }
    backend_path = root / "phase0/scale-ready-backend.json"
    if backend_path.exists():
        if object_at(backend_path) != backend:
            raise RuntimeError("measurement code changed after fingerprint freeze")
    else:
        write_immutable_json(backend_path, backend)
    payload = build_fingerprint(
        experiment_version=SCALE_EXPERIMENT_VERSION,
        scale_series_id=membership["scale_series_id"],
        project_id=vector["project_id"],
        vector_version=vector["vector_version"],
        vector_manifest_hash=vector["manifest_hash"],
        vector_generation_id=vector["vector_generation_id"],
        window_start=vector["window_start"],
        window_end=vector["window_end"],
        source_revision_cutoff=vector["source_revision_cutoff"],
        source_user_count=int(source["source_user_count"]),
        source_vector_revision_count=int(vector["processed_user_count"]),
        raw_event_count=int(source["preview_raw_event_count"]),
        cohort_seed=membership["cohort_seed"],
        membership_sha256=membership["membership_sha256"],
        scenario_manifest_sha256=sha256_file(scenario_path),
        environment_sha256=sha256_file(environment_path),
        resource_manifest_sha256=sha256_file(resource_path),
        code_revision=f"measurement-sha256:{code_sha}",
    )
    path = root / "fingerprint.json"
    if path.exists():
        if object_at(path) != payload:
            raise RuntimeError("frozen benchmark fingerprint changed")
    else:
        write_immutable_json(path, payload)
    return payload


def analyze_cohort(
    *,
    root: Path,
    cohort_size: int,
    manifest: BenchmarkManifest,
    fingerprint_sha256: str,
) -> None:
    if cohort_size not in SCALE_COHORT_SIZES:
        raise ValueError("cohort size is outside the frozen scale series")
    measurement_root = root / "phase4" / f"cohort-{cohort_size}"
    scenario_files = sorted((measurement_root / "raw").glob("*/*.jsonl"))
    if len(scenario_files) != len(manifest.scenarios):
        raise RuntimeError("scenario measurement files are incomplete")
    result_root = root / "phase5" / f"cohort-{cohort_size}"
    combined = result_root / "raw-observations.jsonl"
    if not combined.exists():
        combined.parent.mkdir(parents=True, exist_ok=True)
        partial = combined.with_suffix(".partial.jsonl")
        with partial.open("w", encoding="utf-8") as output:
            for path in scenario_files:
                output.write(path.read_text(encoding="utf-8"))
        partial.replace(combined)
    observations = load_observations(combined)
    summaries = summarize_observations(observations)
    candidates = screening_candidates(summaries)
    survivor_sets = select_survivors(candidates)
    expected_cells = expected_cell_count(
        cohort_size=cohort_size,
        manifest=manifest,
        summaries=summaries,
    )
    if len(summaries) != expected_cells:
        raise RuntimeError(
            f"baseline cell count mismatch: expected {expected_cells}, "
            f"observed {len(summaries)}"
        )
    result = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "cohort_size": cohort_size,
        "scenario_count": len(manifest.scenarios),
        "observation_count": len(observations),
        "measured_observation_count": sum(item.measured for item in observations),
        "baseline_cell_count": len(summaries),
        "summaries": [item.to_dict() for item in summaries],
        "screening_candidates": [item.to_dict() for item in candidates],
        "survivors": survivor_sets.to_dict(),
        "confirmation_used_for_survivor_selection": False,
        "rss_complete": all(item.peak_rss_bytes is not None for item in observations),
        "spill_oom_complete": all(
            isinstance(item.temp_spill, bool) and isinstance(item.oom, bool)
            for item in observations
        ),
    }
    result_path = result_root / "screening-result.json"
    if result_path.exists():
        if object_at(result_path) != result:
            raise RuntimeError("immutable screening result changed")
    else:
        write_immutable_json(result_path, result)

    ground_truth = ground_truth_entries(
        measurement_root=measurement_root,
        manifest=manifest,
        cohort_size=cohort_size,
    )
    cohort_manifest = object_at(
        root / "phase3/cohorts" / f"cohort-{cohort_size}.json"
    )
    checkpoint = checkpoint_payload(
        cohort_size=cohort_size,
        cohort_sha256=str(cohort_manifest["result"]["cohort_sha256"]),
        fingerprint_sha256=fingerprint_sha256,
        raw_path=combined,
        result_path=result_path,
        ground_truth=ground_truth,
        expected_scenario_count=len(manifest.scenarios),
        expected_baseline_cell_count=expected_cells,
        observed_baseline_cell_count=len(summaries),
        scale_survivor_count=len(survivor_sets.scale),
        policy_survivor_count=len(survivor_sets.policy),
        rss_complete=bool(result["rss_complete"]),
        spill_oom_complete=bool(result["spill_oom_complete"]),
    )
    if not all(
        checkpoint[field] is True
        for field in (
            "prefix_validated",
            "ground_truth_complete",
            "baseline_complete",
            "rss_complete",
            "spill_oom_complete",
        )
    ):
        raise RuntimeError("cohort checkpoint completeness gate failed")
    checkpoint_path = root / "checkpoints" / f"cohort-{cohort_size}.json"
    if checkpoint_path.exists():
        if object_at(checkpoint_path) != checkpoint:
            raise RuntimeError("immutable cohort checkpoint changed")
    else:
        write_immutable_json(checkpoint_path, checkpoint)
    print(
        json.dumps(
            {
                "cohort_size": cohort_size,
                "baseline_cells": len(summaries),
                "scale_survivors": len(survivor_sets.scale),
                "policy_survivors": len(survivor_sets.policy),
                "checkpoint": str(checkpoint_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )


def screening_candidates(summaries: Sequence[Any]) -> tuple[ScreeningCandidate, ...]:
    lookup = {
        (item.scenario_id, item.plan, item.requested_k): item
        for item in summaries
    }
    candidates: list[ScreeningCandidate] = []
    for ann in summaries:
        if ann.plan != SearchPlan.ANN_FIRST:
            continue
        if ann.scenario_set != ScenarioSet.TUNING:
            continue
        exact = lookup[(ann.scenario_id, SearchPlan.EXACT_ALL, None)]
        filtered = lookup[(
            ann.scenario_id,
            SearchPlan.FILTER_FIRST_EXACT,
            None,
        )]
        filter_equal = (
            filtered.final_user_count == exact.final_user_count
            and filtered.intersection_count == exact.intersection_count
            and filtered.final_user_count == exact.exact_positive_count
        )
        candidates.append(
            ScreeningCandidate(
                scenario_id=ann.scenario_id,
                candidate_type=ann.candidate_type,
                scenario_set=ann.scenario_set,
                corpus_user_count=ann.corpus_user_count,
                requested_k=int(ann.requested_k),
                exact_positive_count=ann.exact_positive_count,
                recall=ann.recall,
                ann_p95_ms=ann.p95_ms,
                exact_all_p95_ms=exact.p95_ms,
                filter_first_p95_ms=filtered.p95_ms,
                filter_first_results_equal=filter_equal,
                hnsw_index_used=ann.index_used,
                temp_spill=ann.temp_spill,
                oom=ann.oom,
            )
        )
    return tuple(sorted(candidates, key=lambda item: item.identity))


def expected_cell_count(
    *, cohort_size: int, manifest: BenchmarkManifest, summaries: Sequence[Any]
) -> int:
    by_scenario = {item.scenario_id: item for item in summaries if item.plan == SearchPlan.EXACT_ALL}
    expected = 0
    for scenario in manifest.scenarios:
        exact = by_scenario[scenario.scenario_id]
        if scenario.scenario_set == ScenarioSet.TUNING:
            k_values = enumerate_unique_candidate_counts(
                corpus_user_count=cohort_size,
                expected_member_count=exact.expected_member_count,
            )
            expected += 3 + len(k_values)
        else:
            expected_k = confirmation_baseline_k(
                corpus_user_count=cohort_size,
                estimated_member_count=exact.expected_member_count,
            )
            ann = next(
                item for item in summaries
                if item.scenario_id == scenario.scenario_id
                and item.plan == SearchPlan.ANN_FIRST
            )
            if ann.requested_k != expected_k:
                raise RuntimeError("confirmation K differs from preregistration")
            expected += 4
    return expected


def ground_truth_entries(
    *, measurement_root: Path, manifest: BenchmarkManifest, cohort_size: int
) -> Mapping[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for scenario in manifest.scenarios:
        path = measurement_root / "ground-truth" / f"{scenario.scenario_id}.csv.gz"
        summary_path = path.with_suffix("").with_suffix(".summary.json")
        summary = object_at(summary_path)
        if int(summary["corpus_user_count"]) != cohort_size:
            raise RuntimeError("ground-truth summary row count mismatch")
        result[scenario.scenario_id] = {
            "path": str(path),
            "row_count": int(summary["corpus_user_count"]),
            "sha256": sha256_file(path),
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
        }
    return dict(sorted(result.items()))


def measurement_code_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for relative in sorted(paths):
        path = ROOT / relative
        digest.update(str(relative).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def object_at(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def positive_ints(value: str) -> tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("cohort sizes must be positive")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
