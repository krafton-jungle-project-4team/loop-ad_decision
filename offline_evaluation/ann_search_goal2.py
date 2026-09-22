"""Pure planning and analysis helpers for ANN scale-series Goal 2.

Goal 1 artifacts are immutable inputs.  This module deliberately writes
nothing and keeps policy identifiability separate from scale-performance
eligibility so the orchestration layer can fail closed before live DB work.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from offline_evaluation.ann_search_experiment import (
    BenchmarkObservation,
    BenchmarkPhase,
    DEFAULT_CANDIDATE_TYPES,
    SearchPlan,
    summarize_observations,
)
from offline_evaluation.ann_search_scale_artifacts import sha256_file
from offline_evaluation.ann_search_scale_series import (
    SCALE_COHORT_SIZES,
    SCALE_EXPERIMENT_VERSION,
    expected_member_bucket,
    hard_match_bucket,
)


GOAL1_ROOT = Path("artifacts/ann-search/scale-series-v2")
GOAL2_ROOT = GOAL1_ROOT / "goal2"
HNSW_EF_SEARCH = (50, 100, 200, 400)
HNSW_ITERATIVE_SCAN = ("strict_order", "relaxed_order")
HNSW_MAX_SCAN_TUPLES = (20_000, 50_000, 100_000)
GOAL2_WARMUPS = 3
GOAL2_REPETITIONS = 10


@dataclass(frozen=True, slots=True, order=True)
class Goal2TuningCell:
    corpus_user_count: int
    scenario_id: str
    candidate_type: str
    requested_k: int
    ef_search: int
    iterative_scan: str
    max_scan_tuples: int

    @property
    def identity(self) -> tuple[Any, ...]:
        return (
            self.corpus_user_count,
            self.scenario_id,
            self.requested_k,
            self.ef_search,
            self.iterative_scan,
            self.max_scan_tuples,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["cell_id"] = tuning_cell_id(self.identity)
        return value


def load_json_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tuning_cell_id(identity: Sequence[Any]) -> str:
    return "goal2-cell-" + canonical_json_sha256(list(identity))[:24]


def full_hnsw_grid_payload() -> tuple[Mapping[str, Any], ...]:
    return tuple(
        {
            "ef_search": ef_search,
            "iterative_scan": iterative_scan,
            "max_scan_tuples": max_scan_tuples,
        }
        for ef_search in HNSW_EF_SEARCH
        for iterative_scan in HNSW_ITERATIVE_SCAN
        for max_scan_tuples in HNSW_MAX_SCAN_TUPLES
    )


def goal1_required_input_paths(root: Path = GOAL1_ROOT) -> tuple[Path, ...]:
    relative = (
        "reports/goal1-summary.md",
        "reports/preliminary-scale-report.md",
        "goal2-tuning-inputs.json",
        "policy-hnsw-tuning-inputs.json",
        "scale-hnsw-tuning-inputs.json",
        "unvalidated-fallbacks.json",
        "current-runtime-work-amplification.json",
        "manifest.json",
        "integrity-report.json",
        "fingerprint.json",
        "environment.json",
        "resource-manifest.json",
        "phase0/scale-ready-backend.json",
        "phase1/source-preflight.json",
        "phase1/vector-snapshot.json",
        "phase1/membership-manifest.json",
        "phase2/scenario-pairs.json",
        "phase2/scenario-census.jsonl",
        "phase2/selected-scenario-manifest.json",
        "phase3/cohort-preparation-manifest.json",
    )
    paths = tuple(root / item for item in relative)
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError("Goal 1 inputs missing: " + ", ".join(missing))
    return paths


def goal1_required_sha256(root: Path = GOAL1_ROOT) -> Mapping[str, str]:
    return {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in goal1_required_input_paths(root)
    }


def build_policy_coverage_audit(root: Path = GOAL1_ROOT) -> Mapping[str, Any]:
    """Reproduce policy coverage using only frozen Goal 1 scenarios/results."""

    selected_manifest = load_json_object(
        root / "phase2/selected-scenario-manifest.json"
    )
    scenarios_raw = selected_manifest.get("scenarios")
    if not isinstance(scenarios_raw, list) or not scenarios_raw:
        raise ValueError("selected scenario manifest is empty")
    scenarios = {
        str(item["scenario_id"]): dict(item)
        for item in scenarios_raw
        if isinstance(item, Mapping)
    }
    if len(scenarios) != len(scenarios_raw):
        raise ValueError("selected scenario manifest has invalid/duplicate rows")

    pair_manifest = load_json_object(root / "phase2/scenario-pairs.json")
    pairs_raw = pair_manifest.get("pairs")
    if not isinstance(pairs_raw, list):
        raise ValueError("scenario pair manifest is invalid")
    validated_pairs = tuple(
        dict(item)
        for item in pairs_raw
        if isinstance(item, Mapping) and item.get("validated") is True
    )

    tuning_input = load_json_object(root / "goal2-tuning-inputs.json")
    survivor_rows = tuning_input.get("cells")
    if not isinstance(survivor_rows, list) or len(survivor_rows) != 104:
        raise ValueError("Goal 2 tuning input must contain the frozen 104 cells")
    survivors_by_point: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in survivor_rows:
        if not isinstance(item, Mapping):
            raise ValueError("Goal 2 tuning input row is invalid")
        survivors_by_point[
            (int(item["corpus_user_count"]), str(item["scenario_id"]))
        ].append(item)

    buckets: list[dict[str, Any]] = []
    bucket_keys_by_cohort: dict[int, set[tuple[str, str]]] = {}
    for cohort_size in SCALE_COHORT_SIZES:
        summaries = _ground_truth_summaries(root, cohort_size)
        bucket_for_scenario = {
            scenario_id: (
                hard_match_bucket(
                    int(summary["hard_match_user_count"]) / cohort_size
                ).value,
                expected_member_bucket(
                    float(summary["estimated_member_count"]) / cohort_size
                ).value,
            )
            for scenario_id, summary in summaries.items()
        }
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        for scenario_id, bucket in bucket_for_scenario.items():
            grouped[bucket].append(scenario_id)
        bucket_keys_by_cohort[cohort_size] = set(grouped)

        preliminary_filter = _preliminary_filter_first_scenarios(root, cohort_size)
        for (hard_bucket, expected_bucket), scenario_ids in sorted(grouped.items()):
            tuning_ids = sorted(
                item
                for item in scenario_ids
                if str(scenarios[item]["scenario_set"]) == "tuning"
            )
            confirmation_ids = sorted(
                item
                for item in scenario_ids
                if str(scenarios[item]["scenario_set"]) == "confirmation"
            )
            candidate_types = sorted(
                {str(scenarios[item]["candidate_type"]) for item in scenario_ids}
            )
            paired_types: set[str] = set()
            pair_ids: list[dict[str, str]] = []
            for pair in validated_pairs:
                tuning_id = str(pair["tuning_scenario_id"])
                confirmation_id = str(pair["confirmation_scenario_id"])
                if (
                    bucket_for_scenario.get(tuning_id)
                    == (hard_bucket, expected_bucket)
                    and bucket_for_scenario.get(confirmation_id)
                    == (hard_bucket, expected_bucket)
                ):
                    candidate_type = str(pair["candidate_type"])
                    paired_types.add(candidate_type)
                    pair_ids.append(
                        {
                            "candidate_type": candidate_type,
                            "tuning_scenario_id": tuning_id,
                            "confirmation_scenario_id": confirmation_id,
                        }
                    )
            survivor_scenarios = sorted(
                {
                    scenario_id
                    for scenario_id in tuning_ids
                    if survivors_by_point.get((cohort_size, scenario_id))
                }
            )
            filter_candidates = sorted(
                set(scenario_ids) & preliminary_filter
            )
            coverage = set(DEFAULT_CANDIDATE_TYPES).issubset(paired_types)
            buckets.append(
                {
                    "corpus_user_count": cohort_size,
                    "n_band": f"observed_{cohort_size}",
                    "hard_match_bucket": hard_bucket,
                    "expected_member_bucket": expected_bucket,
                    "candidate_types": candidate_types,
                    "tuning_scenario_ids": tuning_ids,
                    "confirmation_scenario_ids": confirmation_ids,
                    "validated_pairs": sorted(
                        pair_ids,
                        key=lambda item: (
                            item["candidate_type"],
                            item["tuning_scenario_id"],
                        ),
                    ),
                    "paired_candidate_types": sorted(paired_types),
                    "tuning_confirmation_pair_exists": bool(pair_ids),
                    "ann_screening_survivor_exists": bool(survivor_scenarios),
                    "ann_screening_survivor_scenarios": survivor_scenarios,
                    "filter_first_preliminary_candidate_exists": bool(
                        filter_candidates
                    ),
                    "filter_first_preliminary_scenarios": filter_candidates,
                    "five_candidate_type_coverage": coverage,
                    "adjacent_cohort_boundary_coverage": False,
                    "policy_rule_possible": False,
                }
            )

    ordered_cohorts = tuple(SCALE_COHORT_SIZES)
    by_identity = {
        (
            int(item["corpus_user_count"]),
            str(item["hard_match_bucket"]),
            str(item["expected_member_bucket"]),
        ): item
        for item in buckets
    }
    for index, cohort_size in enumerate(ordered_cohorts):
        adjacent_sizes = tuple(
            ordered_cohorts[position]
            for position in (index - 1, index + 1)
            if 0 <= position < len(ordered_cohorts)
        )
        for hard_bucket, expected_bucket in bucket_keys_by_cohort[cohort_size]:
            item = by_identity[(cohort_size, hard_bucket, expected_bucket)]
            adjacent = bool(adjacent_sizes) and all(
                (hard_bucket, expected_bucket) in bucket_keys_by_cohort[size]
                for size in adjacent_sizes
            )
            item["adjacent_cohort_boundary_coverage"] = adjacent
            item["policy_rule_possible"] = bool(
                item["five_candidate_type_coverage"]
                and adjacent
                and (
                    item["ann_screening_survivor_exists"]
                    or item["filter_first_preliminary_candidate_exists"]
                )
            )

    full_coverage = sum(
        1 for item in buckets if item["five_candidate_type_coverage"]
    )
    possible = sum(1 for item in buckets if item["policy_rule_possible"])
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "policy_dimensions": ["N", "H/N", "E/N"],
        "required_candidate_types": list(DEFAULT_CANDIDATE_TYPES),
        "bucket_count": len(buckets),
        "five_candidate_type_coverage_bucket_count": full_coverage,
        "policy_rule_possible_bucket_count": possible,
        "product_policy_track_status": (
            "identifiable"
            if possible
            else "not identifiable under current policy dimensions"
        ),
        "product_confirmation_branch_enabled": possible > 0,
        "buckets": sorted(
            buckets,
            key=lambda item: (
                item["corpus_user_count"],
                item["hard_match_bucket"],
                item["expected_member_bucket"],
            ),
        ),
    }


def build_tuning_cells(root: Path = GOAL1_ROOT) -> tuple[Goal2TuningCell, ...]:
    payload = load_json_object(root / "goal2-tuning-inputs.json")
    raw = payload.get("cells")
    if not isinstance(raw, list) or len(raw) != 104:
        raise ValueError("Goal 2 tuning input must contain 104 cells")
    grid = full_hnsw_grid_payload()
    result: list[Goal2TuningCell] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("Goal 2 tuning input row is invalid")
        for hnsw in grid:
            result.append(
                Goal2TuningCell(
                    corpus_user_count=int(item["corpus_user_count"]),
                    scenario_id=str(item["scenario_id"]),
                    candidate_type=str(item["candidate_type"]),
                    requested_k=int(item["requested_k"]),
                    ef_search=int(hnsw["ef_search"]),
                    iterative_scan=str(hnsw["iterative_scan"]),
                    max_scan_tuples=int(hnsw["max_scan_tuples"]),
                )
            )
    cells = tuple(sorted(result))
    if len(cells) != 2_496 or len({item.identity for item in cells}) != len(cells):
        raise ValueError("Goal 2 tuning Cartesian product is not exactly 104x24")
    return cells


def build_execution_manifest(root: Path = GOAL1_ROOT) -> Mapping[str, Any]:
    cells = build_tuning_cells(root)
    points = sorted(
        {
            (item.corpus_user_count, item.scenario_id, item.candidate_type)
            for item in cells
        }
    )
    if len(points) != 19:
        raise ValueError("Goal 2 input must contain exactly 19 cohort/scenario points")
    by_cohort: dict[int, dict[str, int]] = {}
    for cohort_size in SCALE_COHORT_SIZES:
        cohort_cells = [item for item in cells if item.corpus_user_count == cohort_size]
        point_count = sum(1 for item in points if item[0] == cohort_size)
        tuning_cell_count = len(cohort_cells)
        baseline_cell_count = point_count * 3
        by_cohort[cohort_size] = {
            "point_count": point_count,
            "ann_tuning_cell_count": tuning_cell_count,
            "baseline_cell_count": baseline_cell_count,
            "expected_invocation_count": (
                tuning_cell_count + baseline_cell_count
            ) * (GOAL2_WARMUPS + GOAL2_REPETITIONS),
            "expected_measured_observation_count": (
                tuning_cell_count + baseline_cell_count
            ) * GOAL2_REPETITIONS,
        }
    manifest = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "goal": "ann-scale-performance-final-offline-validation",
        "status": "execution_locked",
        "goal1_tuning_input_sha256": sha256_file(
            root / "goal2-tuning-inputs.json"
        ),
        "cohort_sizes": list(SCALE_COHORT_SIZES),
        "unique_point_count": len(points),
        "goal1_union_cell_count": 104,
        "hnsw_setting_count": len(full_hnsw_grid_payload()),
        "ann_tuning_cell_count": len(cells),
        "baseline_cell_count": len(points) * 3,
        "total_core_cell_count": len(cells) + len(points) * 3,
        "warmups_per_cell": GOAL2_WARMUPS,
        "measured_runs_per_cell": GOAL2_REPETITIONS,
        "expected_ann_invocation_count": len(cells)
        * (GOAL2_WARMUPS + GOAL2_REPETITIONS),
        "expected_baseline_invocation_count": len(points)
        * 3
        * (GOAL2_WARMUPS + GOAL2_REPETITIONS),
        "expected_total_invocation_count": (len(cells) + len(points) * 3)
        * (GOAL2_WARMUPS + GOAL2_REPETITIONS),
        "expected_measured_observation_count": (len(cells) + len(points) * 3)
        * GOAL2_REPETITIONS,
        "hnsw_grid": list(full_hnsw_grid_payload()),
        "points": [
            {
                "corpus_user_count": cohort_size,
                "scenario_id": scenario_id,
                "candidate_type": candidate_type,
                "requested_k_values": sorted(
                    {
                        item.requested_k
                        for item in cells
                        if item.corpus_user_count == cohort_size
                        and item.scenario_id == scenario_id
                    }
                ),
            }
            for cohort_size, scenario_id, candidate_type in points
        ],
        "by_cohort": {str(key): value for key, value in by_cohort.items()},
    }
    return {**manifest, "execution_manifest_sha256": canonical_json_sha256(manifest)}


def render_policy_coverage_markdown(audit: Mapping[str, Any]) -> str:
    lines = [
        "# Goal 2 policy coverage audit",
        "",
        f"- status: `{audit['product_policy_track_status']}`",
        f"- observed buckets: `{audit['bucket_count']}`",
        "- five-candidate-type coverage buckets: "
        f"`{audit['five_candidate_type_coverage_bucket_count']}`",
        f"- policy-rule-possible buckets: `{audit['policy_rule_possible_bucket_count']}`",
        "",
        "| N | H/N bucket | E/N bucket | types | pair | ANN survivor | "
        "Filter preliminary | five types | adjacent | rule possible |",
        "|---:|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in audit["buckets"]:
        lines.append(
            "| {corpus_user_count} | {hard_match_bucket} | "
            "{expected_member_bucket} | {types} | {pair} | {ann} | {filter} | "
            "{coverage} | {adjacent} | {possible} |".format(
                **item,
                types=", ".join(item["candidate_types"]),
                pair=int(item["tuning_confirmation_pair_exists"]),
                ann=int(item["ann_screening_survivor_exists"]),
                filter=int(item["filter_first_preliminary_candidate_exists"]),
                coverage=int(item["five_candidate_type_coverage"]),
                adjacent=int(item["adjacent_cohort_boundary_coverage"]),
                possible=int(item["policy_rule_possible"]),
            )
        )
    lines.extend(
        [
            "",
            "Coverage absence is a product-policy negative result, not a scale-track blocker.",
            "Candidate type remains evidence coverage and is not added to the policy key.",
            "",
        ]
    )
    return "\n".join(lines)


def _ground_truth_summaries(
    root: Path, cohort_size: int
) -> Mapping[str, Mapping[str, Any]]:
    directory = root / "phase4" / f"cohort-{cohort_size}" / "ground-truth"
    result: dict[str, Mapping[str, Any]] = {}
    for path in sorted(directory.glob("*.summary.json")):
        payload = load_json_object(path)
        scenario_id = str(payload["scenario_id"])
        if int(payload["corpus_user_count"]) != cohort_size:
            raise ValueError(f"ground truth cohort mismatch: {path}")
        result[scenario_id] = payload
    if len(result) != 22:
        raise ValueError(f"cohort {cohort_size} must contain 22 ground truths")
    return result


def _preliminary_filter_first_scenarios(
    root: Path, cohort_size: int
) -> set[str]:
    payload = load_json_object(
        root / "phase5" / f"cohort-{cohort_size}" / "screening-result.json"
    )
    summaries = payload.get("summaries")
    if not isinstance(summaries, list):
        raise ValueError("Goal 1 screening summaries are invalid")
    grouped: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for item in summaries:
        if not isinstance(item, Mapping):
            raise ValueError("Goal 1 screening summary row is invalid")
        if item.get("scenario_set") != "tuning":
            continue
        grouped[str(item["scenario_id"])][str(item["plan"])] = item
    result: set[str] = set()
    for scenario_id, plans in grouped.items():
        exact = plans.get("exact_all")
        filtered = plans.get("filter_first_exact")
        if exact is None or filtered is None:
            continue
        results_equal = (
            int(exact["final_user_count"]) == int(filtered["final_user_count"])
            and int(exact["intersection_count"])
            == int(filtered["intersection_count"])
            and float(filtered["precision"]) == 1.0
            and float(filtered["recall"]) == 1.0
        )
        if (
            results_equal
            and float(filtered["p95_ms"]) <= float(exact["p95_ms"]) * 0.85
            and float(filtered["p99_ms"]) <= float(exact["p99_ms"])
        ):
            result.add(scenario_id)
    return result


def iter_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise ValueError(f"blank JSONL row at {path}:{line_number}")
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise ValueError(f"invalid JSONL object at {path}:{line_number}")
            yield payload


def analyze_tuning_measurements(
    *,
    goal1_root: Path = GOAL1_ROOT,
    goal2_root: Path = GOAL2_ROOT,
    diagnostic_rows: Sequence[Mapping[str, Any]] = (),
) -> Mapping[str, Any]:
    """Aggregate all registered tuning rows and evaluate the two ANN gates.

    Timing rows intentionally do not run EXPLAIN.  A cell can become a final
    winner only when a separately collected diagnostic row proves HNSW index
    usage and absence of spill.  Cells that already fail another gate remain a
    final negative result without diagnostic work.
    """

    execution = load_json_object(goal2_root / "goal2-execution-manifest.json")
    points = execution.get("points")
    if not isinstance(points, list) or len(points) != 19:
        raise ValueError("Goal 2 execution point manifest is invalid")
    summaries: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, Mapping):
            raise ValueError("Goal 2 execution point is invalid")
        cohort_size = int(point["corpus_user_count"])
        scenario_id = str(point["scenario_id"])
        raw_root = (
            goal2_root
            / "raw/tuning"
            / f"cohort-{cohort_size}"
            / scenario_id
        )
        observations: list[BenchmarkObservation] = []
        for filename in ("baselines.jsonl", "ann.jsonl"):
            path = raw_root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Goal 2 raw block missing: {path}")
            observations.extend(
                BenchmarkObservation.from_dict(item) for item in iter_jsonl(path)
            )
        block_summaries = summarize_observations(
            observations,
            phase=BenchmarkPhase.TUNING,
        )
        for item in block_summaries:
            payload = item.to_dict()
            payload["duration_samples_ms"] = list(item.duration_samples_ms)
            summaries.append(payload)

    baseline_by_point: dict[tuple[int, str], dict[str, Mapping[str, Any]]] = defaultdict(dict)
    ann_summaries: list[Mapping[str, Any]] = []
    for summary in summaries:
        key = (int(summary["corpus_user_count"]), str(summary["scenario_id"]))
        if summary["plan"] == SearchPlan.ANN_FIRST.value:
            ann_summaries.append(summary)
        else:
            baseline_by_point[key][str(summary["plan"])] = summary
    if len(ann_summaries) != 2_496:
        raise ValueError("Goal 2 aggregate does not contain 2,496 ANN cells")
    if any(set(plans) != {"current_runtime", "exact_all", "filter_first_exact"}
           for plans in baseline_by_point.values()):
        raise ValueError("Goal 2 matched baseline coverage is incomplete")

    diagnostics = {
        _diagnostic_identity(item): item for item in diagnostic_rows
    }
    if len(diagnostics) != len(diagnostic_rows):
        raise ValueError("Goal 2 diagnostic rows contain duplicate identities")
    screening = load_json_object(goal1_root / "goal2-tuning-inputs.json")
    screening_by_k = {
        (
            int(item["corpus_user_count"]),
            str(item["scenario_id"]),
            int(item["requested_k"]),
        ): item
        for item in screening["cells"]
    }

    tuning_results: list[dict[str, Any]] = []
    scale_gates: list[dict[str, Any]] = []
    policy_gates: list[dict[str, Any]] = []
    for summary in ann_summaries:
        point_key = (
            int(summary["corpus_user_count"]),
            str(summary["scenario_id"]),
        )
        baselines = baseline_by_point[point_key]
        exact = baselines["exact_all"]
        filtered = baselines["filter_first_exact"]
        current = baselines["current_runtime"]
        fastest_exact = min(
            (exact, filtered), key=lambda item: float(item["p95_ms"])
        )
        hnsw = summary["hnsw"]
        assert isinstance(hnsw, Mapping)
        identity = (
            point_key[0],
            point_key[1],
            int(summary["requested_k"]),
            int(hnsw["ef_search"]),
            str(hnsw["iterative_scan"]),
            int(hnsw["max_scan_tuples"]),
        )
        diagnostic = diagnostics.get(identity)
        screening_row = screening_by_k[
            (point_key[0], point_key[1], int(summary["requested_k"]))
        ]
        exact_p95_ratio = float(summary["p95_ms"]) / float(exact["p95_ms"])
        exact_p99_ratio = float(summary["p99_ms"]) / float(exact["p99_ms"])
        fastest_exact_p95_ratio = (
            float(summary["p95_ms"]) / float(fastest_exact["p95_ms"])
        )
        fastest_exact_p99_ratio = (
            float(summary["p99_ms"]) / float(fastest_exact["p99_ms"])
        )
        current_p99_ratio = (
            float(summary["p99_ms"]) / float(current["p99_ms"])
        )
        exact_rss = exact.get("peak_rss_bytes")
        ann_rss = summary.get("peak_rss_bytes")
        rss_ratio = (
            int(ann_rss) / int(exact_rss)
            if ann_rss is not None and exact_rss not in (None, 0)
            else None
        )
        precision_passed = float(summary["precision"]) == 1.0
        worst_recall_passed = float(summary["recall"]) >= 0.95
        wilson_passed = float(summary["recall_lower_bound"]) >= 0.95
        p95_exact_passed = exact_p95_ratio <= 0.70
        p99_exact_passed = exact_p99_ratio <= 1.0
        rss_passed = rss_ratio is not None and rss_ratio <= 1.25
        oom_passed = summary["oom"] is False
        prediagnostic_scale_passed = all(
            (
                precision_passed,
                worst_recall_passed,
                wilson_passed,
                p95_exact_passed,
                p99_exact_passed,
                rss_passed,
                oom_passed,
            )
        )
        diagnostic_collected = diagnostic is not None
        index_used = (
            bool(diagnostic["hnsw_index_used"])
            if diagnostic is not None
            else None
        )
        temp_spill = (
            bool(diagnostic["temp_spill"])
            if diagnostic is not None
            else None
        )
        diagnostic_passed = bool(
            diagnostic_collected and index_used and temp_spill is False
        )
        scale_passed = prediagnostic_scale_passed and diagnostic_passed
        policy_latency_passed = all(
            (
                fastest_exact_p95_ratio <= 0.70,
                fastest_exact_p99_ratio <= 1.0,
                current_p99_ratio <= 1.0,
            )
        )
        policy_candidate = bool(
            scale_passed and policy_latency_passed
        )
        base = {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "cell_id": tuning_cell_id(identity),
            "corpus_user_count": point_key[0],
            "scenario_id": point_key[1],
            "candidate_type": summary["candidate_type"],
            "requested_k": int(summary["requested_k"]),
            "hnsw": dict(hnsw),
            "run_count": int(summary["run_count"]),
            "p50_ms": float(summary["p50_ms"]),
            "p95_ms": float(summary["p95_ms"]),
            "p99_ms": float(summary["p99_ms"]),
            "duration_samples_ms": summary["duration_samples_ms"],
            "precision": float(summary["precision"]),
            "worst_run_recall": float(summary["recall"]),
            "wilson_lower_bound": float(summary["recall_lower_bound"]),
            "peak_rss_bytes": ann_rss,
            "oom": bool(summary["oom"]),
            "exact_positive_count": int(summary["exact_positive_count"]),
            "final_user_count": int(summary["final_user_count"]),
            "exact_intersection": int(summary["intersection_count"]),
            "hard_match_ratio": float(summary["hard_match_ratio"]),
            "expected_member_ratio": float(summary["expected_member_ratio"]),
            "exact_all": _baseline_metrics(exact),
            "filter_first_exact": _baseline_metrics(filtered),
            "current_runtime": _baseline_metrics(current),
            "exact_all_p95_ratio": exact_p95_ratio,
            "exact_all_p99_ratio": exact_p99_ratio,
            "fastest_exact_plan": fastest_exact["plan"],
            "fastest_exact_p95_ratio": fastest_exact_p95_ratio,
            "fastest_exact_p99_ratio": fastest_exact_p99_ratio,
            "current_runtime_p99_ratio": current_p99_ratio,
            "exact_all_rss_ratio": rss_ratio,
            "screening_index_used": bool(screening_row["hnsw_index_used"]),
            "screening_temp_spill": bool(screening_row["temp_spill"]),
            "diagnostic_collected": diagnostic_collected,
            "hnsw_index_used": index_used,
            "temp_spill": temp_spill,
        }
        tuning_results.append(base)
        scale_gates.append(
            {
                **base,
                "precision_gate_passed": precision_passed,
                "worst_recall_gate_passed": worst_recall_passed,
                "wilson_gate_passed": wilson_passed,
                "p95_exact_all_gate_passed": p95_exact_passed,
                "p99_exact_all_gate_passed": p99_exact_passed,
                "rss_gate_passed": rss_passed,
                "oom_gate_passed": oom_passed,
                "prediagnostic_scale_gate_passed": prediagnostic_scale_passed,
                "diagnostic_gate_passed": diagnostic_passed,
                "ann_vs_exact_all_passed": scale_passed,
            }
        )
        policy_gates.append(
            {
                **base,
                "scale_quality_and_performance_passed": scale_passed,
                "fastest_exact_latency_gate_passed": policy_latency_passed,
                "ann_policy_candidate": policy_candidate,
                "policy_coverage_passed": False,
                "holdout_confirmation_passed": False,
                "exclusion_confirmation_passed": False,
                "cold_confirmation_passed": False,
                "policy_rule_eligible": False,
                "policy_ineligibility_reason": (
                    "not identifiable under current policy dimensions"
                ),
            }
        )

    diagnostic_plan = _build_diagnostic_plan(scale_gates)
    final_winners = _select_scale_winners(scale_gates)
    policy_candidates = [
        item for item in policy_gates if item["ann_policy_candidate"]
    ]
    return {
        "summaries": summaries,
        "tuning_results": sorted(tuning_results, key=_gate_sort_key),
        "scale_gates": sorted(scale_gates, key=_gate_sort_key),
        "policy_gates": sorted(policy_gates, key=_gate_sort_key),
        "diagnostic_plan": diagnostic_plan,
        "winners": final_winners,
        "counts": {
            "summary_count": len(summaries),
            "ann_tuning_result_count": len(tuning_results),
            "prediagnostic_scale_passed_count": sum(
                bool(item["prediagnostic_scale_gate_passed"])
                for item in scale_gates
            ),
            "ann_vs_exact_all_passed_count": sum(
                bool(item["ann_vs_exact_all_passed"])
                for item in scale_gates
            ),
            "ann_policy_candidate_count": len(policy_candidates),
            "policy_rule_eligible_count": 0,
            "scale_winner_count": len(final_winners),
            "diagnostic_plan_count": len(diagnostic_plan),
        },
    }


def _build_diagnostic_plan(
    scale_gates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in scale_gates:
        if not item["prediagnostic_scale_gate_passed"]:
            continue
        identity = _gate_identity(item)
        selected[identity] = {
            **_diagnostic_cell(item),
            "reason": "provisional_scale_winner",
        }
    by_point: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for item in scale_gates:
        by_point[(int(item["corpus_user_count"]), str(item["scenario_id"]))].append(item)
    for point, rows in by_point.items():
        has_provisional = any(
            bool(item["prediagnostic_scale_gate_passed"]) for item in rows
        )
        representative_pool = (
            [
                item
                for item in rows
                if not item["prediagnostic_scale_gate_passed"]
            ]
            if has_provisional
            else list(rows)
        )
        candidates = [
            item
            for item in representative_pool
            if item["precision_gate_passed"]
            and item["worst_recall_gate_passed"]
            and item["wilson_gate_passed"]
            and item["oom_gate_passed"]
        ]
        if not candidates:
            candidates = representative_pool
        representative = min(
            candidates,
            key=lambda item: (
                float(item["exact_all_p95_ratio"]),
                float(item["p95_ms"]),
                int(item["requested_k"]),
            ),
        )
        identity = _gate_identity(representative)
        selected.setdefault(
            identity,
            {
                **_diagnostic_cell(representative),
                "reason": "representative_adjacent_failure",
            },
        )
    return sorted(
        selected.values(),
        key=lambda item: (
            item["corpus_user_count"],
            item["scenario_id"],
            item["requested_k"],
            item["hnsw"]["ef_search"],
            item["hnsw"]["iterative_scan"],
            item["hnsw"]["max_scan_tuples"],
        ),
    )


def _select_scale_winners(
    scale_gates: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    grouped: dict[tuple[int, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for item in scale_gates:
        if item["ann_vs_exact_all_passed"]:
            grouped[
                (
                    int(item["corpus_user_count"]),
                    str(item["scenario_id"]),
                    int(item["requested_k"]),
                )
            ].append(item)
    return [
        min(rows, key=lambda item: (float(item["p95_ms"]), item["cell_id"]))
        for _, rows in sorted(grouped.items())
    ]


def _baseline_metrics(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return {
        "plan": value["plan"],
        "run_count": int(value["run_count"]),
        "p50_ms": float(value["p50_ms"]),
        "p95_ms": float(value["p95_ms"]),
        "p99_ms": float(value["p99_ms"]),
        "peak_rss_bytes": value["peak_rss_bytes"],
        "duration_samples_ms": list(value["duration_samples_ms"]),
        "precision": float(value["precision"]),
        "recall": float(value["recall"]),
        "temp_spill": bool(value["temp_spill"]),
        "oom": bool(value["oom"]),
    }


def _gate_identity(item: Mapping[str, Any]) -> tuple[Any, ...]:
    hnsw = item["hnsw"]
    return (
        int(item["corpus_user_count"]),
        str(item["scenario_id"]),
        int(item["requested_k"]),
        int(hnsw["ef_search"]),
        str(hnsw["iterative_scan"]),
        int(hnsw["max_scan_tuples"]),
    )


def _diagnostic_identity(item: Mapping[str, Any]) -> tuple[Any, ...]:
    return _gate_identity(item)


def _diagnostic_cell(item: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "cell_id": item["cell_id"],
        "corpus_user_count": int(item["corpus_user_count"]),
        "scenario_id": str(item["scenario_id"]),
        "candidate_type": str(item["candidate_type"]),
        "requested_k": int(item["requested_k"]),
        "hnsw": dict(item["hnsw"]),
    }


def _gate_sort_key(item: Mapping[str, Any]) -> tuple[Any, ...]:
    hnsw = item["hnsw"]
    return (
        int(item["corpus_user_count"]),
        str(item["scenario_id"]),
        int(item["requested_k"]),
        int(hnsw["ef_search"]),
        str(hnsw["iterative_scan"]),
        int(hnsw["max_scan_tuples"]),
    )
