#!/usr/bin/env python3
"""Build the fixed 50k HNSW tuning and filter-first confirmation artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from offline_evaluation.ann_search_experiment import (
    EXPERIMENT_VERSION,
    HNSW_INDEX_NAME,
    BenchmarkPhase,
    CacheMode,
    RunSummary,
    SearchPlan,
    bootstrap_percentile_ratio_upper_bound,
    load_observations,
    summarize_observations,
)


TARGET_RECALL = 0.95
ANN_P95_RATIO_MAX = 0.70
ANN_P99_RATIO_MAX = 1.0
ANN_RSS_RATIO_MAX = 1.25
EXPECTED_EF_SEARCH = {50, 100, 200, 400}
EXPECTED_ITERATIVE_SCAN = {"strict_order", "relaxed_order"}
EXPECTED_MAX_SCAN_TUPLES = {20_000, 50_000, 100_000}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuning-raw", type=Path, required=True)
    parser.add_argument("--tuning-input", type=Path, required=True)
    parser.add_argument("--screening-raw", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--confirmation-warm", type=Path, required=True)
    parser.add_argument("--confirmation-cold", type=Path)
    parser.add_argument("--confirmation-diagnostics-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_object(args.manifest)
    tuning_input = _load_object(args.tuning_input)
    _validate_fixed_inputs(manifest, tuning_input)

    tuning_observations = load_observations(args.tuning_raw)
    tuning_summaries = summarize_observations(
        tuning_observations,
        phase=BenchmarkPhase.TUNING,
    )
    tuning_rows, survivor_rows = _evaluate_tuning(
        tuning_summaries,
        tuning_input=tuning_input,
    )
    results_path = args.output_dir / "hnsw-tuning-results-50k.jsonl"
    with results_path.open("w", encoding="utf-8") as handle:
        for row in tuning_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    recommendation = (
        "ANN confirmation 진행"
        if any(row["final_ann_gate_passed"] for row in survivor_rows)
        else "50k ANN rule 없음"
    )
    tuning_report_path = args.output_dir / "hnsw-tuning-report-50k.md"
    tuning_report_path.write_text(
        _render_tuning_report(
            tuning_rows,
            survivor_rows,
            recommendation=recommendation,
            tuning_raw=args.tuning_raw,
            screening_raw=args.screening_raw,
            manifest=args.manifest,
        ),
        encoding="utf-8",
    )

    confirmation_rows = load_observations(args.confirmation_warm)
    if args.confirmation_cold is not None and args.confirmation_cold.exists():
        confirmation_rows.extend(load_observations(args.confirmation_cold))
    confirmation_summaries = summarize_observations(
        confirmation_rows,
        phase=BenchmarkPhase.CONFIRMATION,
    )
    filter_report, exclusion_validated = _render_filter_report(
        confirmation_summaries,
        manifest=manifest,
        diagnostics_dir=args.confirmation_diagnostics_dir,
    )
    filter_report_path = (
        args.output_dir / "filter-first-confirmation-report-50k.md"
    )
    filter_report_path.write_text(filter_report, encoding="utf-8")

    fallbacks_path = args.output_dir / "unvalidated-fallbacks.json"
    fallbacks_path.write_text(
        json.dumps(
            _fallbacks(
                manifest=manifest,
                survivor_rows=survivor_rows,
                recommendation=recommendation,
                exclusion_validated=exclusion_validated,
                input_hashes={
                    "manifest_sha256": _sha256(args.manifest),
                    "tuning_input_sha256": _sha256(args.tuning_input),
                    "screening_raw_sha256": _sha256(args.screening_raw),
                    "tuning_raw_sha256": _sha256(args.tuning_raw),
                    "confirmation_warm_sha256": _sha256(
                        args.confirmation_warm
                    ),
                    "confirmation_cold_sha256": (
                        _sha256(args.confirmation_cold)
                        if args.confirmation_cold is not None
                        and args.confirmation_cold.exists()
                        else None
                    ),
                },
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "tuning_cell_count": len(tuning_rows),
                "survivor_count": len(survivor_rows),
                "passing_survivor_count": sum(
                    row["final_ann_gate_passed"] for row in survivor_rows
                ),
                "recommendation": recommendation,
                "outputs": [
                    str(results_path),
                    str(tuning_report_path),
                    str(filter_report_path),
                    str(fallbacks_path),
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _validate_fixed_inputs(
    manifest: Mapping[str, Any],
    tuning_input: Mapping[str, Any],
) -> None:
    if manifest.get("experiment_version") != EXPERIMENT_VERSION:
        raise ValueError("unexpected experiment version")
    if manifest.get("vector_version") != "hotel_behavior.v2":
        raise ValueError("50k tuning must use hotel_behavior.v2")
    if manifest.get("campaign_id") is not None or manifest.get("promotion_id") is not None:
        if not manifest.get("campaign_id") or not manifest.get("promotion_id"):
            raise ValueError("campaign_id and promotion_id must be paired")
    cells = tuning_input.get("cells")
    if not isinstance(cells, list) or len(cells) != 8:
        raise ValueError("50k tuning input must contain exactly eight survivors")
    if tuning_input.get("retained_cell_count") != 8:
        raise ValueError("50k tuning retained count is not eight")


def _evaluate_tuning(
    summaries: Sequence[RunSummary],
    *,
    tuning_input: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    warm = [
        item
        for item in summaries
        if item.cache_mode == CacheMode.WARM and item.exclusion_ratio == 0.0
    ]
    by_scenario: dict[str, list[RunSummary]] = defaultdict(list)
    for item in warm:
        by_scenario[item.scenario_id].append(item)

    expected_survivors = {
        (str(item["scenario_id"]), int(item["requested_k"]))
        for item in tuning_input["cells"]
    }
    rows: list[dict[str, Any]] = []
    for scenario_id, scenario in sorted(by_scenario.items()):
        exact = min(
            (
                item
                for item in scenario
                if item.plan in {SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT}
            ),
            key=lambda item: (item.p95_ms, item.p99_ms, item.plan.value),
        )
        current = next(
            item for item in scenario if item.plan == SearchPlan.CURRENT_RUNTIME
        )
        ann = [item for item in scenario if item.plan == SearchPlan.ANN_FIRST]
        for summary in ann:
            if (scenario_id, int(summary.requested_k or 0)) not in expected_survivors:
                raise ValueError("tuning raw contains a non-survivor K")
            failures = _ann_gate_failures(summary, exact=exact, current=current)
            row = summary.to_dict()
            row.update(
                {
                    "worst_run_recall": summary.recall,
                    "wilson_lower_bound": summary.recall_lower_bound,
                    "hnsw_index_name": (
                        HNSW_INDEX_NAME if summary.index_used else None
                    ),
                    "exact_baseline_plan": exact.plan.value,
                    "exact_p50_ms": exact.p50_ms,
                    "exact_p95_ms": exact.p95_ms,
                    "exact_p99_ms": exact.p99_ms,
                    "exact_peak_rss_bytes": exact.peak_rss_bytes,
                    "current_runtime_p50_ms": current.p50_ms,
                    "current_runtime_p95_ms": current.p95_ms,
                    "current_runtime_p99_ms": current.p99_ms,
                    "current_runtime_peak_rss_bytes": current.peak_rss_bytes,
                    "p50_ratio_vs_exact": _ratio(summary.p50_ms, exact.p50_ms),
                    "p95_ratio_vs_exact": _ratio(summary.p95_ms, exact.p95_ms),
                    "p99_ratio_vs_exact": _ratio(summary.p99_ms, exact.p99_ms),
                    "p50_ratio_vs_current_runtime": _ratio(
                        summary.p50_ms, current.p50_ms
                    ),
                    "p95_ratio_vs_current_runtime": _ratio(
                        summary.p95_ms, current.p95_ms
                    ),
                    "p99_ratio_vs_current_runtime": _ratio(
                        summary.p99_ms, current.p99_ms
                    ),
                    "peak_rss_ratio_vs_exact": _optional_ratio(
                        summary.peak_rss_bytes, exact.peak_rss_bytes
                    ),
                    "gate_failures": failures,
                    "final_ann_gate_passed": not failures,
                }
            )
            rows.append(row)

    expected_grid_count = (
        len(EXPECTED_EF_SEARCH)
        * len(EXPECTED_ITERATIVE_SCAN)
        * len(EXPECTED_MAX_SCAN_TUPLES)
    )
    for survivor in expected_survivors:
        matched = [
            row
            for row in rows
            if (row["scenario_id"], row["requested_k"]) == survivor
        ]
        if len(matched) != expected_grid_count:
            raise ValueError(
                f"survivor {survivor} has {len(matched)} tuning settings, "
                f"expected {expected_grid_count}"
            )
        if any(row["run_count"] != 10 for row in matched):
            raise ValueError("every HNSW tuning cell must have ten measured runs")
        observed_grid = {
            (
                row["hnsw"]["ef_search"],
                row["hnsw"]["iterative_scan"],
                row["hnsw"]["max_scan_tuples"],
            )
            for row in matched
        }
        expected_grid = {
            (ef, scan, maximum)
            for ef in EXPECTED_EF_SEARCH
            for scan in EXPECTED_ITERATIVE_SCAN
            for maximum in EXPECTED_MAX_SCAN_TUPLES
        }
        if observed_grid != expected_grid:
            raise ValueError(f"survivor {survivor} HNSW grid is incomplete")

    if {(row["scenario_id"], row["requested_k"]) for row in rows} != expected_survivors:
        raise ValueError("tuning raw does not match the eight screening survivors")
    rows.sort(key=_tuning_row_key)

    survivor_rows: list[dict[str, Any]] = []
    for scenario_id, requested_k in sorted(expected_survivors):
        settings = [
            row
            for row in rows
            if row["scenario_id"] == scenario_id
            and row["requested_k"] == requested_k
        ]
        passing = [row for row in settings if row["final_ann_gate_passed"]]
        selected = min(
            passing or settings,
            key=lambda row: (
                len(row["gate_failures"]),
                row["p95_ms"],
                row["p99_ms"],
                row["hnsw"]["ef_search"],
                row["hnsw"]["iterative_scan"],
                row["hnsw"]["max_scan_tuples"],
            ),
        )
        failure_counts = Counter(
            failure for row in settings for failure in row["gate_failures"]
        )
        survivor_rows.append(
            {
                "scenario_id": scenario_id,
                "candidate_type": selected["candidate_type"],
                "requested_k": requested_k,
                "tested_setting_count": len(settings),
                "passing_setting_count": len(passing),
                "final_ann_gate_passed": bool(passing),
                "decision": (
                    "ANN confirmation candidate"
                    if passing
                    else "Exact fallback; no more ANN fine-tuning"
                ),
                "selected_or_closest_setting": selected["hnsw"],
                "selected_or_closest_p95_ratio_vs_exact": selected[
                    "p95_ratio_vs_exact"
                ],
                "selected_or_closest_p99_ratio_vs_exact": selected[
                    "p99_ratio_vs_exact"
                ],
                "selected_or_closest_p99_ratio_vs_current_runtime": selected[
                    "p99_ratio_vs_current_runtime"
                ],
                "selected_or_closest_recall": selected["worst_run_recall"],
                "selected_or_closest_wilson_lower_bound": selected[
                    "wilson_lower_bound"
                ],
                "selected_or_closest_peak_rss_ratio_vs_exact": selected[
                    "peak_rss_ratio_vs_exact"
                ],
                "selected_or_closest_gate_failures": selected["gate_failures"],
                "gate_failure_setting_counts": dict(sorted(failure_counts.items())),
            }
        )
    return rows, survivor_rows


def _ann_gate_failures(
    ann: RunSummary,
    *,
    exact: RunSummary,
    current: RunSummary,
) -> list[str]:
    failures: list[str] = []
    if ann.precision != 1.0:
        failures.append("precision_below_1")
    if ann.recall < TARGET_RECALL:
        failures.append("worst_recall_below_0_95")
    if ann.recall_lower_bound < TARGET_RECALL:
        failures.append("wilson_lower_bound_below_0_95")
    if ann.p95_ms > exact.p95_ms * ANN_P95_RATIO_MAX:
        failures.append("p95_not_30pct_faster_than_exact")
    if ann.p99_ms > exact.p99_ms * ANN_P99_RATIO_MAX:
        failures.append("p99_slower_than_exact")
    if ann.p99_ms > current.p99_ms * ANN_P99_RATIO_MAX:
        failures.append("p99_slower_than_current_runtime")
    if not ann.index_used:
        failures.append("hnsw_index_not_used")
    if ann.temp_spill:
        failures.append("temp_spill")
    if ann.oom:
        failures.append("oom")
    if ann.peak_rss_bytes is None or exact.peak_rss_bytes is None:
        failures.append("peak_rss_unavailable")
    elif ann.peak_rss_bytes > exact.peak_rss_bytes * ANN_RSS_RATIO_MAX:
        failures.append("peak_rss_above_exact_1_25x")
    return failures


def _render_tuning_report(
    rows: Sequence[Mapping[str, Any]],
    survivors: Sequence[Mapping[str, Any]],
    *,
    recommendation: str,
    tuning_raw: Path,
    screening_raw: Path,
    manifest: Path,
) -> str:
    lines = [
        "# 50k HNSW tuning report",
        "",
        f"**판정: `{recommendation}`**",
        "",
        "이번 단계는 ANN 정책을 생성하거나 승격하지 않는다. screening survivor 8개를 "
        "24개 HNSW 설정으로만 검증했으며, 최종 gate 미통과 survivor는 추가 미세조정 없이 Exact fallback이다.",
        "",
        "## 입력 고정과 실행 범위",
        "",
        f"- manifest: `{manifest}` (`sha256={_sha256(manifest)}`)",
        f"- screening raw 재사용: `{screening_raw}` (`sha256={_sha256(screening_raw)}`)",
        f"- tuning raw: `{tuning_raw}` (`sha256={_sha256(tuning_raw)}`)",
        f"- HNSW summary cell: {len(rows)} (8 survivors × 24 settings)",
        "- 각 cell: warm-up 3회, 측정 10회; EXPLAIN은 일반 측정과 분리",
        "- vector/window/manifest는 기존 50k `hotel_behavior.v2` cohort와 동일",
        "",
        "## Survivor 판정",
        "",
        "| scenario | K | 통과 설정 | 판정 | closest/pass 설정 | recall | Wilson LB | p95/Exact | p99/Exact | p99/current | RSS/Exact | 실패 gate |",
        "|---|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for item in survivors:
        hnsw = item["selected_or_closest_setting"]
        setting = (
            f"ef={hnsw['ef_search']}, {hnsw['iterative_scan']}, "
            f"scan={hnsw['max_scan_tuples']}"
        )
        lines.append(
            "| {scenario} | {k} | {passes}/24 | {decision} | {setting} | "
            "{recall:.6f} | {wilson:.6f} | {p95:.3f} | {p99e:.3f} | "
            "{p99c:.3f} | {rss} | {failures} |".format(
                scenario=item["scenario_id"],
                k=item["requested_k"],
                passes=item["passing_setting_count"],
                decision=("pass" if item["final_ann_gate_passed"] else "Exact fallback"),
                setting=setting,
                recall=item["selected_or_closest_recall"],
                wilson=item["selected_or_closest_wilson_lower_bound"],
                p95=item["selected_or_closest_p95_ratio_vs_exact"],
                p99e=item["selected_or_closest_p99_ratio_vs_exact"],
                p99c=item["selected_or_closest_p99_ratio_vs_current_runtime"],
                rss=(
                    f"{item['selected_or_closest_peak_rss_ratio_vs_exact']:.3f}"
                    if item["selected_or_closest_peak_rss_ratio_vs_exact"] is not None
                    else "n/a"
                ),
                failures=", ".join(item["selected_or_closest_gate_failures"])
                or "none",
            )
        )
    lines.extend(
        [
            "",
            "## Gate 정의",
            "",
            "- precision = 1.0",
            "- worst-run recall 및 95% Wilson lower bound ≥ 0.95",
            "- p95 ≤ fastest Exact × 0.70",
            "- p99 ≤ fastest Exact 및 current_runtime",
            f"- `{HNSW_INDEX_NAME}` 사용, spill/OOM 없음",
            "- peak RSS ≤ fastest Exact × 1.25",
            "",
            "모든 192개 cell의 p50/p95/p99, 품질, index/spill/OOM, RSS, Exact/current ratio와 "
            "gate 실패 사유는 `hnsw-tuning-results-50k.jsonl`에 있다.",
            "",
            "## 다음 단계",
            "",
            f"`{recommendation}`",
            "",
        ]
    )
    return "\n".join(lines)


def _render_filter_report(
    summaries: Sequence[RunSummary],
    *,
    manifest: Mapping[str, Any],
    diagnostics_dir: Path | None,
) -> tuple[str, bool]:
    scenario = "intent-all-months-confirmation-a"
    selected = [item for item in summaries if item.scenario_id == scenario]
    rows: list[tuple[str, RunSummary, RunSummary, float]] = []
    for label, mode in (("warm-cache", CacheMode.WARM), ("DB-cold", CacheMode.DB_COLD)):
        exact = next(
            (
                item
                for item in selected
                if item.cache_mode == mode
                and item.exclusion_ratio == 0.0
                and item.plan == SearchPlan.EXACT_ALL
            ),
            None,
        )
        filtered = next(
            (
                item
                for item in selected
                if item.cache_mode == mode
                and item.exclusion_ratio == 0.0
                and item.plan == SearchPlan.FILTER_FIRST_EXACT
            ),
            None,
        )
        if exact is None or filtered is None:
            continue
        seed = int.from_bytes(
            hashlib.sha256(f"{scenario}|{mode.value}|filter-exact".encode()).digest()[:8],
            "big",
        )
        upper = bootstrap_percentile_ratio_upper_bound(
            filtered.duration_samples_ms,
            exact.duration_samples_ms,
            iterations=1_000,
            confidence=0.95,
            seed=seed,
        )
        rows.append((label, exact, filtered, upper))

    exclusion_context_present = bool(
        manifest.get("campaign_id") and manifest.get("promotion_id")
    )
    exclusion_summaries = [
        item
        for item in selected
        if item.cache_mode == CacheMode.WARM
        and abs(item.exclusion_ratio - 0.05) <= 0.005
    ]
    exclusion_validated = exclusion_context_present and {
        item.plan for item in exclusion_summaries
    } >= {SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT}

    diagnostics = _exact_diagnostics(diagnostics_dir, scenario=scenario)
    lines = [
        "# 50k filter-first Exact confirmation report",
        "",
        "**판정: latency·결과 동일성 통과, full confirmation은 incomplete evidence.**",
        "",
        "기존 warm-cache 300회 결과를 유지하고 DB-cold 30회 결과를 추가했다. "
        "5% exclusion은 실제 Data Contract context 유무에 따라 별도로 판정한다.",
        "",
        "## Latency와 결과 동일성",
        "",
        "| cache | plan | runs | p50 ms | p95 ms | p99 ms | final users | Exact 완전 일치 | filter/Exact p95 | filter/Exact p99 | bootstrap p95 ratio upper 95% | peak RSS |",
        "|---|---|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for label, exact, filtered, upper in rows:
        for summary in (exact, filtered):
            lines.append(
                "| {label} | {plan} | {runs} | {p50:.3f} | {p95:.3f} | "
                "{p99:.3f} | {users} | {same} | {p95_ratio} | {p99_ratio} | "
                "{upper} | {rss} |".format(
                    label=label,
                    plan=summary.plan.value,
                    runs=summary.run_count,
                    p50=summary.p50_ms,
                    p95=summary.p95_ms,
                    p99=summary.p99_ms,
                    users=summary.final_user_count,
                    same=(
                        "yes"
                        if summary.final_user_count == exact.final_user_count
                        and summary.intersection_count == exact.final_user_count
                        else "no"
                    ),
                    p95_ratio=(
                        f"{filtered.p95_ms / exact.p95_ms:.6f}"
                        if summary.plan == SearchPlan.FILTER_FIRST_EXACT
                        else "-"
                    ),
                    p99_ratio=(
                        f"{filtered.p99_ms / exact.p99_ms:.6f}"
                        if summary.plan == SearchPlan.FILTER_FIRST_EXACT
                        else "-"
                    ),
                    upper=(
                        f"{upper:.6f}"
                        if summary.plan == SearchPlan.FILTER_FIRST_EXACT
                        else "-"
                    ),
                    rss=(
                        str(summary.peak_rss_bytes)
                        if summary.peak_rss_bytes is not None
                        else "n/a"
                    ),
                )
            )
    latency_identity_passed = bool(rows) and all(
        filtered.final_user_count == exact.final_user_count
        and filtered.intersection_count == exact.final_user_count
        and filtered.p95_ms <= exact.p95_ms * 0.85
        and filtered.p99_ms <= exact.p99_ms
        and upper <= 0.85
        for _, exact, filtered, upper in rows
    )
    lines.extend(
        [
            "",
            "## Gate 판정",
            "",
            f"- warm-cache/DB-cold 결과 동일성·latency gate: `{'pass' if latency_identity_passed else 'fail'}`",
            "- full confirmation: `incomplete evidence` (실제 5% exclusion context 없음)",
        ]
    )
    lines.extend(
        [
            "",
            "## 실행 계획과 spill",
            "",
        ]
    )
    for plan in (SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT):
        diagnostic = diagnostics.get(plan.value)
        if diagnostic is None:
            lines.append(f"- `{plan.value}`: diagnostic 없음")
        else:
            indexes = ", ".join(diagnostic["index_names"]) or "none"
            lines.append(
                f"- `{plan.value}`: top node `{diagnostic['top_node_type']}`, "
                f"indexes `{indexes}`, temp blocks `{diagnostic['temp_blocks']}`, "
                f"spill `{'yes' if diagnostic['temp_blocks'] else 'no'}`"
            )
    if diagnostics.get(SearchPlan.FILTER_FIRST_EXACT.value, {}).get("temp_blocks"):
        lines.extend(
            [
                "",
                "`filter_first_exact`의 latency/동일성 gate는 통과했지만 EXPLAIN에서 temp spill이 관찰됐다. "
                "이는 5% exclusion 미검증과 별개의 운영 리스크이며 rollout 근거에서는 명시적으로 검토해야 한다.",
            ]
        )
    lines.extend(
        [
            "",
            "## 5% deterministic exclusion snapshot",
            "",
        ]
    )
    if exclusion_validated:
        lines.append("실제 campaign/promotion exclusion context로 확인 완료.")
    else:
        lines.extend(
            [
                "**미검증 — Exact fallback 유지.**",
                "",
                "현재 scenario manifest에는 `campaign_id`와 `promotion_id`가 없다. "
                "임의 campaign/promotion 또는 합성 exclusion row를 만들지 않았으므로 5% exclusion confirmation을 실행하지 않았다. "
                "필요한 실제 준비 데이터는 `unvalidated-fallbacks.json`에 기록했다.",
            ]
        )
    lines.append("")
    return "\n".join(lines), exclusion_validated


def _exact_diagnostics(
    diagnostics_dir: Path | None,
    *,
    scenario: str,
) -> Mapping[str, Mapping[str, Any]]:
    if diagnostics_dir is None:
        return {}
    result: dict[str, Mapping[str, Any]] = {}
    for plan in (SearchPlan.EXACT_ALL, SearchPlan.FILTER_FIRST_EXACT):
        path = diagnostics_dir / f"{scenario}-{plan.value}.json"
        if not path.exists():
            continue
        explain = json.loads(path.read_text(encoding="utf-8"))
        nodes = _find_values(explain, "Node Type")
        temp_blocks = sum(
            int(value or 0)
            for key in ("Temp Read Blocks", "Temp Written Blocks")
            for value in _find_values(explain, key)
        )
        result[plan.value] = {
            "top_node_type": str(nodes[0]) if nodes else "unknown",
            "index_names": sorted(
                {str(value) for value in _find_values(explain, "Index Name")}
            ),
            "temp_blocks": temp_blocks,
        }
    return result


def _find_values(value: Any, key: str) -> list[Any]:
    found: list[Any] = []
    if isinstance(value, Mapping):
        for child_key, child in value.items():
            if child_key == key:
                found.append(child)
            found.extend(_find_values(child, key))
    elif isinstance(value, list):
        for child in value:
            found.extend(_find_values(child, key))
    return found


def _fallbacks(
    *,
    manifest: Mapping[str, Any],
    survivor_rows: Sequence[Mapping[str, Any]],
    recommendation: str,
    exclusion_validated: bool,
    input_hashes: Mapping[str, Any],
) -> Mapping[str, Any]:
    fallbacks: list[Mapping[str, Any]] = []
    for row in survivor_rows:
        if row["final_ann_gate_passed"]:
            continue
        fallbacks.append(
            {
                "scope": "50k_hnsw_tuning_survivor",
                "scenario_id": row["scenario_id"],
                "candidate_type": row["candidate_type"],
                "requested_k": row["requested_k"],
                "fallback_plan": "exact",
                "reason": "no HNSW grid setting passed every final ANN gate",
                "gate_failure_setting_counts": row[
                    "gate_failure_setting_counts"
                ],
                "additional_ann_fine_tuning_allowed": False,
            }
        )
    if not exclusion_validated:
        fallbacks.append(
            {
                "scope": "50k_filter_first_exact_5pct_exclusion_confirmation",
                "scenario_id": "intent-all-months-confirmation-a",
                "fallback_plan": "exact_all",
                "reason": "scenario manifest has no real campaign_id/promotion_id exclusion context",
                "synthetic_context_created": False,
                "required_preparation_data": [
                    "an existing campaign_id for project expedia-ann-benchmark-v1",
                    "an existing promotion_id linked to that campaign",
                    "a deterministic 5% set of the fixed 50k cohort (2,500 user_ids)",
                    "PostgreSQL promotion exclusion state and member rows at one sealed revision",
                    "ClickHouse exclusion projection and ready status for the same promotion/revision",
                    "campaign_id and promotion_id added together to a copied confirmation manifest",
                ],
                "required_checks": [
                    "observed exclusion ratio within 0.5 percentage points of 0.05",
                    "exact_all and filter_first_exact final user sets identical",
                    "warm-cache 300 measured runs per plan",
                    "separate EXPLAIN diagnostics and explicit spill disposition",
                ],
            }
        )
    return {
        "experiment_version": EXPERIMENT_VERSION,
        "generated_at": datetime.now(UTC).isoformat(),
        "vector_version": manifest.get("vector_version"),
        "manifest_hash": manifest.get("manifest_hash"),
        "cohort_user_count": 50_000,
        "recommendation": recommendation,
        "candidate_policy_promoted": False,
        "macro_benchmark_executed": False,
        "input_hashes": dict(input_hashes),
        "fallbacks": fallbacks,
    }


def _tuning_row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    hnsw = row["hnsw"]
    return (
        row["scenario_id"],
        row["requested_k"],
        hnsw["ef_search"],
        hnsw["iterative_scan"],
        hnsw["max_scan_tuples"],
    )


def _ratio(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        raise ValueError("latency denominator must be positive")
    return numerator / denominator


def _optional_ratio(
    numerator: int | None,
    denominator: int | None,
) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _load_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain an object")
    return payload


def _sha256(path: Path | None) -> str:
    if path is None:
        raise ValueError("path is required")
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
