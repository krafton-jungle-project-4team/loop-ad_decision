#!/usr/bin/env python3
"""Run checkpointed Goal 1 measurements for ANN benchmark v2 cohorts.

The runner deliberately keeps every cohort and scenario in a distinct file.  A
completed file is validated and reused; an interrupted command only leaves a
``.partial`` file which is rebuilt on resume.  The historical 50K pilot output
root is never read or written.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    sha256_file,
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_EXPERIMENT_VERSION,
    SCALE_OUTPUT_ROOT,
    ScenarioSet,
    confirmation_baseline_k,
)
from scripts.prepare_ann_scale_postgres import (  # noqa: E402
    cohort_database_name,
    positive_ints,
)


DEFAULT_COHORTS = (50_000, 100_000, 250_000, 500_000, 750_000, 1_000_000)
DEFAULT_SERIES_ID = "expedia-full-2015-v2-86ded0d7"
DEFAULT_SAMPLE_SEED = "ann-score-pass-v2-20260718"
ALL_PLANS = (
    "current_runtime",
    "exact_all",
    "filter_first_exact",
    "ann_first",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("preflight", "measure", "all")
    )
    parser.add_argument("--env-file", type=Path, default=Path(".env.ann-source.local"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=SCALE_OUTPUT_ROOT / "phase2/selected-scenario-manifest.json",
    )
    parser.add_argument("--output-root", type=Path, default=SCALE_OUTPUT_ROOT)
    parser.add_argument("--clickhouse-database", default="ann_scale_v2_build")
    parser.add_argument("--postgres-prefix", default="loopad_ann_v2")
    parser.add_argument(
        "--postgres-container",
        default="loop-ad_data-source_contract-postgres-1",
    )
    parser.add_argument("--scale-series-id", default=DEFAULT_SERIES_ID)
    parser.add_argument("--reference-sample-seed", default=DEFAULT_SAMPLE_SEED)
    parser.add_argument(
        "--cohort-sizes",
        type=positive_ints,
        default=DEFAULT_COHORTS,
    )
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=10)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _require_v2_output_root(args.output_root)
    if args.warmups < 0 or args.repetitions <= 0:
        raise ValueError("warmups/repetitions are invalid")
    cohorts = tuple(args.cohort_sizes)
    manifest = _object_at(args.manifest)
    _require_v2_manifest(manifest)
    if args.command in {"preflight", "all"}:
        run_preflights(args, cohorts=cohorts)
    if args.command in {"measure", "all"}:
        require_preflights(args, cohorts=DEFAULT_COHORTS)
        run_measurements(args, manifest=manifest, cohorts=cohorts)
    return 0


def run_preflights(args: argparse.Namespace, *, cohorts: Sequence[int]) -> None:
    for size in cohorts:
        output = (
            args.output_root
            / "phase3/preflight-scale-ready"
            / f"cohort-{size}.json"
        )
        if output.exists():
            payload = _object_at(output)
            _validate_preflight(payload, size=size, args=args)
            _progress("preflight_resumed", size, output)
            continue
        _progress("preflight_started", size, output)
        completed = _run_cli(
            args,
            size=size,
            arguments=("preflight", "--manifest", str(args.manifest)),
            capture_json=True,
        )
        assert isinstance(completed, Mapping)
        _validate_preflight(completed, size=size, args=args)
        write_immutable_json(output, completed)
        _progress("preflight_completed", size, output)


def require_preflights(args: argparse.Namespace, *, cohorts: Sequence[int]) -> None:
    missing: list[int] = []
    for size in cohorts:
        output = (
            args.output_root
            / "phase3/preflight-scale-ready"
            / f"cohort-{size}.json"
        )
        if not output.exists():
            missing.append(size)
            continue
        _validate_preflight(_object_at(output), size=size, args=args)
    if missing:
        raise RuntimeError(
            "all scale-series cohorts must pass preflight before measurement: "
            + ",".join(map(str, missing))
        )


def run_measurements(
    args: argparse.Namespace,
    *,
    manifest: Mapping[str, Any],
    cohorts: Sequence[int],
) -> None:
    scenarios = tuple(_scenario_records(manifest))
    for size in cohorts:
        started = time.monotonic()
        cohort_root = args.output_root / "phase4" / f"cohort-{size}"
        _progress("cohort_measurement_started", size, cohort_root)
        summaries: dict[str, Mapping[str, Any]] = {}
        for scenario in scenarios:
            summaries[str(scenario["scenario_id"])] = _ground_truth(
                args,
                size=size,
                scenario=scenario,
                cohort_root=cohort_root,
            )
        for scenario in scenarios:
            scenario_set = ScenarioSet(str(scenario["scenario_set"]))
            if scenario_set == ScenarioSet.TUNING:
                requested_k: int | None = None
                phase = "screening"
            else:
                requested_k = confirmation_baseline_k(
                    corpus_user_count=size,
                    estimated_member_count=float(
                        summaries[str(scenario["scenario_id"])][
                            "estimated_member_count"
                        ]
                    ),
                )
                phase = "confirmation"
            _run_scenario_measurement(
                args,
                size=size,
                scenario=scenario,
                phase=phase,
                requested_k=requested_k,
                cohort_root=cohort_root,
            )
        elapsed = time.monotonic() - started
        _progress(
            "cohort_measurement_completed",
            size,
            cohort_root,
            elapsed_seconds=round(elapsed, 3),
        )


def _ground_truth(
    args: argparse.Namespace,
    *,
    size: int,
    scenario: Mapping[str, Any],
    cohort_root: Path,
) -> Mapping[str, Any]:
    scenario_id = str(scenario["scenario_id"])
    output = cohort_root / "ground-truth" / f"{scenario_id}.csv.gz"
    summary = output.with_suffix("").with_suffix(".summary.json")
    if output.exists() and summary.exists():
        payload = _object_at(summary)
        _validate_ground_truth(payload, output=output, size=size, scenario_id=scenario_id)
        _progress("ground_truth_resumed", size, output, scenario_id=scenario_id)
        return payload
    partial = output.with_name(f"{scenario_id}.partial.csv.gz")
    partial_summary = partial.with_suffix("").with_suffix(".summary.json")
    _progress("ground_truth_started", size, output, scenario_id=scenario_id)
    completed = _run_cli(
        args,
        size=size,
        arguments=(
            "ground-truth",
            "--manifest",
            str(args.manifest),
            "--scenario-id",
            scenario_id,
            "--reference-sample-size",
            "50000",
            "--output",
            str(partial),
        ),
        capture_json=True,
    )
    assert isinstance(completed, Mapping)
    generated = _object_at(partial_summary)
    if dict(completed) != dict(generated):
        raise RuntimeError("ground-truth stdout differs from generated summary")
    _validate_ground_truth(
        completed, output=partial, size=size, scenario_id=scenario_id
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output)
    rewritten = {**dict(completed), "ground_truth_csv_gz": str(output)}
    partial_summary.unlink(missing_ok=True)
    write_immutable_json(summary, rewritten)
    _progress("ground_truth_completed", size, output, scenario_id=scenario_id)
    return rewritten


def _run_scenario_measurement(
    args: argparse.Namespace,
    *,
    size: int,
    scenario: Mapping[str, Any],
    phase: str,
    requested_k: int | None,
    cohort_root: Path,
) -> None:
    scenario_id = str(scenario["scenario_id"])
    scenario_set = str(scenario["scenario_set"])
    output = cohort_root / "raw" / scenario_set / f"{scenario_id}.jsonl"
    if output.exists():
        _validate_observations(
            output,
            size=size,
            scenario_id=scenario_id,
            warmups=args.warmups,
            repetitions=args.repetitions,
            confirmation=requested_k is not None,
            requested_k=requested_k,
        )
        _progress("measurement_resumed", size, output, scenario_id=scenario_id)
        return
    partial = output.with_name(f"{scenario_id}.partial.jsonl")
    diagnostics = cohort_root / "diagnostics" / scenario_set / scenario_id
    arguments: list[str] = [
        "run-cell",
        "--manifest",
        str(args.manifest),
        "--scenario-id",
        scenario_id,
        "--phase",
        phase,
        "--cache-mode",
        "warm",
        "--warmups",
        str(args.warmups),
        "--repetitions",
        str(args.repetitions),
        "--sample-sizes",
        "50000",
        "--hnsw-grid",
        "baseline",
        "--output",
        str(partial),
        "--diagnostics-dir",
        str(diagnostics),
        "--confirm-disposable-postgres",
    ]
    for plan in ALL_PLANS:
        arguments.extend(("--plan", plan))
    if requested_k is not None:
        arguments.extend(("--requested-k", str(requested_k)))
    _progress(
        "measurement_started",
        size,
        output,
        scenario_id=scenario_id,
        scenario_set=scenario_set,
        requested_k=requested_k,
    )
    _run_cli(args, size=size, arguments=arguments, capture_json=False)
    _validate_observations(
        partial,
        size=size,
        scenario_id=scenario_id,
        warmups=args.warmups,
        repetitions=args.repetitions,
        confirmation=requested_k is not None,
        requested_k=requested_k,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, output)
    _progress(
        "measurement_completed",
        size,
        output,
        scenario_id=scenario_id,
        scenario_set=scenario_set,
        requested_k=requested_k,
    )


def _run_cli(
    args: argparse.Namespace,
    *,
    size: int,
    arguments: Sequence[str],
    capture_json: bool,
) -> Mapping[str, Any] | None:
    environment = {
        **os.environ,
        "LOOPAD_AURORA_DATABASE": cohort_database_name(args.postgres_prefix, size),
        "LOOPAD_CLICKHOUSE_DATABASE": args.clickhouse_database,
        "ANN_POSTGRES_CONTAINER": args.postgres_container,
    }
    scope = (
        "--env-file",
        str(args.env_file),
        "--scale-series-id",
        args.scale_series_id,
        "--scale-cohort-size",
        str(size),
        "--reference-sample-seed",
        args.reference_sample_seed,
    )
    command = (
        sys.executable,
        str(ROOT / "scripts/benchmark_audience_search.py"),
        arguments[0],
        *scope,
        *arguments[1:],
    )
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "benchmark command failed\n"
            + "command="
            + " ".join(command[:3])
            + " ...\nstdout:\n"
            + completed.stdout[-8_000:]
            + "\nstderr:\n"
            + completed.stderr[-8_000:]
        )
    if not capture_json:
        return None
    payload = json.loads(completed.stdout)
    if not isinstance(payload, Mapping):
        raise RuntimeError("benchmark command did not return a JSON object")
    return payload


def _validate_preflight(
    payload: Mapping[str, Any], *, size: int, args: argparse.Namespace
) -> None:
    if int(payload.get("corpus_user_count", -1)) != size:
        raise RuntimeError("preflight PostgreSQL cohort size mismatch")
    if int(payload.get("cohort_membership_count", -1)) != size:
        raise RuntimeError("preflight ClickHouse membership size mismatch")
    if int(payload.get("cohort_signal_count", -1)) != size:
        raise RuntimeError("preflight ClickHouse frozen signal size mismatch")
    if payload.get("scale_series_id") != args.scale_series_id:
        raise RuntimeError("preflight scale-series identity mismatch")
    index = payload.get("hnsw_index")
    if not isinstance(index, Mapping):
        raise RuntimeError("preflight HNSW metadata missing")
    if not bool(index.get("indisvalid")) or not bool(index.get("indisready")):
        raise RuntimeError("preflight HNSW index is not valid and ready")


def _validate_ground_truth(
    payload: Mapping[str, Any], *, output: Path, size: int, scenario_id: str
) -> None:
    if payload.get("experiment_version") != SCALE_EXPERIMENT_VERSION:
        raise RuntimeError("ground truth experiment version mismatch")
    if payload.get("scenario_id") != scenario_id:
        raise RuntimeError("ground truth scenario mismatch")
    if int(payload.get("corpus_user_count", -1)) != size:
        raise RuntimeError("ground truth is incomplete")
    if int(payload.get("reference_sample_size", -1)) != min(size, 50_000):
        raise RuntimeError("ground truth reference sample is not fixed at 50K")
    if not output.exists() or output.stat().st_size <= 0:
        raise RuntimeError("ground truth gzip is missing or empty")
    with gzip.open(output, "rt", encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n")
        if header != (
            "cosine_rank,user_id,behavior_fit_score,score_pass,final_positive"
        ):
            raise RuntimeError("ground truth CSV header is invalid")
        row_count = sum(1 for _ in handle)
    if row_count != size:
        raise RuntimeError("ground truth gzip row count is incomplete")


def _validate_observations(
    path: Path,
    *,
    size: int,
    scenario_id: str,
    warmups: int,
    repetitions: int,
    confirmation: bool,
    requested_k: int | None,
) -> None:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError("measurement JSONL is missing or empty")
    identities: Counter[tuple[Any, ...]] = Counter()
    plan_ks: set[tuple[str, int | None]] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        payload = json.loads(line)
        if not isinstance(payload, Mapping):
            raise RuntimeError(f"observation {line_number} is not an object")
        if payload.get("experiment_version") != SCALE_EXPERIMENT_VERSION:
            raise RuntimeError("observation experiment version mismatch")
        if payload.get("scenario_id") != scenario_id:
            raise RuntimeError("observation scenario mismatch")
        if int(payload.get("corpus_user_count", -1)) != size:
            raise RuntimeError("observation cohort mismatch")
        plan = str(payload.get("plan"))
        k_value = payload.get("requested_k")
        key = (plan, int(k_value) if k_value is not None else None)
        plan_ks.add(key)
        identities[(key, bool(payload.get("measured")))] += 1
        if int(payload.get("score_pass_sample_size", -1)) != min(size, 50_000):
            raise RuntimeError("observation P sample is not fixed and cohort-bound")
        if payload.get("peak_rss_bytes") is None:
            raise RuntimeError("observation RSS is missing")
        if payload.get("temp_spill") is None or payload.get("oom") is None:
            raise RuntimeError("observation spill/OOM evidence is missing")
    expected_plans = set(ALL_PLANS)
    if {plan for plan, _ in plan_ks} != expected_plans:
        raise RuntimeError("measurement plan coverage is incomplete")
    for key in plan_ks:
        if identities[(key, False)] != warmups:
            raise RuntimeError(f"warm-up count is incomplete for {key!r}")
        if identities[(key, True)] != repetitions:
            raise RuntimeError(f"measured count is incomplete for {key!r}")
    ann_ks = {k for plan, k in plan_ks if plan == "ann_first"}
    if confirmation:
        if ann_ks != {requested_k}:
            raise RuntimeError("confirmation did not use exactly its registered K")
    elif not ann_ks or None in ann_ks:
        raise RuntimeError("tuning K screening is empty")


def _scenario_records(manifest: Mapping[str, Any]) -> Iterable[Mapping[str, Any]]:
    raw = manifest.get("scenarios")
    if not isinstance(raw, list) or not raw:
        raise ValueError("scenario manifest is empty")
    scenarios = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise ValueError("scenario manifest row is invalid")
        ScenarioSet(str(item.get("scenario_set")))
        scenarios.append(item)
    return sorted(
        scenarios,
        key=lambda item: (
            0 if item["scenario_set"] == ScenarioSet.TUNING.value else 1,
            str(item["scenario_id"]),
        ),
    )


def _require_v2_manifest(payload: Mapping[str, Any]) -> None:
    if payload.get("experiment_version") != SCALE_EXPERIMENT_VERSION:
        raise ValueError("Goal 1 runner only accepts benchmark v2 manifests")
    if payload.get("campaign_id") is not None or payload.get("promotion_id") is not None:
        raise ValueError("Goal 1 must not run exclusion confirmation")


def _require_v2_output_root(path: Path) -> None:
    if path.resolve() != (ROOT / SCALE_OUTPUT_ROOT).resolve():
        raise ValueError("Goal 1 output must remain isolated under scale-series-v2")


def _object_at(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _progress(event: str, size: int, path: Path, **extra: Any) -> None:
    print(
        json.dumps(
            {"event": event, "cohort_size": size, "path": str(path), **extra},
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    raise SystemExit(main())
