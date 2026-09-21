#!/usr/bin/env python3
"""Freeze only validated tuning/confirmation pairs for scale measurement."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from offline_evaluation.ann_search_benchmark import BenchmarkManifest  # noqa: E402
from offline_evaluation.ann_search_scale_artifacts import (  # noqa: E402
    write_immutable_json,
)
from offline_evaluation.ann_search_scale_series import (  # noqa: E402
    SCALE_EXPERIMENT_VERSION,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = object_at(args.manifest)
    definitions = object_at(args.definitions)
    pairs = object_at(args.pairs)
    selected_ids, fallbacks = selected_scenario_ids(pairs)
    selected_scenarios = filter_scenarios(manifest["scenarios"], selected_ids)
    selected_definitions = filter_scenarios(definitions["scenarios"], selected_ids)
    validate_selected_pairs(selected_scenarios, pairs)
    selected_manifest = {
        "experiment_version": SCALE_EXPERIMENT_VERSION,
        "project_id": manifest["project_id"],
        "vector_version": manifest["vector_version"],
        "manifest_hash": manifest["manifest_hash"],
        "scenarios": selected_scenarios,
    }
    manifest_path = args.output_dir / "selected-scenario-manifest.json"
    write_immutable_json(manifest_path, selected_manifest)
    BenchmarkManifest.load(manifest_path)
    write_immutable_json(
        args.output_dir / "selected-scenario-definitions.json",
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "scenarios": selected_definitions,
        },
    )
    write_immutable_json(
        args.output_dir / "unvalidated-scenario-fallbacks.json",
        {
            "experiment_version": SCALE_EXPERIMENT_VERSION,
            "fallback_plan": "exact_all",
            "fallbacks": fallbacks,
        },
    )
    print(
        json.dumps(
            {
                "selected_scenario_count": len(selected_scenarios),
                "validated_pair_count": len(selected_scenarios) // 2,
                "fallback_count": len(fallbacks),
            },
            indent=2,
        )
    )
    return 0


def selected_scenario_ids(
    pairs: Mapping[str, Any],
) -> tuple[set[str], list[Mapping[str, Any]]]:
    selected: set[str] = set()
    fallbacks: list[Mapping[str, Any]] = []
    for pair in pairs["pairs"]:
        if bool(pair["validated"]):
            tuning = str(pair["tuning_scenario_id"])
            confirmation = str(pair["confirmation_scenario_id"])
            if not tuning or not confirmation or tuning == confirmation:
                raise ValueError("validated pair has invalid scenario IDs")
            selected.update((tuning, confirmation))
        else:
            fallbacks.append(dict(pair))
    return selected, fallbacks


def filter_scenarios(raw: Any, selected_ids: set[str]) -> list[Mapping[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("scenario list is invalid")
    by_id = {str(row["scenario_id"]): row for row in raw}
    missing = selected_ids - set(by_id)
    if missing:
        raise ValueError(f"selected scenarios are absent: {sorted(missing)}")
    return [dict(by_id[scenario_id]) for scenario_id in sorted(selected_ids)]


def validate_selected_pairs(
    scenarios: list[Mapping[str, Any]],
    pairs: Mapping[str, Any],
) -> None:
    by_id = {str(row["scenario_id"]): row for row in scenarios}
    for pair in pairs["pairs"]:
        if not bool(pair["validated"]):
            continue
        tuning = by_id[str(pair["tuning_scenario_id"])]
        confirmation = by_id[str(pair["confirmation_scenario_id"])]
        if tuning["scenario_set"] != "tuning":
            raise ValueError("pair tuning scenario has the wrong set")
        if confirmation["scenario_set"] != "confirmation":
            raise ValueError("pair confirmation scenario has the wrong set")
        if tuning["definition_sha256"] == confirmation["definition_sha256"]:
            raise ValueError("pair definitions are not distinct")


def object_at(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain an object")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
