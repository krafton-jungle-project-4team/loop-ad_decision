#!/usr/bin/env python3
"""Compile benchmark scenarios through the production audience compiler.

The input contains only registered template parameters and scenario metadata.
Query vectors, score thresholds, hard predicates, and provenance are always
derived from the production binder/compiler/calibration path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.audience_contract import SEGMENT_AUDIENCE_CONTRACT  # noqa: E402
from app.analysis.segment_audience_templates import (  # noqa: E402
    RegisteredSegmentAudienceBinder,
)
from app.analysis.semantic_selection import (  # noqa: E402
    compile_registered_segment_audience,
)
from offline_evaluation.ann_search_benchmark import (  # noqa: E402
    BenchmarkScenario,
)
from offline_evaluation.ann_search_experiment import (  # noqa: E402
    EXPERIMENT_VERSION,
    SUPPORTED_EXPERIMENT_VERSIONS,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-vectors", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source = _load_object(args.source_vectors)
    definitions = _load_object(args.definitions)
    raw_scenarios = definitions.get("scenarios")
    if not isinstance(raw_scenarios, list) or not raw_scenarios:
        raise ValueError("scenario definitions require a non-empty scenarios array")

    binder = RegisteredSegmentAudienceBinder()
    scenarios: list[Mapping[str, Any]] = []
    for raw in raw_scenarios:
        if not isinstance(raw, Mapping):
            raise ValueError("each scenario definition must be an object")
        scenario_id = str(raw["scenario_id"])
        candidate_type = str(raw["candidate_type"])
        binding = binder.bind(
            candidate_type=candidate_type,
            destination_ids=_string_list(raw.get("destination_ids", [])),
            season_months=_integer_list(raw.get("season_months", [])),
            benefit_keys=_string_list(raw.get("benefit_keys", [])),
        )
        compiled = compile_registered_segment_audience(
            segment_id=scenario_id,
            rule_json={
                "audience_resolution_contract": SEGMENT_AUDIENCE_CONTRACT,
                "segment_audience_spec": dict(binding),
            },
        )
        scenarios.append(
            {
                "scenario_id": scenario_id,
                "scenario_set": str(raw["scenario_set"]),
                "candidate_type": candidate_type,
                "query_vector": list(compiled.query_vector),
                "score_threshold": compiled.score_threshold,
                "hard_predicate_keys": list(compiled.hard_predicate_keys),
                "predicate_parameters": {
                    key: list(value)
                    for key, value in compiled.predicate_parameters.items()
                },
                "compiler_provenance": {
                    "manifest_hash": compiled.manifest_hash,
                    "calibration_version": compiled.calibration_version,
                    "calibration_hash": compiled.calibration_hash,
                    "query_compiler_version": compiled.query_compiler_version,
                    "query_compiler_hash": compiled.query_compiler_hash,
                    "template_id": compiled.template_id,
                    "template_semantic_hash": compiled.template_semantic_hash,
                },
            }
        )

    experiment_version = str(
        source.get("experiment_version", EXPERIMENT_VERSION)
    )
    if experiment_version not in SUPPORTED_EXPERIMENT_VERSIONS:
        raise ValueError("source vector experiment version is unsupported")
    payload = {
        "experiment_version": experiment_version,
        "project_id": str(source["project_id"]),
        "vector_version": str(source["vector_version"]),
        "manifest_hash": str(source["manifest_hash"]),
        "scenarios": scenarios,
    }
    validated = tuple(BenchmarkScenario.from_dict(item) for item in scenarios)
    if len({item.scenario_id for item in validated}) != len(validated):
        raise ValueError("benchmark scenario IDs must be unique")
    if any(
        item.compiler_provenance["manifest_hash"] != payload["manifest_hash"]
        for item in validated
    ):
        raise ValueError("scenario compiler manifest hash mismatches source vectors")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "scenario_count": len(scenarios),
                "scenario_ids": [item["scenario_id"] for item in scenarios],
            },
            indent=2,
        )
    )
    return 0


def _load_object(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        raise ValueError("string parameters must be JSON arrays")
    return [str(item) for item in value]


def _integer_list(value: Any) -> list[int]:
    if not isinstance(value, list) or any(
        not isinstance(item, int) or isinstance(item, bool) for item in value
    ):
        raise ValueError("season_months must be an integer JSON array")
    return list(value)


if __name__ == "__main__":
    raise SystemExit(main())
