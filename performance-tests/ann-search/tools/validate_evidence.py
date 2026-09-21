#!/usr/bin/env python3
"""Validate the committed ANN experiment checkpoint and optional local sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_ROOT = ROOT / "performance-tests" / "ann-search" / "evidence"
INDEX_PATH = EVIDENCE_ROOT / "experiment-index.json"


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(*, verify_local_artifacts: bool) -> list[str]:
    index = _load_json(INDEX_PATH)
    experiments = index.get("experiments")
    _require(isinstance(experiments, list), "index.experiments must be an array")
    _require(index.get("experimentCount") == len(experiments), "experiment count mismatch")

    messages: list[str] = []
    for entry in experiments:
        _require(isinstance(entry, dict), "experiment index entry must be an object")
        summary_path = EVIDENCE_ROOT / str(entry["summary"])
        report_path = EVIDENCE_ROOT / str(entry["report"])
        _require(summary_path.is_file(), f"missing summary: {summary_path}")
        _require(report_path.is_file(), f"missing report: {report_path}")

        summary = _load_json(summary_path)
        _require(summary["id"] == entry["id"], "summary id does not match index")
        _require(summary["status"] == entry["status"], "summary status does not match index")
        _require(
            summary["claimScope"] == "query_specific_candidate_retrieval_only",
            "claim scope was widened",
        )

        decision = summary["decision"]
        _require(isinstance(decision, dict), "decision must be an object")
        _require(decision["candidateRetrievalPassed"] is True, "candidate result changed")
        _require(decision["fullMembershipPassed"] is False, "full membership must remain false")
        _require(decision["deployableCommonSetting"] is False, "common setting must remain false")
        _require(decision["annPolicyCandidate"] is False, "ANN policy must remain false")
        _require(decision["runtimePolicy"] == "keep_exact_fallback", "fallback boundary changed")

        representative = summary["representativeResult"]
        _require(isinstance(representative, dict), "representative result must be an object")
        warm = representative["warm"]
        _require(isinstance(warm, dict), "warm result must be an object")
        expected_speedup = warm["exactP95Ms"] / warm["annP95Ms"]
        _require(
            math.isclose(expected_speedup, warm["speedup"], rel_tol=0.002),
            "representative speedup is inconsistent",
        )
        gates = summary["gates"]
        _require(isinstance(gates, dict), "gates must be an object")
        _require(warm["recallAtK"] >= gates["minimumRecallAtK"], "recall gate failed")
        _require(
            warm["wilsonLowerBound"] >= gates["minimumWilsonLowerBound"],
            "Wilson gate failed",
        )
        _require(
            warm["bootstrapP95RatioUpperBound"] <= gates["maximumAnnToExactP95Ratio"],
            "latency ratio gate failed",
        )

        for code in summary["measurementCode"]:
            code_path = ROOT / code["path"]
            _require(code_path.is_file(), f"missing measurement code: {code_path}")
            _require(
                _sha256(code_path) == code["checkpointSha256"],
                f"checkpoint code hash mismatch: {code['path']}",
            )

        if verify_local_artifacts:
            for source in summary["sourceEvidence"]:
                source_path = ROOT / source["path"]
                _require(source_path.is_file(), f"missing local artifact: {source_path}")
                _require(source_path.stat().st_size == source["bytes"], f"size mismatch: {source_path}")
                _require(_sha256(source_path) == source["sha256"], f"hash mismatch: {source_path}")

        messages.append(f"validated {entry['id']} ({entry['status']})")

    return messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--verify-local-artifacts",
        action="store_true",
        help="also verify ignored local raw sources by size and SHA-256",
    )
    args = parser.parse_args()
    for message in validate(verify_local_artifacts=args.verify_local_artifacts):
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
