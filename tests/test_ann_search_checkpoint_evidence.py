from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR_PATH = (
    ROOT / "performance-tests" / "ann-search" / "tools" / "validate_evidence.py"
)


def _validator_module():
    spec = importlib.util.spec_from_file_location("validate_ann_search_evidence", VALIDATOR_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_ann_search_evidence_is_self_consistent() -> None:
    messages = _validator_module().validate(verify_local_artifacts=False)
    assert messages == ["validated ann-efficiency-recovery-v1 (follow_up_required)"]


def test_local_ann_search_source_artifacts_match_recorded_hashes() -> None:
    validator = _validator_module()
    index = json.loads(validator.INDEX_PATH.read_text(encoding="utf-8"))
    sources = [
        ROOT / source["path"]
        for entry in index["experiments"]
        for source in json.loads(
            (validator.EVIDENCE_ROOT / entry["summary"]).read_text(encoding="utf-8")
        )["sourceEvidence"]
    ]
    if not any(path.exists() for path in sources):
        pytest.skip("optional local ANN source artifacts are not installed")
    # A partial archive or a changed hash still fails; only an absent archive skips.
    messages = validator.validate(verify_local_artifacts=True)
    assert messages == ["validated ann-efficiency-recovery-v1 (follow_up_required)"]
