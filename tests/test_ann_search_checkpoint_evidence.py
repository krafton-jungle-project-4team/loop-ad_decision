from __future__ import annotations

import importlib.util
from pathlib import Path


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
    messages = _validator_module().validate(verify_local_artifacts=True)
    assert messages == ["validated ann-efficiency-recovery-v1 (follow_up_required)"]
