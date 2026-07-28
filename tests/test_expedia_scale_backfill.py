from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.backfill_expedia_scale_series import (
    completed_partitions,
    plan_resume_actions,
)


def test_partition_resume_skips_only_fully_matching_checkpointed_rows() -> None:
    expected = {
        0: {"raw_event_count": 10, "user_count": 2},
        1: {"raw_event_count": 20, "user_count": 3},
    }
    assert plan_resume_actions(
        expected=expected,
        actual={0: expected[0]},
        completed={0},
    ) == [1]


def test_partition_resume_rejects_uncheckpointed_partial_rows() -> None:
    expected = {0: {"raw_event_count": 10, "user_count": 2}}
    with pytest.raises(RuntimeError, match="uncheckpointed partial"):
        plan_resume_actions(
            expected=expected,
            actual={0: {"raw_event_count": 9, "user_count": 2}},
            completed=set(),
        )


def test_partition_resume_rejects_completed_count_mismatch() -> None:
    expected = {0: {"raw_event_count": 10, "user_count": 2}}
    with pytest.raises(RuntimeError, match="differs from frozen plan"):
        plan_resume_actions(
            expected=expected,
            actual={0: {"raw_event_count": 9, "user_count": 2}},
            completed={0},
        )


def test_completed_partition_checkpoint_rejects_duplicate_rows(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.jsonl"
    row = {"partition_remainder": 0, "status": "insert_command_succeeded"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="duplicates"):
        completed_partitions(path)
