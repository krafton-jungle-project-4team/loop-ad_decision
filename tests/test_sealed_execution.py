from __future__ import annotations

import json
from pathlib import Path

import pytest

from offline_evaluation.sealed_execution import (
    STATUS_COMPLETED,
    STATUS_FAILED_AFTER_OUTCOMES,
    STATUS_RESERVED,
    STATUS_RESULT_STAGED,
    STATUS_RETRYABLE_PRE_OUTCOME_FAILURE,
    SealedExecutionError,
    mark_execution_failure,
    mark_outcomes_opened,
    mark_result_staged,
    prepare_staging_output,
    publish_staged_result,
    reserve_sealed_execution,
)


def test_pre_outcome_failure_requires_same_execution_id_to_resume(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "sealed.json"
    output_dir = tmp_path / "result"
    execution = _reserve(manifest_path, output_dir)

    failed = mark_execution_failure(execution, RuntimeError("source unavailable"))

    assert failed.status == STATUS_RETRYABLE_PRE_OUTCOME_FAILURE
    with pytest.raises(SealedExecutionError, match="exact value"):
        _reserve(manifest_path, output_dir)
    with pytest.raises(SealedExecutionError, match="does not match"):
        _reserve(
            manifest_path,
            output_dir,
            resume_execution_id="different-execution",
        )

    resumed = _reserve(
        manifest_path,
        output_dir,
        resume_execution_id=execution.execution_id,
    )

    assert resumed.execution_id == execution.execution_id
    assert resumed.status == STATUS_RESERVED
    assert resumed.attempt_count == 2


def test_failure_after_outcomes_blocks_re_evaluation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sealed.json"
    output_dir = tmp_path / "result"
    execution = _reserve(manifest_path, output_dir)
    mark_outcomes_opened(execution)

    failed = mark_execution_failure(execution, RuntimeError("evaluation failed"))

    assert failed.status == STATUS_FAILED_AFTER_OUTCOMES
    with pytest.raises(SealedExecutionError, match="repeat the final test"):
        _reserve(
            manifest_path,
            output_dir,
            resume_execution_id=execution.execution_id,
        )


def test_staged_result_can_resume_publication_without_re_evaluation(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "sealed.json"
    output_dir = tmp_path / "result"
    execution = _reserve(manifest_path, output_dir)
    staging_dir = prepare_staging_output(execution)
    staging_dir.mkdir()
    (staging_dir / "summary.json").write_text("{}\n", encoding="utf-8")
    mark_outcomes_opened(execution)
    staged = mark_result_staged(execution)
    staging_dir.replace(output_dir)

    assert staged.status == STATUS_RESULT_STAGED
    resumed = _reserve(
        manifest_path,
        output_dir,
        resume_execution_id=execution.execution_id,
    )
    completed = publish_staged_result(resumed)

    assert completed.status == STATUS_COMPLETED
    assert (output_dir / "summary.json").is_file()
    assert not staging_dir.exists()
    with pytest.raises(SealedExecutionError, match="already completed"):
        _reserve(
            manifest_path,
            output_dir,
            resume_execution_id=execution.execution_id,
        )


def test_legacy_execution_marker_remains_consumed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sealed.json"
    journal_path = tmp_path / "sealed.execution-started.json"
    journal_path.write_text(
        json.dumps({"status": "started_outcomes_unsealed"}),
        encoding="utf-8",
    )

    with pytest.raises(SealedExecutionError, match="remains consumed"):
        _reserve(manifest_path, tmp_path / "result")


def test_tampered_staging_path_is_rejected_before_cleanup(tmp_path: Path) -> None:
    manifest_path = tmp_path / "sealed.json"
    execution = _reserve(manifest_path, tmp_path / "result")
    payload = json.loads(execution.journal_path.read_text(encoding="utf-8"))
    payload["staging_dir"] = str(tmp_path)
    execution.journal_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(SealedExecutionError, match="invariants are invalid"):
        prepare_staging_output(execution)


def _reserve(
    manifest_path: Path,
    output_dir: Path,
    *,
    resume_execution_id: str | None = None,
):
    return reserve_sealed_execution(
        manifest_path,
        manifest_id="manifest-1",
        manifest_integrity_sha256="integrity-1",
        code_commit="commit-1",
        output_dir=output_dir,
        resume_execution_id=resume_execution_id,
    )
