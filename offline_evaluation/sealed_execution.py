from __future__ import annotations

import fcntl
import json
import os
import shutil
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator, Mapping
from uuid import uuid4


SEALED_EXECUTION_VERSION = "sealed-execution.v1"

STATUS_RESERVED = "reserved"
STATUS_RETRYABLE_PRE_OUTCOME_FAILURE = "retryable_pre_outcome_failure"
STATUS_OUTCOMES_OPENED = "outcomes_opened"
STATUS_FAILED_AFTER_OUTCOMES = "failed_after_outcomes"
STATUS_RESULT_STAGED = "result_staged"
STATUS_COMPLETED = "completed"

_RETRYABLE_STATUSES = frozenset(
    {
        STATUS_RESERVED,
        STATUS_RETRYABLE_PRE_OUTCOME_FAILURE,
        STATUS_RESULT_STAGED,
    }
)
_EXECUTION_STATUSES = frozenset(
    {
        STATUS_RESERVED,
        STATUS_RETRYABLE_PRE_OUTCOME_FAILURE,
        STATUS_OUTCOMES_OPENED,
        STATUS_FAILED_AFTER_OUTCOMES,
        STATUS_RESULT_STAGED,
        STATUS_COMPLETED,
    }
)


class SealedExecutionError(ValueError):
    """Raised when a sealed evaluation violates its execution contract."""


@dataclass(frozen=True, slots=True)
class SealedExecution:
    journal_path: Path
    execution_id: str
    manifest_id: str
    manifest_integrity_sha256: str
    code_commit: str
    output_dir: Path
    staging_dir: Path
    status: str
    attempt_count: int
    created_at: str
    updated_at: str
    failure: Mapping[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "version": SEALED_EXECUTION_VERSION,
            "execution_id": self.execution_id,
            "manifest_id": self.manifest_id,
            "manifest_integrity_sha256": self.manifest_integrity_sha256,
            "code_commit": self.code_commit,
            "output_dir": str(self.output_dir),
            "staging_dir": str(self.staging_dir),
            "status": self.status,
            "attempt_count": self.attempt_count,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "failure": dict(self.failure) if self.failure is not None else None,
        }


def reserve_sealed_execution(
    manifest_path: Path,
    *,
    manifest_id: str,
    manifest_integrity_sha256: str,
    code_commit: str,
    output_dir: Path,
    resume_execution_id: str | None = None,
) -> SealedExecution:
    journal_path = execution_journal_path(manifest_path)
    normalized_output_dir = output_dir.expanduser().resolve()
    if not journal_path.exists():
        if resume_execution_id is not None:
            raise SealedExecutionError(
                "sealed execution does not exist; remove --resume-execution-id "
                "to start it"
            )
        if normalized_output_dir.exists():
            raise SealedExecutionError(
                "sealed final output already exists; choose a new empty path"
            )
        execution_id = uuid4().hex
        timestamp = _now()
        execution = SealedExecution(
            journal_path=journal_path,
            execution_id=execution_id,
            manifest_id=manifest_id,
            manifest_integrity_sha256=manifest_integrity_sha256,
            code_commit=code_commit,
            output_dir=normalized_output_dir,
            staging_dir=_staging_dir(normalized_output_dir, execution_id),
            status=STATUS_RESERVED,
            attempt_count=1,
            created_at=timestamp,
            updated_at=timestamp,
        )
        _write_new_execution(execution)
        return execution

    execution = load_sealed_execution(journal_path)
    _validate_execution_binding(
        execution,
        manifest_id=manifest_id,
        manifest_integrity_sha256=manifest_integrity_sha256,
        code_commit=code_commit,
        output_dir=normalized_output_dir,
    )
    if resume_execution_id is None:
        raise SealedExecutionError(
            "sealed final test already has execution "
            f"{execution.execution_id!r}; retryable recovery requires "
            "--resume-execution-id with that exact value"
        )
    if resume_execution_id != execution.execution_id:
        raise SealedExecutionError(
            "resume execution ID does not match the sealed execution journal"
        )
    if execution.status not in _RETRYABLE_STATUSES:
        raise SealedExecutionError(_non_retryable_message(execution))

    resumed_status = (
        STATUS_RESERVED
        if execution.status == STATUS_RETRYABLE_PRE_OUTCOME_FAILURE
        else execution.status
    )
    resumed = replace(
        execution,
        status=resumed_status,
        attempt_count=execution.attempt_count + 1,
        updated_at=_now(),
        failure=None if resumed_status == STATUS_RESERVED else execution.failure,
    )
    _write_execution(resumed)
    return resumed


def load_sealed_execution(path: Path) -> SealedExecution:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SealedExecutionError(
            "sealed execution journal is not valid JSON"
        ) from exc
    if not isinstance(payload, Mapping):
        raise SealedExecutionError("sealed execution journal must be a JSON object")
    if payload.get("version") != SEALED_EXECUTION_VERSION:
        if payload.get("status") == "started_outcomes_unsealed":
            raise SealedExecutionError(
                "legacy sealed execution marker cannot prove that outcomes were "
                "not opened; this manifest remains consumed"
            )
        raise SealedExecutionError("unsupported sealed execution journal version")
    try:
        execution = SealedExecution(
            journal_path=path,
            execution_id=str(payload["execution_id"]),
            manifest_id=str(payload["manifest_id"]),
            manifest_integrity_sha256=str(
                payload["manifest_integrity_sha256"]
            ),
            code_commit=str(payload["code_commit"]),
            output_dir=Path(str(payload["output_dir"])),
            staging_dir=Path(str(payload["staging_dir"])),
            status=str(payload["status"]),
            attempt_count=int(payload["attempt_count"]),
            created_at=str(payload["created_at"]),
            updated_at=str(payload["updated_at"]),
            failure=(
                dict(payload["failure"])
                if isinstance(payload.get("failure"), Mapping)
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SealedExecutionError(
            "sealed execution journal schema is invalid"
        ) from exc
    if (
        not execution.execution_id
        or not execution.manifest_id
        or execution.attempt_count <= 0
        or execution.status not in _EXECUTION_STATUSES
        or not execution.output_dir.is_absolute()
        or execution.staging_dir
        != _staging_dir(execution.output_dir, execution.execution_id)
    ):
        raise SealedExecutionError(
            "sealed execution journal invariants are invalid"
        )
    return execution


@contextmanager
def sealed_execution_attempt(
    execution: SealedExecution,
) -> Iterator[None]:
    lock_path = execution.journal_path.with_name(
        f"{execution.journal_path.name}.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SealedExecutionError(
                "another process is already running this sealed execution"
            ) from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def prepare_staging_output(execution: SealedExecution) -> Path:
    current = load_sealed_execution(execution.journal_path)
    if current.status != STATUS_RESERVED:
        raise SealedExecutionError(
            "staging output can only be prepared before outcomes are opened"
        )
    if current.output_dir.exists():
        raise SealedExecutionError(
            "sealed final output appeared before publication"
        )
    if current.staging_dir.exists():
        shutil.rmtree(current.staging_dir)
    current.staging_dir.parent.mkdir(parents=True, exist_ok=True)
    return current.staging_dir


def mark_outcomes_opened(execution: SealedExecution) -> SealedExecution:
    current = load_sealed_execution(execution.journal_path)
    if current.status == STATUS_OUTCOMES_OPENED:
        return current
    if current.status != STATUS_RESERVED:
        raise SealedExecutionError(
            f"cannot open outcomes while execution is {current.status!r}"
        )
    return _transition(current, status=STATUS_OUTCOMES_OPENED)


def mark_execution_failure(
    execution: SealedExecution,
    error: BaseException,
) -> SealedExecution:
    current = load_sealed_execution(execution.journal_path)
    failure = {
        "type": type(error).__name__,
        "message": str(error),
        "recorded_at": _now(),
    }
    if current.status == STATUS_RESERVED:
        return _transition(
            current,
            status=STATUS_RETRYABLE_PRE_OUTCOME_FAILURE,
            failure=failure,
        )
    if current.status == STATUS_OUTCOMES_OPENED:
        return _transition(
            current,
            status=STATUS_FAILED_AFTER_OUTCOMES,
            failure=failure,
        )
    if current.status == STATUS_RESULT_STAGED:
        return _transition(current, failure=failure)
    return current


def mark_result_staged(execution: SealedExecution) -> SealedExecution:
    current = load_sealed_execution(execution.journal_path)
    if current.status != STATUS_OUTCOMES_OPENED:
        raise SealedExecutionError(
            "sealed result can only be staged after outcomes are opened"
        )
    if not current.staging_dir.is_dir():
        raise SealedExecutionError("sealed result staging directory is missing")
    return _transition(
        current,
        status=STATUS_RESULT_STAGED,
        failure=None,
    )


def publish_staged_result(execution: SealedExecution) -> SealedExecution:
    current = load_sealed_execution(execution.journal_path)
    if current.status != STATUS_RESULT_STAGED:
        raise SealedExecutionError(
            "only a fully staged sealed result can be published"
        )
    staging_exists = current.staging_dir.exists()
    output_exists = current.output_dir.exists()
    if staging_exists and output_exists:
        raise SealedExecutionError(
            "both sealed staging and final output directories exist"
        )
    if staging_exists:
        current.staging_dir.replace(current.output_dir)
    elif not output_exists:
        raise SealedExecutionError(
            "sealed staged result disappeared before publication"
        )
    return _transition(
        current,
        status=STATUS_COMPLETED,
        failure=None,
    )


def execution_journal_path(manifest_path: Path) -> Path:
    return manifest_path.with_name(
        f"{manifest_path.stem}.execution-started.json"
    )


def _validate_execution_binding(
    execution: SealedExecution,
    *,
    manifest_id: str,
    manifest_integrity_sha256: str,
    code_commit: str,
    output_dir: Path,
) -> None:
    expected = (
        execution.manifest_id == manifest_id
        and execution.manifest_integrity_sha256 == manifest_integrity_sha256
        and execution.code_commit == code_commit
        and execution.output_dir == output_dir
    )
    if not expected:
        raise SealedExecutionError(
            "sealed execution does not match the manifest, code, or output path"
        )


def _non_retryable_message(execution: SealedExecution) -> str:
    if execution.status == STATUS_OUTCOMES_OPENED:
        return (
            "sealed outcomes were opened and the prior attempt did not finish; "
            "re-evaluation is blocked"
        )
    if execution.status == STATUS_FAILED_AFTER_OUTCOMES:
        return (
            "sealed evaluation failed after outcomes were opened; rerunning it "
            "would repeat the final test"
        )
    if execution.status == STATUS_COMPLETED:
        return "sealed final test is already completed"
    return f"sealed execution status {execution.status!r} is not retryable"


def _write_new_execution(execution: SealedExecution) -> None:
    execution.journal_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with execution.journal_path.open("x", encoding="utf-8") as destination:
            _dump_execution(destination, execution)
    except FileExistsError as exc:
        raise SealedExecutionError(
            "sealed execution was created concurrently; inspect its journal"
        ) from exc


def _write_execution(execution: SealedExecution) -> None:
    temporary_path = execution.journal_path.with_name(
        f".{execution.journal_path.name}.{uuid4().hex}.tmp"
    )
    try:
        with temporary_path.open("x", encoding="utf-8") as destination:
            _dump_execution(destination, execution)
        temporary_path.replace(execution.journal_path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _dump_execution(destination: Any, execution: SealedExecution) -> None:
    json.dump(execution.to_json(), destination, ensure_ascii=False, indent=2)
    destination.write("\n")
    destination.flush()
    os.fsync(destination.fileno())


def _transition(
    execution: SealedExecution,
    *,
    status: str | None = None,
    failure: Mapping[str, Any] | None = None,
) -> SealedExecution:
    updated = replace(
        execution,
        status=status or execution.status,
        updated_at=_now(),
        failure=failure,
    )
    _write_execution(updated)
    return updated


def _staging_dir(output_dir: Path, execution_id: str) -> Path:
    return output_dir.with_name(
        f".{output_dir.name}.{execution_id}.staging"
    )


def _now() -> str:
    return datetime.now(UTC).isoformat()
