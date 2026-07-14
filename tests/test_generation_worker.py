from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

import pytest

from app.config import Settings
from app.generation.errors import (
    PermanentGenerationError,
    RetryableGenerationError,
)
from app.generation.repositories import ContentCandidateRecord
from app.generation.schemas import ContentChannel
from app.generation.service import DurableGenerationResult
from app.generation import worker as worker_module
from app.generation.worker import GenerationJobProcessor


NOW = datetime(2026, 7, 14, 12, 0, tzinfo=timezone.utc)
LEASE_TOKEN = UUID("00000000-0000-0000-0000-000000000247")


def settings() -> Settings:
    return Settings(
        env="test",
        service_id="decision-api",
        port=8080,
        internal_api_key="test-key",
        aurora_host="localhost",
        aurora_port=5432,
        aurora_database="loopad",
        aurora_username="loopad",
        aurora_password="secret",
        clickhouse_url="http://localhost:8123",
        clickhouse_database="loopad",
        clickhouse_username="loopad",
        clickhouse_password="secret",
        data_storage_bucket="loopad-test",
        genai_assets_base_prefix="genai",
        openai_api_key="openai-test",
        gemini_api_key="gemini-test",
        generation_max_retries=3,
        generation_retry_backoff_seconds=(60, 300, 900),
    )


def candidate(index: int) -> ContentCandidateRecord:
    return ContentCandidateRecord(
        content_id=f"content_sms_{index}",
        content_option_id=f"sms_option_{index}",
        generation_id="generation_247",
        analysis_id="analysis_247",
        project_id="hotel-client-a",
        campaign_id="campaign_247",
        promotion_id="promo_247",
        segment_id=f"segment_{index}",
        channel=ContentChannel.SMS,
        message=f"호텔 프로모션 {index}",
        landing_url="https://demo.example.test/hotel",
        creative_format="sms_text",
        image_generation_status="not_required",
        artifact_status="not_required",
    )


def durable_result() -> DurableGenerationResult:
    return DurableGenerationResult(
        generation_id="generation_247",
        content_candidates=(candidate(1), candidate(2)),
        output_json={
            "content_candidate_ids": ["content_sms_1", "content_sms_2"]
        },
        generation_report_json={"content_candidate_count": 2},
    )


def claimed_row(*, retry_count: int = 0) -> dict[str, Any]:
    return {
        "generation_id": "generation_247",
        "retry_count": retry_count,
        "input_json": {"schema_version": "generation.request.v1"},
    }


class FakeExecutor:
    def __init__(
        self,
        *,
        result: DurableGenerationResult | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result
        self.error = error
        self.calls: list[dict[str, Any]] = []

    def execute_durable(
        self,
        *,
        generation_id: str,
        prompt_inputs: list[Any],
    ) -> DurableGenerationResult:
        self.calls.append(
            {
                "generation_id": generation_id,
                "prompt_inputs": prompt_inputs,
            }
        )
        if self.error is not None:
            raise self.error
        assert self.result is not None
        return self.result


@dataclass
class RepositoryState:
    candidate_calls: list[dict[str, Any]] = field(default_factory=list)
    completion_calls: list[dict[str, Any]] = field(default_factory=list)
    retry_calls: list[dict[str, Any]] = field(default_factory=list)
    failed_calls: list[dict[str, Any]] = field(default_factory=list)
    reject_candidate: bool = False
    reject_completion: bool = False
    transition_succeeds: bool = True
    transition_error: Exception | None = None


class FakeConnection:
    def __init__(self, state: RepositoryState) -> None:
        self.state = state
        self.commit_count = 0
        self.rollback_count = 0
        self.close_count = 0

    def commit(self) -> None:
        self.commit_count += 1

    def rollback(self) -> None:
        self.rollback_count += 1

    def close(self) -> None:
        self.close_count += 1


class FakeConnectionFactory:
    def __init__(self, state: RepositoryState) -> None:
        self.state = state
        self.connections: list[FakeConnection] = []

    def __call__(self, _settings: Settings) -> FakeConnection:
        connection = FakeConnection(self.state)
        self.connections.append(connection)
        return connection


class FakeContentCandidateRepository:
    def __init__(self, connection: FakeConnection) -> None:
        self.state = connection.state

    def upsert_fenced(
        self,
        record: ContentCandidateRecord,
        *,
        worker_id: str,
        lease_token: UUID,
    ) -> dict[str, Any] | None:
        self.state.candidate_calls.append(
            {
                "record": record,
                "worker_id": worker_id,
                "lease_token": lease_token,
            }
        )
        if self.state.reject_candidate:
            return None
        return {"content_id": record.content_id}


class FakeGenerationRunRepository:
    def __init__(self, connection: FakeConnection) -> None:
        self.state = connection.state

    def complete_if_ready_fenced(self, **values: Any) -> dict[str, Any] | None:
        self.state.completion_calls.append(values)
        if self.state.reject_completion:
            return None
        return {"generation_id": values["generation_id"], "status": "completed"}

    def schedule_retry_fenced(self, **values: Any) -> bool:
        self.state.retry_calls.append(values)
        if self.state.transition_error is not None:
            raise self.state.transition_error
        return self.state.transition_succeeds

    def mark_failed_fenced(self, **values: Any) -> bool:
        self.state.failed_calls.append(values)
        if self.state.transition_error is not None:
            raise self.state.transition_error
        return self.state.transition_succeeds


def processor(
    *,
    monkeypatch: pytest.MonkeyPatch,
    state: RepositoryState,
    executor: FakeExecutor,
) -> tuple[GenerationJobProcessor, FakeConnectionFactory]:
    monkeypatch.setattr(
        worker_module,
        "ContentCandidateRepository",
        FakeContentCandidateRepository,
    )
    monkeypatch.setattr(
        worker_module,
        "GenerationRunRepository",
        FakeGenerationRunRepository,
    )
    monkeypatch.setattr(
        worker_module,
        "prompt_inputs_from_snapshot",
        lambda _snapshot: ["durable-prompt-input"],
    )
    connection_factory = FakeConnectionFactory(state)
    job_processor = GenerationJobProcessor(
        settings=settings(),
        connection_factory=connection_factory,
        generation_service_factory=lambda: executor,
        clock=lambda: NOW,
        jitter=lambda _base_seconds: 0.0,
    )
    return job_processor, connection_factory


def test_success_upserts_all_candidates_then_completes_and_commits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RepositoryState()
    executor = FakeExecutor(result=durable_result())
    job_processor, connection_factory = processor(
        monkeypatch=monkeypatch,
        state=state,
        executor=executor,
    )

    job_processor.process(
        claimed_row(),
        worker_id="worker-247",
        lease_token=LEASE_TOKEN,
    )

    assert executor.calls == [
        {
            "generation_id": "generation_247",
            "prompt_inputs": ["durable-prompt-input"],
        }
    ]
    assert [
        call["record"].content_id for call in state.candidate_calls
    ] == ["content_sms_1", "content_sms_2"]
    assert all(call["worker_id"] == "worker-247" for call in state.candidate_calls)
    assert all(call["lease_token"] == LEASE_TOKEN for call in state.candidate_calls)
    assert state.completion_calls == [
        {
            "generation_id": "generation_247",
            "worker_id": "worker-247",
            "lease_token": LEASE_TOKEN,
            "output_json": {
                "content_candidate_ids": ["content_sms_1", "content_sms_2"]
            },
            "generation_report_json": {"content_candidate_count": 2},
        }
    ]
    assert state.retry_calls == []
    assert state.failed_calls == []
    assert len(connection_factory.connections) == 1
    connection = connection_factory.connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1


def test_retryable_failure_schedules_next_retry_count_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RepositoryState()
    executor = FakeExecutor(
        error=RetryableGenerationError(
            code="provider_rate_limited",
            safe_message="Provider rate limit was reached.",
            status_code=429,
        )
    )
    job_processor, connection_factory = processor(
        monkeypatch=monkeypatch,
        state=state,
        executor=executor,
    )

    job_processor.process(
        claimed_row(retry_count=1),
        worker_id="worker-247",
        lease_token=LEASE_TOKEN,
    )

    assert state.retry_calls == [
        {
            "generation_id": "generation_247",
            "worker_id": "worker-247",
            "lease_token": LEASE_TOKEN,
            "next_retry_at": NOW + timedelta(seconds=300),
            "error_code": "provider_rate_limited",
            "error_message": "Provider rate limit was reached.",
        }
    ]
    assert state.failed_calls == []
    connection = connection_factory.connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1


@pytest.mark.parametrize(
    ("error", "retry_count", "expected_code"),
    [
        (
            PermanentGenerationError(
                code="generation_invalid_input",
                safe_message="Generation input was invalid.",
            ),
            0,
            "generation_invalid_input",
        ),
        (
            RetryableGenerationError(
                code="provider_timeout",
                safe_message="Provider request timed out.",
            ),
            3,
            "provider_timeout",
        ),
    ],
)
def test_permanent_or_exhausted_failure_marks_run_failed(
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
    retry_count: int,
    expected_code: str,
) -> None:
    state = RepositoryState()
    job_processor, connection_factory = processor(
        monkeypatch=monkeypatch,
        state=state,
        executor=FakeExecutor(error=error),
    )

    job_processor.process(
        claimed_row(retry_count=retry_count),
        worker_id="worker-247",
        lease_token=LEASE_TOKEN,
    )

    assert state.retry_calls == []
    assert len(state.failed_calls) == 1
    assert state.failed_calls[0]["error_code"] == expected_code
    assert state.failed_calls[0]["generation_id"] == "generation_247"
    connection = connection_factory.connections[0]
    assert connection.commit_count == 1
    assert connection.rollback_count == 0
    assert connection.close_count == 1


@pytest.mark.parametrize("rejected_write", ["candidate", "completion"])
def test_fenced_write_rejection_never_commits_completion(
    monkeypatch: pytest.MonkeyPatch,
    rejected_write: str,
) -> None:
    state = RepositoryState(
        reject_candidate=rejected_write == "candidate",
        reject_completion=rejected_write == "completion",
        transition_succeeds=False,
    )
    job_processor, connection_factory = processor(
        monkeypatch=monkeypatch,
        state=state,
        executor=FakeExecutor(result=durable_result()),
    )

    job_processor.process(
        claimed_row(),
        worker_id="stale-worker",
        lease_token=LEASE_TOKEN,
    )

    assert len(connection_factory.connections) == 2
    result_connection, failure_connection = connection_factory.connections
    assert result_connection.commit_count == 0
    assert result_connection.rollback_count == 1
    assert result_connection.close_count == 1
    assert failure_connection.commit_count == 1
    assert failure_connection.close_count == 1
    assert state.retry_calls == []
    assert len(state.failed_calls) == 1
    assert state.failed_calls[0]["error_code"] == "generation_fence_rejected"
    if rejected_write == "candidate":
        assert state.completion_calls == []
    else:
        assert len(state.completion_calls) == 1


def test_failure_transition_error_rolls_back_and_closes_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = RepositoryState(transition_error=RuntimeError("database unavailable"))
    job_processor, connection_factory = processor(
        monkeypatch=monkeypatch,
        state=state,
        executor=FakeExecutor(
            error=RetryableGenerationError(
                code="provider_timeout",
                safe_message="Provider request timed out.",
            )
        ),
    )

    with pytest.raises(RuntimeError, match="database unavailable"):
        job_processor.process(
            claimed_row(),
            worker_id="worker-247",
            lease_token=LEASE_TOKEN,
        )

    connection = connection_factory.connections[0]
    assert connection.commit_count == 0
    assert connection.rollback_count == 1
    assert connection.close_count == 1
