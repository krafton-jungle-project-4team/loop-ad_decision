from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from types import SimpleNamespace
from typing import Any, cast

import pytest

from app.config import Settings
from app.generation import coordinator as coordinator_module
from app.generation.coordinator import GenerationCoordinator


class RepositoryState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.claims: deque[dict[str, Any]] = deque()
        self.claim_calls = 0
        self.claim_failures = 0
        self.heartbeat_calls: list[dict[str, Any]] = []
        self.heartbeat_false_ids: set[str] = set()
        self.recovery_calls: list[dict[str, Any]] = []
        self.connections: list[FakeConnection] = []

    def enqueue(self, generation_id: str) -> None:
        with self.lock:
            self.claims.append({"generation_id": generation_id})


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


class FakeGenerationRunRepository:
    def __init__(self, connection: FakeConnection) -> None:
        self._state = connection.state

    def claim_next(
        self,
        *,
        worker_id: str,
        lease_token: uuid.UUID,
        lease_seconds: int,
    ) -> dict[str, Any] | None:
        with self._state.lock:
            self._state.claim_calls += 1
            if self._state.claim_failures:
                self._state.claim_failures -= 1
                raise RuntimeError("temporary claim failure")
            if not self._state.claims:
                return None
            claimed = dict(self._state.claims.popleft())
        return {
            **claimed,
            "worker_id": worker_id,
            "lease_token": lease_token,
            "lease_seconds": lease_seconds,
        }

    def heartbeat(
        self,
        *,
        generation_id: str,
        worker_id: str,
        lease_token: uuid.UUID,
        lease_seconds: int,
    ) -> bool:
        with self._state.lock:
            self._state.heartbeat_calls.append(
                {
                    "generation_id": generation_id,
                    "worker_id": worker_id,
                    "lease_token": lease_token,
                    "lease_seconds": lease_seconds,
                }
            )
        return generation_id not in self._state.heartbeat_false_ids

    def recover_expired(
        self,
        *,
        max_retries: int,
        retry_backoff_seconds: tuple[int, ...],
        limit: int,
    ) -> int:
        with self._state.lock:
            self._state.recovery_calls.append(
                {
                    "max_retries": max_retries,
                    "retry_backoff_seconds": retry_backoff_seconds,
                    "limit": limit,
                }
            )
        return 0


class ProcessorTracker:
    def __init__(self, *, release: threading.Event | None = None) -> None:
        self._lock = threading.Lock()
        self._release = release
        self.factory_count = 0
        self.active_count = 0
        self.max_active_count = 0
        self.processed_ids: list[str] = []

    def factory(self) -> TrackingProcessor:
        with self._lock:
            self.factory_count += 1
        return TrackingProcessor(self)

    def process(
        self,
        claimed_row: dict[str, Any],
        *,
        worker_id: str,
        lease_token: uuid.UUID,
    ) -> None:
        assert worker_id == claimed_row["worker_id"]
        assert lease_token == claimed_row["lease_token"]
        with self._lock:
            self.active_count += 1
            self.max_active_count = max(
                self.max_active_count,
                self.active_count,
            )
        try:
            if self._release is not None:
                if not self._release.wait(timeout=2.0):
                    raise TimeoutError("processor release was not signaled")
            with self._lock:
                self.processed_ids.append(str(claimed_row["generation_id"]))
        finally:
            with self._lock:
                self.active_count -= 1


class TrackingProcessor:
    def __init__(self, tracker: ProcessorTracker) -> None:
        self._tracker = tracker

    def process(
        self,
        claimed_row: dict[str, Any],
        *,
        worker_id: str,
        lease_token: uuid.UUID,
    ) -> None:
        self._tracker.process(
            claimed_row,
            worker_id=worker_id,
            lease_token=lease_token,
        )


@pytest.fixture
def repository_state(monkeypatch: pytest.MonkeyPatch) -> RepositoryState:
    state = RepositoryState()
    monkeypatch.setattr(
        coordinator_module,
        "GenerationRunRepository",
        FakeGenerationRunRepository,
    )
    return state


def test_executor_has_fixed_worker_loops_and_heartbeats_active_leases(
    repository_state: RepositoryState,
) -> None:
    for index in range(6):
        repository_state.enqueue(f"generation-{index}")
    release = threading.Event()
    processors = ProcessorTracker(release=release)
    coordinator = _coordinator(
        repository_state,
        processors,
        worker_count=2,
        heartbeat_seconds=0.01,
    )

    coordinator.start()
    assert coordinator.accepting is True
    _wait_until(lambda: processors.max_active_count == 2)
    _wait_until(lambda: len(repository_state.heartbeat_calls) >= 2)

    assert processors.factory_count == 2
    assert processors.max_active_count == 2

    release.set()
    _wait_until(lambda: len(processors.processed_ids) == 6)
    coordinator.shutdown()

    assert coordinator.accepting is False
    assert repository_state.recovery_calls
    assert all(connection.commit_count == 1 for connection in repository_state.connections)
    assert all(connection.rollback_count == 0 for connection in repository_state.connections)
    assert all(connection.close_count == 1 for connection in repository_state.connections)


def test_wake_interrupts_idle_poll_after_job_commit(
    repository_state: RepositoryState,
) -> None:
    processors = ProcessorTracker()
    coordinator = _coordinator(
        repository_state,
        processors,
        idle_poll_seconds=5.0,
    )
    coordinator.start()
    _wait_until(lambda: repository_state.claim_calls >= 1)

    started_at = time.monotonic()
    repository_state.enqueue("generation-woken")
    coordinator.wake()
    _wait_until(lambda: processors.processed_ids == ["generation-woken"])

    assert time.monotonic() - started_at < 0.5
    coordinator.shutdown()


def test_repository_failure_rolls_back_and_worker_keeps_polling(
    repository_state: RepositoryState,
) -> None:
    repository_state.claim_failures = 1
    repository_state.enqueue("generation-after-error")
    processors = ProcessorTracker()
    coordinator = _coordinator(
        repository_state,
        processors,
        poll_seconds=0.01,
        idle_poll_seconds=0.02,
    )

    coordinator.start()
    _wait_until(lambda: processors.processed_ids == ["generation-after-error"])
    coordinator.shutdown()

    assert any(
        connection.rollback_count == 1
        for connection in repository_state.connections
    )
    assert all(connection.close_count == 1 for connection in repository_state.connections)


def test_shutdown_stops_claims_but_keeps_heartbeat_during_grace(
    repository_state: RepositoryState,
) -> None:
    repository_state.enqueue("generation-in-flight")
    release = threading.Event()
    processors = ProcessorTracker(release=release)
    coordinator = _coordinator(
        repository_state,
        processors,
        heartbeat_seconds=0.01,
        shutdown_grace_seconds=0.5,
    )
    coordinator.start()
    _wait_until(lambda: processors.active_count == 1)
    _wait_until(lambda: len(repository_state.heartbeat_calls) >= 1)

    shutdown_thread = threading.Thread(target=coordinator.shutdown)
    shutdown_thread.start()
    _wait_until(lambda: coordinator.accepting is False)
    heartbeat_count_at_shutdown = len(repository_state.heartbeat_calls)
    _wait_until(
        lambda: len(repository_state.heartbeat_calls) > heartbeat_count_at_shutdown
    )

    release.set()
    shutdown_thread.join(timeout=1.0)
    assert shutdown_thread.is_alive() is False

    final_heartbeat_count = len(repository_state.heartbeat_calls)
    time.sleep(0.03)
    assert len(repository_state.heartbeat_calls) == final_heartbeat_count


def test_lost_heartbeat_does_not_block_other_active_leases(
    repository_state: RepositoryState,
) -> None:
    processors = ProcessorTracker()
    coordinator = _coordinator(repository_state, processors)
    lost_token = uuid.uuid4()
    active_token = uuid.uuid4()
    repository_state.heartbeat_false_ids.add("generation-lost")
    lost = coordinator_module.ActiveLease(
        generation_id="generation-lost",
        worker_id="worker-1",
        lease_token=lost_token,
    )
    active = coordinator_module.ActiveLease(
        generation_id="generation-active",
        worker_id="worker-1",
        lease_token=active_token,
    )
    coordinator._register_active_lease(lost)
    coordinator._register_active_lease(active)

    coordinator._heartbeat_active_leases()
    coordinator._heartbeat_active_leases()

    generation_ids = [
        call["generation_id"] for call in repository_state.heartbeat_calls
    ]
    assert generation_ids.count("generation-lost") == 1
    assert generation_ids.count("generation-active") == 2


def test_start_and_shutdown_are_idempotent(
    repository_state: RepositoryState,
) -> None:
    processors = ProcessorTracker()
    coordinator = _coordinator(repository_state, processors)

    coordinator.start()
    coordinator.start()
    assert processors.factory_count == 1

    coordinator.shutdown()
    coordinator.shutdown()
    assert coordinator.accepting is False
    with pytest.raises(RuntimeError, match="cannot restart"):
        coordinator.start()


def _coordinator(
    state: RepositoryState,
    processors: ProcessorTracker,
    *,
    worker_count: int = 1,
    poll_seconds: float = 0.01,
    idle_poll_seconds: float = 0.05,
    heartbeat_seconds: float = 0.02,
    shutdown_grace_seconds: float = 0.2,
) -> GenerationCoordinator:
    settings = cast(
        Settings,
        SimpleNamespace(
            generation_worker_max_concurrency=worker_count,
            generation_poll_interval_seconds=poll_seconds,
            generation_idle_poll_interval_seconds=idle_poll_seconds,
            generation_lease_seconds=10,
            generation_heartbeat_seconds=heartbeat_seconds,
            generation_max_retries=3,
            generation_retry_backoff_seconds=(60, 300, 900),
            generation_shutdown_grace_seconds=shutdown_grace_seconds,
        ),
    )

    def connection_factory(_settings: Settings) -> FakeConnection:
        connection = FakeConnection(state)
        with state.lock:
            state.connections.append(connection)
        return connection

    return GenerationCoordinator(
        settings=settings,
        connection_factory=connection_factory,
        processor_factory=processors.factory,
    )


def _wait_until(predicate: Any, *, timeout: float = 1.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition was not met before timeout")
