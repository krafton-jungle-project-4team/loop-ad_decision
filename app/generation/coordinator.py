from __future__ import annotations

import os
import socket
import threading
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Any, Protocol

from app.config import Settings
from app.generation.repositories import GenerationRunRepository
from app.logging import log


class GenerationJobProcessor(Protocol):
    def process(
        self,
        claimed_row: Mapping[str, Any],
        *,
        worker_id: str,
        lease_token: uuid.UUID,
    ) -> None:
        ...


ConnectionFactory = Callable[[Settings], Any]
ProcessorFactory = Callable[[], GenerationJobProcessor]


@dataclass(frozen=True, slots=True)
class ActiveLease:
    generation_id: str
    worker_id: str
    lease_token: uuid.UUID


class GenerationCoordinator:
    """Application-owned executor for durable generation jobs.

    The executor receives a fixed number of long-lived loops exactly once. Jobs
    are claimed from PostgreSQL by those loops instead of being submitted to the
    executor's unbounded work queue.
    """

    def __init__(
        self,
        *,
        settings: Settings,
        connection_factory: ConnectionFactory,
        processor_factory: ProcessorFactory,
    ) -> None:
        self._settings = settings
        self._connection_factory = connection_factory
        self._processor_factory = processor_factory

        self._worker_count = max(
            1,
            int(getattr(settings, "generation_worker_max_concurrency", 2)),
        )
        self._poll_interval_seconds = max(
            0.01,
            float(getattr(settings, "generation_poll_interval_seconds", 1.0)),
        )
        self._idle_poll_interval_seconds = max(
            self._poll_interval_seconds,
            float(
                getattr(settings, "generation_idle_poll_interval_seconds", 30.0)
            ),
        )
        self._lease_seconds = max(
            1,
            int(getattr(settings, "generation_lease_seconds", 120)),
        )
        self._heartbeat_seconds = max(
            0.01,
            float(getattr(settings, "generation_heartbeat_seconds", 30.0)),
        )
        self._max_retries = max(
            0,
            int(getattr(settings, "generation_max_retries", 3)),
        )
        self._retry_backoff_seconds = _retry_backoff_seconds(settings)
        self._shutdown_grace_seconds = max(
            0.0,
            float(getattr(settings, "generation_shutdown_grace_seconds", 20.0)),
        )
        self._worker_id = _worker_id()

        self._state_lock = threading.Lock()
        self._work_available = threading.Condition()
        self._wake_sequence = 0
        self._active_lock = threading.Lock()
        self._active_leases: dict[str, ActiveLease] = {}
        self._stop_claiming = threading.Event()
        self._stop_heartbeat = threading.Event()

        self._executor: ThreadPoolExecutor | None = None
        self._worker_futures: tuple[Future[None], ...] = ()
        self._heartbeat_future: Future[None] | None = None
        self._started = False
        self._shutdown = False
        self._accepting = False

    @property
    def accepting(self) -> bool:
        with self._state_lock:
            return self._accepting

    def start(self) -> None:
        with self._state_lock:
            if self._shutdown:
                raise RuntimeError("generation coordinator cannot restart after shutdown")
            if self._started:
                return

            executor = ThreadPoolExecutor(
                max_workers=self._worker_count + 1,
                thread_name_prefix="loop-ad-generation",
            )
            self._executor = executor
            self._started = True
            self._accepting = True

            self._heartbeat_future = executor.submit(self._heartbeat_loop)
            self._worker_futures = tuple(
                executor.submit(self._worker_loop, worker_index)
                for worker_index in range(self._worker_count)
            )

        log.info(
            "generation_coordinator_started",
            {"workerId": self._worker_id, "workerCount": self._worker_count},
        )

    def wake(self) -> None:
        if not self.accepting:
            return
        with self._work_available:
            self._wake_sequence += 1
            self._work_available.notify_all()

    def shutdown(self) -> None:
        with self._state_lock:
            if self._shutdown:
                return
            self._shutdown = True
            self._accepting = False
            started = self._started
            worker_futures = self._worker_futures
            heartbeat_future = self._heartbeat_future
            executor = self._executor

        if not started or executor is None:
            return

        # Phase one prevents new claims while active processors finish. The
        # heartbeat loop intentionally remains alive throughout this grace.
        self._stop_claiming.set()
        with self._work_available:
            self._work_available.notify_all()

        _, unfinished_workers = wait(
            worker_futures,
            timeout=self._shutdown_grace_seconds,
        )

        # Phase two stops lease renewal. An unfinished processor may still
        # return, but its eventual write is fenced by the expired lease token.
        self._stop_heartbeat.set()
        if heartbeat_future is not None:
            remaining_grace = max(
                0.0,
                self._shutdown_grace_seconds,
            )
            wait((heartbeat_future,), timeout=remaining_grace)

        executor.shutdown(wait=False, cancel_futures=True)
        log.info(
            "generation_coordinator_stopped",
            {
                "workerId": self._worker_id,
                "unfinishedWorkerCount": len(unfinished_workers),
            },
        )

    def _worker_loop(self, worker_index: int) -> None:
        processor = self._processor_factory()
        if worker_index == 0:
            self._recover_expired()

        while not self._stop_claiming.is_set():
            wake_sequence = self._current_wake_sequence()
            claimed_row = self._claim_next()
            if claimed_row is None:
                self._wait_for_work(
                    self._idle_poll_interval_seconds,
                    wake_sequence=wake_sequence,
                )
                continue

            lease = self._active_lease(claimed_row)
            self._register_active_lease(lease)
            try:
                processor.process(
                    claimed_row,
                    worker_id=lease.worker_id,
                    lease_token=lease.lease_token,
                )
            except Exception as exc:
                # The processor owns retry/failed transitions. If it exits
                # unexpectedly, the row remains running and lease recovery
                # makes it claimable again after heartbeat stops for this job.
                log.error(
                    "generation_processor_failed",
                    {
                        "generationId": lease.generation_id,
                        "workerId": lease.worker_id,
                        "err": exc,
                    },
                )
            finally:
                self._unregister_active_lease(lease)

    def _heartbeat_loop(self) -> None:
        while not self._stop_heartbeat.wait(self._heartbeat_seconds):
            self._heartbeat_active_leases()
            self._recover_expired()

    def _claim_next(self) -> Mapping[str, Any] | None:
        lease_token = uuid.uuid4()
        try:
            claimed = self._repository_operation(
                lambda repository: repository.claim_next(
                    worker_id=self._worker_id,
                    lease_token=lease_token,
                    lease_seconds=self._lease_seconds,
                )
            )
        except Exception as exc:
            log.error(
                "generation_claim_failed",
                {"workerId": self._worker_id, "err": exc},
            )
            self._wait_for_work(self._poll_interval_seconds)
            return None

        if claimed is None:
            return None
        claimed_row = dict(claimed)
        claimed_row.setdefault("worker_id", self._worker_id)
        claimed_row.setdefault("lease_token", lease_token)
        return claimed_row

    def _heartbeat_active_leases(self) -> None:
        with self._active_lock:
            active_leases = tuple(self._active_leases.values())

        for lease in active_leases:
            try:
                renewed = self._repository_operation(
                    lambda repository, active_lease=lease: repository.heartbeat(
                        generation_id=active_lease.generation_id,
                        worker_id=active_lease.worker_id,
                        lease_token=active_lease.lease_token,
                        lease_seconds=self._lease_seconds,
                    )
                )
                if not renewed:
                    self._unregister_active_lease(lease)
                    log.warn(
                        "generation_heartbeat_lease_lost",
                        {
                            "generationId": lease.generation_id,
                            "workerId": lease.worker_id,
                        },
                    )
            except Exception as exc:
                log.error(
                    "generation_heartbeat_failed",
                    {
                        "generationId": lease.generation_id,
                        "workerId": lease.worker_id,
                        "err": exc,
                    },
                )

    def _recover_expired(self) -> None:
        try:
            self._repository_operation(
                lambda repository: repository.recover_expired(
                    max_retries=self._max_retries,
                    retry_backoff_seconds=self._retry_backoff_seconds,
                    limit=100,
                )
            )
        except Exception as exc:
            log.error(
                "generation_lease_recovery_failed",
                {"workerId": self._worker_id, "err": exc},
            )

    def _repository_operation(self, operation: Callable[[Any], Any]) -> Any:
        connection = self._connection_factory(self._settings)
        try:
            repository = GenerationRunRepository(connection)
            result = operation(repository)
            connection.commit()
            return result
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _active_lease(self, claimed_row: Mapping[str, Any]) -> ActiveLease:
        generation_id = str(claimed_row.get("generation_id") or "").strip()
        if not generation_id:
            raise ValueError("claimed generation row is missing generation_id")

        worker_id = str(claimed_row.get("worker_id") or self._worker_id).strip()
        raw_token = claimed_row.get("lease_token")
        lease_token = (
            raw_token
            if isinstance(raw_token, uuid.UUID)
            else uuid.UUID(str(raw_token))
        )
        return ActiveLease(
            generation_id=generation_id,
            worker_id=worker_id,
            lease_token=lease_token,
        )

    def _register_active_lease(self, lease: ActiveLease) -> None:
        with self._active_lock:
            self._active_leases[lease.generation_id] = lease

    def _unregister_active_lease(self, lease: ActiveLease) -> None:
        with self._active_lock:
            if self._active_leases.get(lease.generation_id) == lease:
                self._active_leases.pop(lease.generation_id, None)

    def _current_wake_sequence(self) -> int:
        with self._work_available:
            return self._wake_sequence

    def _wait_for_work(
        self,
        timeout: float,
        *,
        wake_sequence: int | None = None,
    ) -> None:
        with self._work_available:
            if (
                not self._stop_claiming.is_set()
                and (
                    wake_sequence is None
                    or wake_sequence == self._wake_sequence
                )
            ):
                self._work_available.wait(timeout=timeout)


def _retry_backoff_seconds(settings: Settings) -> tuple[int, ...]:
    configured = getattr(
        settings,
        "generation_retry_backoff_seconds",
        (60, 300, 900),
    )
    if isinstance(configured, str):
        values: Sequence[Any] = configured.split(",")
    else:
        values = tuple(configured)
    parsed = tuple(max(0, int(value)) for value in values)
    return parsed or (60, 300, 900)


def _worker_id() -> str:
    hostname = socket.gethostname().strip() or "decision-api"
    suffix = uuid.uuid4().hex[:12]
    return f"{hostname}:{os.getpid()}:{suffix}"[:200]
