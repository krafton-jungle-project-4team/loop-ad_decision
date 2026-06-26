from contextlib import contextmanager
from types import SimpleNamespace

from app.analysis.worker import process_one_analysis_job
from app.persistence.job_statuses import (
    ANALYSIS_JOB_STATUS_QUEUED,
    ANALYSIS_JOB_STATUS_RUNNING,
)
from app.persistence.repository import PostgresRepository


VALID_REQUEST_JSON = {
    "project_id": "loopad-demo-shop",
    "window_start": "2026-06-24T17:00:00+09:00",
    "window_end": "2026-06-24T18:00:00+09:00",
    "top_n": 5,
}


class FakeSession:
    def __init__(self, index: int) -> None:
        self.index = index


class FakeSessionFactory:
    def __init__(self) -> None:
        self.sessions: list[FakeSession] = []

    @contextmanager
    def __call__(self):
        session = FakeSession(len(self.sessions))
        self.sessions.append(session)
        yield session


class FakeRepositoryFactory:
    def __init__(self, job: SimpleNamespace | None) -> None:
        self.job = job
        self.repositories: list[FakeWorkerRepository] = []
        self.done: tuple[int, int] | None = None
        self.failed: tuple[int, str] | None = None
        self.events: list[tuple[str, int]] = []

    def __call__(self, session: FakeSession) -> "FakeWorkerRepository":
        repository = FakeWorkerRepository(self, session)
        self.repositories.append(repository)
        return repository


class FakeWorkerRepository:
    def __init__(self, factory: FakeRepositoryFactory, session: FakeSession) -> None:
        self.factory = factory
        self.session = session

    def claim_next_analysis_job(self) -> SimpleNamespace | None:
        self.factory.events.append(("claim", self.session.index))
        return self.factory.job if self.session.index == 0 else None

    def mark_analysis_job_done(
        self,
        job_id: int,
        recommendation_result_id: int,
    ) -> None:
        self.factory.done = (job_id, recommendation_result_id)
        self.factory.events.append(("done", self.session.index))

    def mark_analysis_job_failed(self, job_id: int, error_message: str) -> None:
        self.factory.failed = (job_id, error_message)
        self.factory.events.append(("failed", self.session.index))

    def commit(self) -> None:
        self.factory.events.append(("commit", self.session.index))

    def rollback(self) -> None:
        self.factory.events.append(("rollback", self.session.index))


@contextmanager
def fake_clickhouse_client():
    yield object()


def test_process_one_analysis_job_marks_done_with_separate_transactions() -> None:
    session_factory = FakeSessionFactory()
    repository_factory = FakeRepositoryFactory(
        SimpleNamespace(id=7, request_json=VALID_REQUEST_JSON),
    )

    def fake_analysis_runner(**kwargs):
        kwargs["persistence_repository"].commit()
        return SimpleNamespace(recommendation_result_id=25)

    processed = process_one_analysis_job(
        session_factory=session_factory,
        clickhouse_client_factory=fake_clickhouse_client,
        repository_factory=repository_factory,
        analysis_runner=fake_analysis_runner,
        metrics_repository_factory=lambda client: object(),
        root_cause_repository_factory=lambda client: object(),
    )

    assert processed is True
    assert repository_factory.done == (7, 25)
    assert repository_factory.failed is None
    assert len(session_factory.sessions) == 3
    assert repository_factory.events == [
        ("claim", 0),
        ("commit", 0),
        ("commit", 1),
        ("done", 2),
        ("commit", 2),
    ]


def test_process_one_analysis_job_marks_failed_and_truncates_error_message() -> None:
    session_factory = FakeSessionFactory()
    repository_factory = FakeRepositoryFactory(
        SimpleNamespace(id=8, request_json=VALID_REQUEST_JSON),
    )

    def fake_analysis_runner(**kwargs):
        raise ValueError("x" * 1500)

    processed = process_one_analysis_job(
        session_factory=session_factory,
        clickhouse_client_factory=fake_clickhouse_client,
        repository_factory=repository_factory,
        analysis_runner=fake_analysis_runner,
        metrics_repository_factory=lambda client: object(),
        root_cause_repository_factory=lambda client: object(),
    )

    assert processed is True
    assert repository_factory.done is None
    assert repository_factory.failed is not None
    job_id, error_message = repository_factory.failed
    assert job_id == 8
    assert error_message.startswith("ValueError: ")
    assert len(error_message) == 1000
    assert len(session_factory.sessions) == 3
    assert repository_factory.events == [
        ("claim", 0),
        ("commit", 0),
        ("rollback", 1),
        ("failed", 2),
        ("commit", 2),
    ]


def test_claim_next_analysis_job_updates_queued_job_to_running() -> None:
    job = SimpleNamespace(
        status=ANALYSIS_JOB_STATUS_QUEUED,
        attempts=0,
        locked_at=None,
        started_at=None,
        error_message="old error",
    )

    class FakeSqlAlchemySession:
        def __init__(self) -> None:
            self.statement = None
            self.flushed = False

        def scalar(self, statement):
            self.statement = statement
            return job

        def flush(self) -> None:
            self.flushed = True

    session = FakeSqlAlchemySession()
    repository = PostgresRepository(session)  # type: ignore[arg-type]

    claimed_job = repository.claim_next_analysis_job()

    assert claimed_job is job
    assert job.status == ANALYSIS_JOB_STATUS_RUNNING
    assert job.attempts == 1
    assert job.locked_at is not None
    assert job.started_at is not None
    assert job.error_message is None
    assert session.flushed is True
    assert session.statement._for_update_arg.skip_locked is True
