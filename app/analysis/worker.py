from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
import logging
import time
from typing import Any

from sqlalchemy.orm import Session

from app.analysis.schemas import FunnelRecommendationAnalysisRequest
from app.analysis.service import run_funnel_recommendation_analysis
from app.core.config import Settings, get_settings
from app.db.clickhouse import ClickHouseClient, ClickHouseClientFactory, get_clickhouse_client
from app.db.postgres import get_postgres_sessionmaker
from app.metrics.repository import FunnelMetricsRepository
from app.persistence.repository import PostgresRepository
from app.root_causes.repository import RootCauseRepository

logger = logging.getLogger(__name__)

MAX_ANALYSIS_JOB_ERROR_MESSAGE_LENGTH = 1000
ANALYSIS_WORKER_POLL_INTERVAL_SECONDS = 2.0

SessionFactory = Callable[[], AbstractContextManager[Session]]
RepositoryFactory = Callable[[Session], PostgresRepository]
AnalysisRunner = Callable[..., Any]
MetricsRepositoryFactory = Callable[[ClickHouseClient], Any]
RootCauseRepositoryFactory = Callable[[ClickHouseClient], Any]


@dataclass(frozen=True)
class ClaimedAnalysisJob:
    id: int
    request_json: dict[str, Any]


def truncate_error_message(error_message: str) -> str:
    return error_message[:MAX_ANALYSIS_JOB_ERROR_MESSAGE_LENGTH]


def format_error_message(exc: BaseException) -> str:
    return truncate_error_message(f"{type(exc).__name__}: {exc}")


def process_one_analysis_job(
    *,
    session_factory: SessionFactory,
    clickhouse_client_factory: ClickHouseClientFactory = get_clickhouse_client,
    repository_factory: RepositoryFactory = PostgresRepository,
    analysis_runner: AnalysisRunner = run_funnel_recommendation_analysis,
    metrics_repository_factory: MetricsRepositoryFactory = FunnelMetricsRepository,
    root_cause_repository_factory: RootCauseRepositoryFactory = RootCauseRepository,
) -> bool:
    claimed_job = claim_one_analysis_job(
        session_factory=session_factory,
        repository_factory=repository_factory,
    )
    if claimed_job is None:
        return False

    try:
        recommendation_result_id = run_claimed_analysis_job(
            claimed_job=claimed_job,
            session_factory=session_factory,
            clickhouse_client_factory=clickhouse_client_factory,
            repository_factory=repository_factory,
            analysis_runner=analysis_runner,
            metrics_repository_factory=metrics_repository_factory,
            root_cause_repository_factory=root_cause_repository_factory,
        )
    except Exception as exc:
        logger.exception("Analysis job failed.", extra={"job_id": claimed_job.id})
        mark_analysis_job_failed(
            job_id=claimed_job.id,
            error_message=format_error_message(exc),
            session_factory=session_factory,
            repository_factory=repository_factory,
        )
        return True

    mark_analysis_job_done(
        job_id=claimed_job.id,
        recommendation_result_id=recommendation_result_id,
        session_factory=session_factory,
        repository_factory=repository_factory,
    )
    return True


def claim_one_analysis_job(
    *,
    session_factory: SessionFactory,
    repository_factory: RepositoryFactory = PostgresRepository,
) -> ClaimedAnalysisJob | None:
    with session_factory() as session:
        repository = repository_factory(session)
        job = repository.claim_next_analysis_job()
        if job is None:
            repository.rollback()
            return None

        claimed_job = ClaimedAnalysisJob(
            id=job.id,
            request_json=dict(job.request_json),
        )
        repository.commit()
        return claimed_job


def run_claimed_analysis_job(
    *,
    claimed_job: ClaimedAnalysisJob,
    session_factory: SessionFactory,
    clickhouse_client_factory: ClickHouseClientFactory,
    repository_factory: RepositoryFactory,
    analysis_runner: AnalysisRunner,
    metrics_repository_factory: MetricsRepositoryFactory,
    root_cause_repository_factory: RootCauseRepositoryFactory,
) -> int:
    request = FunnelRecommendationAnalysisRequest.model_validate(claimed_job.request_json)
    with session_factory() as session:
        repository = repository_factory(session)
        try:
            with clickhouse_client_factory() as client:
                response = analysis_runner(
                    request=request,
                    metrics_repository=metrics_repository_factory(client),
                    root_cause_repository=root_cause_repository_factory(client),
                    persistence_repository=repository,
                )
        except Exception:
            repository.rollback()
            raise
    return response.recommendation_result_id


def mark_analysis_job_done(
    *,
    job_id: int,
    recommendation_result_id: int,
    session_factory: SessionFactory,
    repository_factory: RepositoryFactory = PostgresRepository,
) -> None:
    with session_factory() as session:
        repository = repository_factory(session)
        try:
            repository.mark_analysis_job_done(
                job_id=job_id,
                recommendation_result_id=recommendation_result_id,
            )
            repository.commit()
        except Exception:
            repository.rollback()
            raise


def mark_analysis_job_failed(
    *,
    job_id: int,
    error_message: str,
    session_factory: SessionFactory,
    repository_factory: RepositoryFactory = PostgresRepository,
) -> None:
    with session_factory() as session:
        repository = repository_factory(session)
        try:
            repository.mark_analysis_job_failed(
                job_id=job_id,
                error_message=truncate_error_message(error_message),
            )
            repository.commit()
        except Exception:
            repository.rollback()
            raise


def run_worker_loop(
    *,
    settings: Settings | None = None,
    session_factory: SessionFactory | None = None,
    clickhouse_client_factory: ClickHouseClientFactory = get_clickhouse_client,
) -> None:
    resolved_settings = settings or get_settings()
    resolved_session_factory = session_factory or get_postgres_sessionmaker(resolved_settings)

    while True:
        try:
            did_work = process_one_analysis_job(
                session_factory=resolved_session_factory,
                clickhouse_client_factory=clickhouse_client_factory,
            )
        except Exception:
            logger.exception("Analysis worker loop iteration failed.")
            did_work = False
        if not did_work:
            time.sleep(ANALYSIS_WORKER_POLL_INTERVAL_SECONDS)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    run_worker_loop()


if __name__ == "__main__":
    main()
