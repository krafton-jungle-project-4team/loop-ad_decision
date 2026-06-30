from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Protocol

from app.analysis.clickhouse_repository import ClickHouseAnalysisRepository
from app.analysis.models import AnalysisResult
from app.analysis.postgres_repository import PostgresAnalysisRepository
from app.analysis.service import AnalysisService
from app.analysis.time_window import build_analysis_window
from app.config import Settings
from app.decision.repositories import (
    ClickHouseExperimentResultRepository,
    ClickHouseUserSegmentCandidateRepository,
    PostgresDecisionRepository,
)
from app.decision.services import (
    ExperimentConfig,
    ExperimentResultUpdateService,
    RecommendationService,
    UserSegmentMatchingService,
)
from app.dependencies import connect_clickhouse, connect_postgres
from app.jobs.daily_analysis import run_daily_analysis_flow


class ProjectNotFoundError(LookupError):
    pass


class Cursor(Protocol):
    def execute(self, query: str, parameters: tuple[object, ...] = ()) -> object:
        ...

    def fetchone(self) -> tuple[object, ...] | None:
        ...


class Connection(Protocol):
    def cursor(self) -> object:
        ...

    def commit(self) -> None:
        ...

    def rollback(self) -> None:
        ...


@dataclass(frozen=True)
class DecisionRunRequest:
    project_key: str
    analysis_date: date
    mode: str
    force: bool
    run_type: str
    trigger_source: str
    requested_by: str | None = None


@dataclass(frozen=True)
class DecisionRunHandle:
    run_id: int
    project_id: int
    project_key: str
    analysis_date: date
    status: str


class DailyDecisionJobService:
    def __init__(
        self,
        *,
        postgres_connection_factory: Callable[[], Connection],
        clickhouse_client_factory: Callable[[], object],
    ) -> None:
        self.postgres_connection_factory = postgres_connection_factory
        self.clickhouse_client_factory = clickhouse_client_factory

    def start_run(self, request: DecisionRunRequest) -> DecisionRunHandle:
        with self.postgres_connection_factory() as connection:
            project_id, timezone = self._resolve_project(connection, request.project_key)
            window = build_analysis_window(request.analysis_date, timezone)
            baseline_window = build_analysis_window(
                request.analysis_date - timedelta(days=7),
                timezone,
            )
            idempotency_key = (
                None
                if request.force
                else (
                    f"{request.run_type}:{request.project_key}:"
                    f"{request.analysis_date.isoformat()}:{request.mode}"
                )
            )
            with connection.cursor() as cursor:
                cursor.execute(
                    CREATE_DECISION_RUN_SQL,
                    (
                        project_id,
                        request.run_type,
                        request.trigger_source,
                        request.requested_by,
                        idempotency_key,
                        request.mode,
                        request.force,
                        request.analysis_date,
                        window.window_start,
                        window.window_end,
                        baseline_window.window_start,
                        window.window_start,
                        json.dumps(
                            {
                                "project_key": request.project_key,
                                "service": "decision-api",
                            },
                            sort_keys=True,
                        ),
                    ),
                )
                row = cursor.fetchone()
            if row is None:
                raise RuntimeError("decision run was not created")
            return DecisionRunHandle(
                run_id=int(row[0]),
                project_id=project_id,
                project_key=request.project_key,
                analysis_date=request.analysis_date,
                status=str(row[1]),
            )

    def execute_run(self, run_id: int) -> AnalysisResult:
        connection = self.postgres_connection_factory()
        try:
            with connection:
                run_context = self._get_run_context(connection, run_id)
                self._mark_running(connection, run_id)

                postgres_analysis_repository = PostgresAnalysisRepository(connection)
                decision_repository = PostgresDecisionRepository(connection)
                clickhouse_client = self.clickhouse_client_factory()

                analysis_service = AnalysisService(
                    project_repository=postgres_analysis_repository,
                    segment_aggregate_repository=ClickHouseAnalysisRepository(clickhouse_client),
                    segment_metrics_repository=postgres_analysis_repository,
                    anomaly_repository=postgres_analysis_repository,
                )
                experiment_config = ExperimentConfig.for_mode(run_context.mode)
                result = run_daily_analysis_flow(
                    project_id=run_context.project_id,
                    analysis_date=run_context.analysis_date,
                    run_id=run_id,
                    analysis_service=analysis_service,
                    user_segment_matching_runner=UserSegmentMatchingService(
                        repository=decision_repository,
                        candidate_repository=ClickHouseUserSegmentCandidateRepository(
                            clickhouse_client
                        ),
                    ),
                    experiment_result_update_runner=_ExperimentResultUpdateRunner(
                        repository=decision_repository,
                        result_repository=ClickHouseExperimentResultRepository(
                            clickhouse_client
                        ),
                        config=experiment_config,
                    ),
                    downstream_runner=_RecommendationExperimentRunner(
                        repository=decision_repository,
                        project_id=run_context.project_id,
                        analysis_date=run_context.analysis_date,
                        run_id=run_id,
                        config=experiment_config,
                    ),
                )
                self._mark_success(connection, run_id, result)
                return result
        except Exception as exc:
            self._record_failure(run_id, exc)
            raise

    def _resolve_project(self, connection: Connection, project_key: str) -> tuple[int, str]:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, timezone FROM projects WHERE project_key = %s",
                (project_key,),
            )
            row = cursor.fetchone()
        if row is None:
            raise ProjectNotFoundError(f"project not found: {project_key}")
        return int(row[0]), str(row[1])

    def _get_run_context(self, connection: Connection, run_id: int) -> "_RunContext":
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT project_id, analysis_date, mode FROM decision_runs WHERE id = %s",
                (run_id,),
            )
            row = cursor.fetchone()
        if row is None:
            raise LookupError(f"decision run not found: {run_id}")
        return _RunContext(project_id=int(row[0]), analysis_date=row[1], mode=str(row[2]))

    def _mark_running(self, connection: Connection, run_id: int) -> None:
        with connection.cursor() as cursor:
            cursor.execute(
                "UPDATE decision_runs SET status = 'running', error_message = NULL WHERE id = %s",
                (run_id,),
            )

    def _mark_success(
        self,
        connection: Connection,
        run_id: int,
        result: AnalysisResult,
    ) -> None:
        metadata = {
            "segment_count": result.segment_count,
            "membership_count": result.membership_count,
            "metric_count": result.metric_count,
            "anomaly_count": result.anomaly_count,
            "root_cause_count": result.root_cause_count,
        }
        with connection.cursor() as cursor:
            cursor.execute(
                """
                UPDATE decision_runs
                SET status = 'success',
                    error_message = NULL,
                    finished_at = now(),
                    metadata = metadata || %s::jsonb
                WHERE id = %s
                """.strip(),
                (json.dumps(metadata, sort_keys=True), run_id),
            )

    def _record_failure(self, run_id: int, exc: Exception) -> None:
        failure_connection = self.postgres_connection_factory()
        with failure_connection:
            with failure_connection.cursor() as cursor:
                cursor.execute(
                    """
                    UPDATE decision_runs
                    SET status = 'failed',
                        error_message = %s,
                        finished_at = now()
                    WHERE id = %s
                    """.strip(),
                    (str(exc)[:4000], run_id),
                )


@dataclass(frozen=True)
class _RunContext:
    project_id: int
    analysis_date: date
    mode: str


class _ExperimentResultUpdateRunner:
    def __init__(
        self,
        *,
        repository: PostgresDecisionRepository,
        result_repository: ClickHouseExperimentResultRepository,
        config: ExperimentConfig,
    ) -> None:
        self.service = ExperimentResultUpdateService(
            repository=repository,
            result_repository=result_repository,
        )
        self.config = config

    def run(self, *, project_id: int, analysis_date: date) -> object:
        return self.service.update_running(
            project_id=project_id,
            analysis_date=analysis_date,
            config=self.config,
        )


class _RecommendationExperimentRunner:
    def __init__(
        self,
        *,
        repository: PostgresDecisionRepository,
        project_id: int,
        analysis_date: date,
        run_id: int,
        config: ExperimentConfig,
    ) -> None:
        self.repository = repository
        self.project_id = project_id
        self.analysis_date = analysis_date
        self.run_id = run_id
        self.config = config

    def run(self, result: AnalysisResult) -> None:
        RecommendationService(self.repository).create_for_anomalies(
            project_id=self.project_id,
            analysis_date=self.analysis_date,
            run_id=self.run_id,
        )


def build_daily_decision_job_service(settings: Settings) -> DailyDecisionJobService:
    return DailyDecisionJobService(
        postgres_connection_factory=lambda: connect_postgres(settings),
        clickhouse_client_factory=lambda: connect_clickhouse(settings),
    )


CREATE_DECISION_RUN_SQL = """
INSERT INTO decision_runs (
    project_id,
    run_type,
    trigger_source,
    requested_by,
    idempotency_key,
    mode,
    force,
    analysis_date,
    window_start,
    window_end,
    baseline_start,
    baseline_end,
    status,
    metadata
) VALUES (
    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'running', %s::jsonb
)
ON CONFLICT (project_id, idempotency_key)
WHERE idempotency_key IS NOT NULL
DO UPDATE SET
    run_type = EXCLUDED.run_type,
    trigger_source = EXCLUDED.trigger_source,
    requested_by = EXCLUDED.requested_by,
    mode = EXCLUDED.mode,
    force = EXCLUDED.force,
    analysis_date = EXCLUDED.analysis_date,
    window_start = EXCLUDED.window_start,
    window_end = EXCLUDED.window_end,
    baseline_start = EXCLUDED.baseline_start,
    baseline_end = EXCLUDED.baseline_end,
    status = 'running',
    error_message = NULL,
    metadata = EXCLUDED.metadata,
    started_at = now(),
    finished_at = NULL
RETURNING id, status
""".strip()
