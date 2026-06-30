import json
from datetime import date
from decimal import Decimal

import pytest

from app.analysis.models import AnalysisResult
from app.contents.types import ContentGenerationActionResult, ContentGenerationSummary
from app.decision.models import RootCauseCandidate, SegmentAnomaly
from app.decision.services import ExperimentConfig
from app.jobs.decision_job import DailyDecisionJobService, _RecommendationExperimentRunner
from tests.fakes import InMemoryDecisionRepository


ANALYSIS_DATE = date(2021, 1, 4)


class RecordingContentGenerationService:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def generate_for_actions(
        self,
        *,
        project_id: int,
        analysis_date: date,
        run_id: int | None = None,
        force: bool = False,
    ) -> ContentGenerationSummary:
        self.calls.append(
            {
                "project_id": project_id,
                "analysis_date": analysis_date,
                "run_id": run_id,
                "force": force,
            }
        )
        return ContentGenerationSummary(
            actions_seen=1,
            actions_created=1,
            created_actions=1,
            variants_created=2,
            created_contents=2,
            mock_calls=2,
            results=[
                ContentGenerationActionResult(
                    recommendation_action_id=1,
                    status="created",
                    created_variant_keys=("control", "treatment_a"),
                    mock_calls=2,
                )
            ],
        )


class FailingContentGenerationService:
    def generate_for_actions(
        self,
        *,
        project_id: int,
        analysis_date: date,
        run_id: int | None = None,
        force: bool = False,
    ) -> ContentGenerationSummary:
        del project_id, analysis_date, run_id, force
        raise RuntimeError("content database unavailable")


class FakeCursor:
    def __init__(self) -> None:
        self.executed: list[tuple[str, tuple[object, ...]]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def execute(self, query: str, parameters: tuple[object, ...] = ()) -> None:
        self.executed.append((query, parameters))


class FakeConnection:
    def __init__(self) -> None:
        self.cursor_instance = FakeCursor()

    def cursor(self) -> FakeCursor:
        return self.cursor_instance


def test_downstream_runner_creates_recommendations_without_immediate_experiment_sync() -> None:
    repo = InMemoryDecisionRepository()
    repo.add_all_action_catalog()
    repo.anomalies.append(
        SegmentAnomaly(
            id=1,
            project_id=1,
            segment_id=10,
            analysis_date=ANALYSIS_DATE,
            metric_name="view_to_purchase_rate",
            severity="high",
            impact_score=Decimal("0.8"),
            status="detected",
            evidence_json={},
        )
    )
    repo.root_causes.append(
        RootCauseCandidate(
            id=1,
            anomaly_id=1,
            cause_type="funnel_step_drop",
            cause_key="view_to_cart",
            title="view_to_cart title",
            description=None,
            confidence_score=Decimal("0.9"),
            impact_score=Decimal("0.8"),
            rank_no=1,
            evidence_json={},
        )
    )

    _RecommendationExperimentRunner(
        repository=repo,
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
        config=ExperimentConfig(),
    ).run(AnalysisResult(anomaly_count=1))

    assert len(repo.actions) == 1
    assert repo.actions[0].status == "recommended"
    assert repo.experiments == []
    assert repo.variants == []
    assert repo.mappings == []


def test_downstream_runner_generates_content_after_recommendations() -> None:
    repo = InMemoryDecisionRepository()
    repo.add_all_action_catalog()
    repo.anomalies.append(
        SegmentAnomaly(
            id=1,
            project_id=1,
            segment_id=10,
            analysis_date=ANALYSIS_DATE,
            metric_name="view_to_purchase_rate",
            severity="high",
            impact_score=Decimal("0.8"),
            status="detected",
            evidence_json={},
        )
    )
    repo.root_causes.append(
        RootCauseCandidate(
            id=1,
            anomaly_id=1,
            cause_type="funnel_step_drop",
            cause_key="view_to_cart",
            title="view_to_cart title",
            description=None,
            confidence_score=Decimal("0.9"),
            impact_score=Decimal("0.8"),
            rank_no=1,
            evidence_json={},
        )
    )
    content_generation_service = RecordingContentGenerationService()

    runner = _RecommendationExperimentRunner(
        repository=repo,
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=77,
        config=ExperimentConfig(),
        force=True,
        content_generation_service=content_generation_service,
    )
    runner.run(AnalysisResult(anomaly_count=1))

    assert len(repo.actions) == 1
    assert content_generation_service.calls == [
        {
            "project_id": 1,
            "analysis_date": ANALYSIS_DATE,
            "run_id": 77,
            "force": True,
        }
    ]
    assert runner.content_generation_metadata is not None
    assert runner.content_generation_metadata["status"] == "success"
    assert runner.content_generation_metadata["actions_seen"] == 1
    assert runner.content_generation_metadata["created_contents"] == 2
    assert runner.content_generation_metadata["mock_calls"] == 2


def test_downstream_runner_propagates_content_generation_service_failure() -> None:
    repo = InMemoryDecisionRepository()
    repo.add_all_action_catalog()

    runner = _RecommendationExperimentRunner(
        repository=repo,
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=77,
        config=ExperimentConfig(),
        content_generation_service=FailingContentGenerationService(),
    )

    with pytest.raises(RuntimeError, match="content database unavailable"):
        runner.run(AnalysisResult(anomaly_count=1))

    assert runner.content_generation_metadata is None


def test_mark_success_stores_content_generation_metadata() -> None:
    connection = FakeConnection()
    service = DailyDecisionJobService(
        postgres_connection_factory=lambda: connection,
        clickhouse_client_factory=lambda: object(),
    )

    service._mark_success(
        connection,
        99,
        AnalysisResult(anomaly_count=1, root_cause_count=1),
        content_generation_metadata={
            "status": "success",
            "created_contents": 2,
            "mock_calls": 2,
        },
    )

    query, parameters = connection.cursor_instance.executed[0]
    metadata = json.loads(str(parameters[0]))
    assert "UPDATE decision_runs" in query
    assert parameters[1] == 99
    assert metadata["anomaly_count"] == 1
    assert metadata["root_cause_count"] == 1
    assert metadata["content_generation"] == {
        "status": "success",
        "created_contents": 2,
        "mock_calls": 2,
    }
