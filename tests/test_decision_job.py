from datetime import date
from decimal import Decimal

from app.analysis.models import AnalysisResult
from app.decision.models import RootCauseCandidate, SegmentAnomaly
from app.decision.services import ExperimentConfig
from app.jobs.decision_job import _RecommendationExperimentRunner
from tests.fakes import InMemoryDecisionRepository


ANALYSIS_DATE = date(2021, 1, 4)


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
