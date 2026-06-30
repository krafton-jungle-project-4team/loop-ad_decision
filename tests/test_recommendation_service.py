from datetime import date
from decimal import Decimal

import pytest

from app.decision.errors import ConfigurationError
from app.decision.models import RootCauseCandidate, SegmentAnomaly
from app.decision.services import RecommendationService
from tests.fakes import InMemoryDecisionRepository


ANALYSIS_DATE = date(2021, 1, 4)


class SegmentWriteGuardRepository(InMemoryDecisionRepository):
    def insert_segment(self, *args, **kwargs):
        raise AssertionError("RecommendationService must not insert segments")

    def upsert_segments(self, *args, **kwargs):
        raise AssertionError("RecommendationService must not upsert segments")

    def update_segment(self, *args, **kwargs):
        raise AssertionError("RecommendationService must not update segments")

    def delete_segment(self, *args, **kwargs):
        raise AssertionError("RecommendationService must not delete segments")


def anomaly(*, anomaly_id: int = 1, segment_id: int = 10, evidence: dict | None = None) -> SegmentAnomaly:
    return SegmentAnomaly(
        id=anomaly_id,
        project_id=1,
        segment_id=segment_id,
        analysis_date=ANALYSIS_DATE,
        metric_name="view_to_purchase_rate",
        severity="high",
        impact_score=Decimal("0.8"),
        status="detected",
        evidence_json=evidence or {},
    )


def root_cause(
    *,
    cause_id: int = 1,
    anomaly_id: int = 1,
    cause_type: str = "funnel_step_drop",
    cause_key: str = "view_to_cart",
    evidence: dict | None = None,
) -> RootCauseCandidate:
    return RootCauseCandidate(
        id=cause_id,
        anomaly_id=anomaly_id,
        cause_type=cause_type,
        cause_key=cause_key,
        title=f"{cause_key} title",
        description=None,
        confidence_score=Decimal("0.9"),
        impact_score=Decimal("0.8"),
        rank_no=1,
        evidence_json=evidence or {},
    )


def test_anomaly_absence_creates_no_recommendations() -> None:
    repo = InMemoryDecisionRepository()
    repo.add_all_action_catalog()

    actions = RecommendationService(repo).create_for_anomalies(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    assert actions == []
    assert repo.results == []
    assert repo.actions == []
    assert repo.experiments == []
    assert repo.mappings == []


def test_funnel_action_is_primary_and_idempotent() -> None:
    repo = InMemoryDecisionRepository()
    repo.add_all_action_catalog()
    repo.anomalies.append(anomaly())
    repo.root_causes.append(root_cause(cause_key="view_to_cart"))

    service = RecommendationService(repo)
    service.create_for_anomalies(project_id=1, analysis_date=ANALYSIS_DATE, run_id=1)
    service.create_for_anomalies(project_id=1, analysis_date=ANALYSIS_DATE, run_id=1)

    assert len(repo.results) == 1
    assert repo.results[0].status == "pending_content"
    assert len(repo.actions) == 1
    assert repo.actions[0].action_key == "highlight_benefit_banner"
    assert repo.actions[0].status == "recommended"


def test_recommendation_service_does_not_write_segments() -> None:
    repo = SegmentWriteGuardRepository()
    repo.add_all_action_catalog()
    repo.anomalies.append(anomaly())
    repo.root_causes.append(root_cause(cause_key="view_to_cart"))

    RecommendationService(repo).create_for_anomalies(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    assert repo.actions[0].segment_id == 10


def test_stockout_evidence_selects_single_mvp_banner() -> None:
    repo = InMemoryDecisionRepository()
    repo.add_all_action_catalog()
    repo.anomalies.append(anomaly(evidence={"stockout": True}))
    repo.root_causes.extend(
        [
            root_cause(cause_id=1, cause_type="funnel_step_drop", cause_key="view_to_cart"),
            root_cause(cause_id=2, cause_type="stockout", cause_key="stockout"),
        ]
    )

    RecommendationService(repo).create_for_anomalies(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    assert [action.action_key for action in repo.actions] == ["highlight_benefit_banner"]


def test_all_funnel_causes_select_single_mvp_banner() -> None:
    for cause_key in ("view_to_cart", "cart_to_checkout", "checkout_to_purchase"):
        repo = InMemoryDecisionRepository()
        repo.add_action_catalog("highlight_benefit_banner")
        repo.anomalies.append(anomaly())
        repo.root_causes.append(root_cause(cause_key=cause_key))

        RecommendationService(repo).create_for_anomalies(
            project_id=1,
            analysis_date=ANALYSIS_DATE,
            run_id=1,
        )

        assert [action.action_key for action in repo.actions] == ["highlight_benefit_banner"]


def test_missing_action_catalog_raises_configuration_error() -> None:
    repo = InMemoryDecisionRepository()
    repo.anomalies.append(anomaly())
    repo.root_causes.append(root_cause(cause_key="cart_to_checkout"))

    with pytest.raises(ConfigurationError):
        RecommendationService(repo).create_for_anomalies(
            project_id=1,
            analysis_date=ANALYSIS_DATE,
            run_id=1,
        )


def test_unmatched_root_cause_records_no_action_without_action_row() -> None:
    repo = InMemoryDecisionRepository()
    repo.add_all_action_catalog()
    repo.anomalies.append(anomaly())
    repo.root_causes.append(root_cause(cause_key="unknown_step"))

    RecommendationService(repo).create_for_anomalies(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    assert len(repo.results) == 1
    assert repo.results[0].status == "no_action"
    assert repo.actions == []
