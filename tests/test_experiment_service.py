from datetime import date
from decimal import Decimal

from app.decision.models import (
    Experiment,
    GeneratedContent,
    RecommendationAction,
    RecommendationResult,
    SegmentAdMapping,
)
from app.decision.services import ExperimentService
from tests.fakes import InMemoryDecisionRepository


ANALYSIS_DATE = date(2021, 1, 4)


class SegmentWriteGuardRepository(InMemoryDecisionRepository):
    def insert_segment(self, *args, **kwargs):
        raise AssertionError("ExperimentService must not insert segments")

    def upsert_segments(self, *args, **kwargs):
        raise AssertionError("ExperimentService must not upsert segments")

    def update_segment(self, *args, **kwargs):
        raise AssertionError("ExperimentService must not update segments")

    def delete_segment(self, *args, **kwargs):
        raise AssertionError("ExperimentService must not delete segments")


def seed_action(
    repo: InMemoryDecisionRepository,
    *,
    status: str = "content_generated",
) -> RecommendationAction:
    result = RecommendationResult(
        id=1,
        project_id=1,
        segment_id=10,
        anomaly_id=1,
        primary_root_cause_id=1,
        analysis_date=ANALYSIS_DATE,
        summary="summary",
        status="pending_content",
        recommendation_json={},
    )
    action = RecommendationAction(
        id=1,
        recommendation_result_id=result.id,
        project_id=1,
        segment_id=10,
        action_catalog_id=1,
        action_key="highlight_benefit_banner",
        title="Highlight",
        description=None,
        priority=1,
        expected_effect_metric="view_to_purchase_rate",
        expected_effect_direction="increase",
        expected_effect_value=None,
        status=status,
        metadata={},
    )
    repo.results.append(result)
    repo.actions.append(action)
    repo.next_ids["action"] = 2
    return action


def test_recommended_action_is_not_synced_before_content_generation() -> None:
    repo = InMemoryDecisionRepository()
    action = seed_action(repo, status="recommended")
    add_action_content(repo, action, content_id=101, variant_key="control")
    add_action_content(repo, action, content_id=201, variant_key="treatment_a")

    synced = ExperimentService(repo).sync_for_recommendation_actions(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    assert synced == []
    assert repo.experiments == []
    assert repo.variants == []
    assert repo.mappings == []
    assert action.status == "recommended"


def add_default_content(repo: InMemoryDecisionRepository, *, content_id: int = 100) -> None:
    repo.contents.append(
        GeneratedContent(
            id=content_id,
            project_id=1,
            segment_id=999,
            recommendation_action_id=None,
            variant_key="default",
            generation_status="approved",
        )
    )


def add_action_content(
    repo: InMemoryDecisionRepository,
    action: RecommendationAction,
    *,
    content_id: int,
    variant_key: str,
    generation_status: str = "generated",
) -> None:
    repo.contents.append(
        GeneratedContent(
            id=content_id,
            project_id=1,
            segment_id=action.segment_id,
            recommendation_action_id=action.id,
            variant_key=variant_key,
            generation_status=generation_status,
        )
    )


def test_missing_treatment_content_creates_draft_and_only_deactivates_same_experiment() -> None:
    repo = InMemoryDecisionRepository()
    action = seed_action(repo)
    add_default_content(repo)
    repo.experiments.append(
        Experiment(
            id=7,
            project_id=1,
            segment_id=10,
            recommendation_action_id=action.id,
            name="old",
            objective_metric="view_to_purchase_rate",
            target_value=Decimal("0.05"),
            allocation_policy="fixed_split",
            status="running",
            start_date=ANALYSIS_DATE,
        )
    )
    repo.next_ids["experiment"] = 8
    same_experiment_mapping = SegmentAdMapping(
        id=1,
        project_id=1,
        segment_id=10,
        placement_key="main_banner",
        experiment_id=7,
        experiment_variant_id=1,
        generated_content_id=100,
        traffic_weight=Decimal("0.5"),
        is_active=True,
        is_winner=False,
    )
    default_mapping = SegmentAdMapping(
        id=2,
        project_id=1,
        segment_id=999,
        placement_key="main_banner",
        experiment_id=None,
        experiment_variant_id=None,
        generated_content_id=100,
        traffic_weight=Decimal("1"),
        is_active=True,
        is_winner=True,
    )
    other_mapping = SegmentAdMapping(
        id=3,
        project_id=1,
        segment_id=10,
        placement_key="main_banner",
        experiment_id=99,
        experiment_variant_id=99,
        generated_content_id=100,
        traffic_weight=Decimal("0.5"),
        is_active=True,
        is_winner=False,
    )
    repo.mappings.extend([same_experiment_mapping, default_mapping, other_mapping])

    ExperimentService(repo).sync_for_recommendation_actions(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    assert repo.experiments[0].status == "draft"
    assert repo.actions[0].status == "experiment_created"
    control = next(variant for variant in repo.variants if variant.variant_key == "control")
    treatment = next(variant for variant in repo.variants if variant.variant_key == "treatment_a")
    assert control.generated_content_id is None
    assert treatment.generated_content_id is None
    assert same_experiment_mapping.is_active is False
    assert default_mapping.is_active is True
    assert other_mapping.is_active is True


def test_content_ready_creates_running_experiment_variants_mappings_and_is_idempotent() -> None:
    repo = InMemoryDecisionRepository()
    action = seed_action(repo)
    add_default_content(repo, content_id=100)
    add_action_content(repo, action, content_id=101, variant_key="control")
    repo.contents.append(
        GeneratedContent(
            id=200,
            project_id=1,
            segment_id=10,
            recommendation_action_id=action.id,
            variant_key="treatment_a",
            generation_status="failed",
        )
    )
    repo.contents.append(
        GeneratedContent(
            id=201,
            project_id=1,
            segment_id=10,
            recommendation_action_id=action.id,
            variant_key="treatment_a",
            generation_status="generated",
        )
    )

    service = ExperimentService(repo)
    service.sync_for_recommendation_actions(project_id=1, analysis_date=ANALYSIS_DATE, run_id=1)
    service.sync_for_recommendation_actions(project_id=1, analysis_date=ANALYSIS_DATE, run_id=1)

    assert len(repo.experiments) == 1
    assert repo.experiments[0].status == "running"
    assert repo.experiments[0].objective_metric == "view_to_purchase_rate"
    assert repo.results[0].status == "experiment_running"
    assert repo.actions[0].status == "running"
    assert len(repo.variants) == 2
    control = next(variant for variant in repo.variants if variant.variant_key == "control")
    treatment = next(variant for variant in repo.variants if variant.variant_key == "treatment_a")
    assert control.generated_content_id == 101
    assert treatment.generated_content_id == 201
    assert len(repo.mappings) == 2
    assert all(mapping.is_active for mapping in repo.mappings)
    assert {mapping.traffic_weight for mapping in repo.mappings} == {Decimal("0.5")}


def test_default_content_is_not_used_as_action_control_content() -> None:
    repo = InMemoryDecisionRepository()
    action = seed_action(repo)
    add_default_content(repo, content_id=100)
    add_action_content(repo, action, content_id=201, variant_key="treatment_a")

    ExperimentService(repo).sync_for_recommendation_actions(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    control = next(variant for variant in repo.variants if variant.variant_key == "control")
    treatment = next(variant for variant in repo.variants if variant.variant_key == "treatment_a")
    assert repo.experiments[0].status == "draft"
    assert control.generated_content_id is None
    assert treatment.generated_content_id == 201
    assert repo.mappings == []


def test_experiment_service_does_not_write_segments() -> None:
    repo = SegmentWriteGuardRepository()
    action = seed_action(repo)
    add_default_content(repo, content_id=100)
    add_action_content(repo, action, content_id=101, variant_key="control")
    repo.contents.append(
        GeneratedContent(
            id=201,
            project_id=1,
            segment_id=10,
            recommendation_action_id=action.id,
            variant_key="treatment_a",
            generation_status="generated",
        )
    )

    ExperimentService(repo).sync_for_recommendation_actions(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
        run_id=1,
    )

    assert repo.experiments[0].segment_id == 10
