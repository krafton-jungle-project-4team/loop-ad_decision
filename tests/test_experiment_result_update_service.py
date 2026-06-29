from datetime import date
from decimal import Decimal

from app.decision.models import (
    Experiment,
    ExperimentVariant,
    RecommendationAction,
    RecommendationResult,
    SegmentAdMapping,
    VariantPerformance,
)
from app.decision.services import (
    ExperimentConfig,
    ExperimentResultUpdateService,
    WinnerDecisionService,
)
from tests.fakes import FakeExperimentResultRepository, InMemoryDecisionRepository


ANALYSIS_DATE = date(2021, 1, 4)


def seed_running_experiment(
    repo: InMemoryDecisionRepository,
    *,
    experiment_id: int = 1,
    status: str = "running",
) -> tuple[Experiment, ExperimentVariant, ExperimentVariant]:
    result = RecommendationResult(
        id=experiment_id,
        project_id=1,
        segment_id=10,
        anomaly_id=experiment_id,
        primary_root_cause_id=1,
        analysis_date=ANALYSIS_DATE,
        summary="summary",
        status="experiment_running",
        recommendation_json={},
    )
    action = RecommendationAction(
        id=experiment_id,
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
        status="running",
        metadata={},
    )
    experiment = Experiment(
        id=experiment_id,
        project_id=1,
        segment_id=10,
        recommendation_action_id=action.id,
        name="experiment",
        objective_metric="click_to_purchase_rate",
        target_value=Decimal("0.05"),
        allocation_policy="fixed_split",
        status=status,
        start_date=ANALYSIS_DATE,
    )
    control = ExperimentVariant(
        id=experiment_id * 10 + 1,
        experiment_id=experiment.id,
        project_id=1,
        variant_key="control",
        name="control",
        generated_content_id=100,
        is_control=True,
        traffic_weight=Decimal("0.5"),
        impression_count=0,
        click_count=0,
        conversion_count=0,
        ctr=Decimal("0"),
        conversion_rate=Decimal("0"),
        status="active",
    )
    treatment = ExperimentVariant(
        id=experiment_id * 10 + 2,
        experiment_id=experiment.id,
        project_id=1,
        variant_key="treatment_a",
        name="treatment",
        generated_content_id=200,
        is_control=False,
        traffic_weight=Decimal("0.5"),
        impression_count=0,
        click_count=0,
        conversion_count=0,
        ctr=Decimal("0"),
        conversion_rate=Decimal("0"),
        status="active",
    )
    repo.results.append(result)
    repo.actions.append(action)
    repo.experiments.append(experiment)
    repo.variants.extend([control, treatment])
    for variant in (control, treatment):
        repo.mappings.append(
            SegmentAdMapping(
                id=variant.id,
                project_id=1,
                segment_id=10,
                placement_key="main_banner",
                experiment_id=experiment.id,
                experiment_variant_id=variant.id,
                generated_content_id=variant.generated_content_id,
                traffic_weight=Decimal("0.5"),
                is_active=True,
                is_winner=False,
            )
        )
    return experiment, control, treatment


def update_service(
    repo: InMemoryDecisionRepository,
    clickhouse: FakeExperimentResultRepository,
    *,
    config: ExperimentConfig | None = None,
) -> ExperimentResultUpdateService:
    return ExperimentResultUpdateService(
        repo,
        clickhouse,
        winner_service=WinnerDecisionService(config or ExperimentConfig()),
    )


def test_update_running_only_and_zero_denominators_are_zero() -> None:
    repo = InMemoryDecisionRepository()
    _, control, treatment = seed_running_experiment(repo, experiment_id=1, status="running")
    seed_running_experiment(repo, experiment_id=2, status="draft")
    seed_running_experiment(repo, experiment_id=3, status="paused")
    seed_running_experiment(repo, experiment_id=4, status="completed")
    seed_running_experiment(repo, experiment_id=5, status="winner_selected")
    clickhouse = FakeExperimentResultRepository()
    clickhouse.results = {
        control.id: VariantPerformance(control.id, 0, 0, 0),
        treatment.id: VariantPerformance(treatment.id, 100, 0, 0),
    }

    update_service(repo, clickhouse).update_running(
        project_id=1,
        analysis_date=ANALYSIS_DATE,
    )

    assert [call[0] for call in clickhouse.calls] == [1]
    assert control.ctr == Decimal("0")
    assert control.conversion_rate == Decimal("0")
    assert treatment.ctr == Decimal("0")
    assert treatment.conversion_rate == Decimal("0")
    assert repo.experiments[0].status == "running"


def test_treatment_winner_marks_action_won_and_mappings() -> None:
    repo = InMemoryDecisionRepository()
    experiment, control, treatment = seed_running_experiment(repo)
    clickhouse = FakeExperimentResultRepository()
    clickhouse.results = {
        control.id: VariantPerformance(control.id, 100, 30, 2),
        treatment.id: VariantPerformance(treatment.id, 100, 30, 3),
    }

    update_service(repo, clickhouse).update_running(project_id=1, analysis_date=ANALYSIS_DATE)

    assert experiment.status == "winner_selected"
    assert repo.results[0].status == "winner_selected"
    assert repo.actions[0].status == "won"
    assert treatment.status == "winner"
    assert control.status == "loser"
    winner_mapping = next(mapping for mapping in repo.mappings if mapping.experiment_variant_id == treatment.id)
    loser_mapping = next(mapping for mapping in repo.mappings if mapping.experiment_variant_id == control.id)
    assert winner_mapping.is_active is True
    assert winner_mapping.is_winner is True
    assert winner_mapping.traffic_weight == Decimal("1")
    assert loser_mapping.is_active is False
    assert loser_mapping.traffic_weight == Decimal("0")


def test_control_winner_marks_action_lost() -> None:
    repo = InMemoryDecisionRepository()
    _, control, treatment = seed_running_experiment(repo)
    clickhouse = FakeExperimentResultRepository()
    clickhouse.results = {
        control.id: VariantPerformance(control.id, 100, 30, 3),
        treatment.id: VariantPerformance(treatment.id, 100, 30, 2),
    }

    update_service(repo, clickhouse).update_running(project_id=1, analysis_date=ANALYSIS_DATE)

    assert repo.actions[0].status == "lost"
    assert control.status == "winner"
    assert treatment.status == "loser"


def test_conversion_rate_tie_breaks_by_conversion_count() -> None:
    repo = InMemoryDecisionRepository()
    _, control, treatment = seed_running_experiment(repo)
    clickhouse = FakeExperimentResultRepository()
    clickhouse.results = {
        control.id: VariantPerformance(control.id, 100, 30, 3),
        treatment.id: VariantPerformance(treatment.id, 100, 40, 4),
    }

    update_service(repo, clickhouse).update_running(project_id=1, analysis_date=ANALYSIS_DATE)

    assert treatment.status == "winner"
    assert control.status == "loser"


def test_exact_tie_defers_winner_and_keeps_running() -> None:
    repo = InMemoryDecisionRepository()
    experiment, control, treatment = seed_running_experiment(repo)
    clickhouse = FakeExperimentResultRepository()
    clickhouse.results = {
        control.id: VariantPerformance(control.id, 100, 30, 3),
        treatment.id: VariantPerformance(treatment.id, 100, 30, 3),
    }

    update_service(repo, clickhouse).update_running(project_id=1, analysis_date=ANALYSIS_DATE)

    assert experiment.status == "running"
    assert control.status == "active"
    assert treatment.status == "active"
    assert repo.actions[0].status == "running"


def test_winner_selected_is_not_reupdated_or_flipped() -> None:
    repo = InMemoryDecisionRepository()
    experiment, control, treatment = seed_running_experiment(repo)
    clickhouse = FakeExperimentResultRepository()
    clickhouse.results = {
        control.id: VariantPerformance(control.id, 100, 30, 2),
        treatment.id: VariantPerformance(treatment.id, 100, 30, 3),
    }
    service = update_service(repo, clickhouse)
    service.update_running(project_id=1, analysis_date=ANALYSIS_DATE)
    clickhouse.results = {
        control.id: VariantPerformance(control.id, 100, 30, 20),
        treatment.id: VariantPerformance(treatment.id, 100, 30, 1),
    }

    service.update_running(project_id=1, analysis_date=ANALYSIS_DATE)

    assert experiment.winner_variant_id == treatment.id
    assert repo.actions[0].status == "won"
    assert [call[0] for call in clickhouse.calls] == [1]
