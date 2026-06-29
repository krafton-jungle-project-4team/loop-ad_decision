from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from typing import Protocol
from zoneinfo import ZoneInfo

from app.decision.errors import ConfigurationError
from app.decision.models import (
    ActionCatalogItem,
    Experiment,
    ExperimentVariant,
    GeneratedContent,
    RecommendationAction,
    RecommendationResult,
    RootCauseCandidate,
    SegmentAnomaly,
    VariantPerformance,
)


ACTION_BY_CAUSE_KEY = {
    "view_to_cart": "highlight_benefit_banner",
    "cart_to_checkout": "cart_coupon_banner",
    "checkout_to_purchase": "checkout_coupon_banner",
}
STOCKOUT_ACTION_KEY = "alternative_product_banner"
OBJECTIVE_METRIC = "click_to_purchase_rate"
PLACEMENT_KEY = "main_banner"


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    minimum_impressions: int = 100
    minimum_clicks: int = 30
    target_value: Decimal = Decimal("0.05")

    @classmethod
    def for_mode(cls, mode: str) -> "ExperimentConfig":
        if mode == "demo":
            return cls(minimum_impressions=10, minimum_clicks=3)
        return cls()


@dataclass(frozen=True, slots=True)
class WinnerDecision:
    winner_variant_id: int
    winner_variant_key: str


class DecisionRepository(Protocol):
    def list_detected_anomalies(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> list[SegmentAnomaly]: ...

    def list_root_causes(self, *, anomaly_id: int) -> list[RootCauseCandidate]: ...

    def get_active_action_catalog(self, *, action_key: str) -> ActionCatalogItem | None: ...

    def upsert_recommendation_result(
        self,
        *,
        project_id: int,
        segment_id: int,
        anomaly_id: int,
        primary_root_cause_id: int | None,
        analysis_date: date,
        summary: str,
        status: str,
        recommendation_json: dict,
        run_id: int,
    ) -> RecommendationResult: ...

    def upsert_recommendation_action(
        self,
        *,
        recommendation_result_id: int,
        project_id: int,
        segment_id: int,
        action_catalog_id: int | None,
        action_key: str,
        title: str,
        description: str | None,
        priority: int,
        expected_effect_metric: str,
        expected_effect_direction: str,
        expected_effect_value: Decimal | None,
        status: str,
        metadata: dict,
    ) -> RecommendationAction: ...

    def list_actions_for_experiment_sync(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> list[RecommendationAction]: ...

    def get_recommendation_result(self, *, result_id: int) -> RecommendationResult | None: ...

    def find_action_content(
        self,
        *,
        project_id: int,
        recommendation_action_id: int,
        variant_key: str,
        statuses: tuple[str, ...],
    ) -> GeneratedContent | None: ...

    def find_project_default_content(self, *, project_id: int) -> GeneratedContent | None: ...

    def get_experiment_by_recommendation_action(
        self,
        *,
        project_id: int,
        recommendation_action_id: int,
    ) -> Experiment | None: ...

    def upsert_experiment(
        self,
        *,
        project_id: int,
        segment_id: int,
        recommendation_action_id: int,
        name: str,
        objective_metric: str,
        target_value: Decimal,
        allocation_policy: str,
        status: str,
        start_date: date,
        run_id: int,
    ) -> Experiment: ...

    def upsert_experiment_variant(
        self,
        *,
        experiment_id: int,
        project_id: int,
        variant_key: str,
        name: str,
        generated_content_id: int | None,
        is_control: bool,
        traffic_weight: Decimal,
        status: str,
    ) -> ExperimentVariant: ...

    def deactivate_mappings_for_experiment(self, *, experiment_id: int) -> int: ...

    def upsert_segment_ad_mapping(
        self,
        *,
        project_id: int,
        segment_id: int,
        placement_key: str,
        experiment_id: int,
        experiment_variant_id: int,
        generated_content_id: int,
        traffic_weight: Decimal,
        is_active: bool,
        is_winner: bool,
        priority: int,
        run_id: int,
    ): ...

    def update_recommendation_result_status(self, *, result_id: int, status: str): ...

    def update_recommendation_action_status(self, *, action_id: int, status: str): ...

    def get_project_timezone(self, *, project_id: int) -> str: ...

    def list_experiments_by_status(
        self,
        *,
        project_id: int,
        status: str,
    ) -> list[Experiment]: ...

    def list_experiment_variants(self, *, experiment_id: int) -> list[ExperimentVariant]: ...

    def update_experiment_variant_results(
        self,
        *,
        variant_id: int,
        impression_count: int,
        click_count: int,
        conversion_count: int,
        ctr: Decimal,
        conversion_rate: Decimal,
    ) -> ExperimentVariant: ...

    def set_experiment_winner(
        self,
        *,
        experiment: Experiment,
        variants: list[ExperimentVariant],
        winner_variant: ExperimentVariant,
    ) -> None: ...


class ExperimentResultRepository(Protocol):
    def fetch_variant_results(
        self,
        *,
        project_id: int,
        experiment: Experiment,
        variants: list[ExperimentVariant],
        window_start: datetime,
        window_end: datetime,
    ) -> dict[int, VariantPerformance]: ...


class RecommendationService:
    def __init__(self, repository: DecisionRepository) -> None:
        self.repository = repository

    def create_for_anomalies(
        self,
        *,
        project_id: int,
        analysis_date: date,
        run_id: int,
    ) -> list[RecommendationAction]:
        created_actions: list[RecommendationAction] = []
        anomalies = self.repository.list_detected_anomalies(
            project_id=project_id,
            analysis_date=analysis_date,
        )

        for anomaly in anomalies:
            root_causes = self.repository.list_root_causes(anomaly_id=anomaly.id)
            selected = self._select_action(anomaly, root_causes)
            if selected is None:
                self.repository.upsert_recommendation_result(
                    project_id=project_id,
                    segment_id=anomaly.segment_id,
                    anomaly_id=anomaly.id,
                    primary_root_cause_id=self._primary_root_cause_id(root_causes),
                    analysis_date=analysis_date,
                    summary="No actionable recommendation was selected for this anomaly.",
                    status="no_action",
                    recommendation_json={"selected_action_keys": []},
                    run_id=run_id,
                )
                continue

            action_key, root_cause = selected
            action_catalog = self.repository.get_active_action_catalog(action_key=action_key)
            if action_catalog is None:
                raise ConfigurationError(f"missing active action_catalog row: {action_key}")

            result = self.repository.upsert_recommendation_result(
                project_id=project_id,
                segment_id=anomaly.segment_id,
                anomaly_id=anomaly.id,
                primary_root_cause_id=root_cause.id if root_cause is not None else None,
                analysis_date=analysis_date,
                summary=self._build_summary(anomaly, root_cause, action_catalog),
                status="pending_content",
                recommendation_json={
                    "selected_action_keys": [action_key],
                    "reason": root_cause.cause_key if root_cause is not None else "stockout",
                },
                run_id=run_id,
            )
            action = self.repository.upsert_recommendation_action(
                recommendation_result_id=result.id,
                project_id=project_id,
                segment_id=anomaly.segment_id,
                action_catalog_id=action_catalog.id,
                action_key=action_key,
                title=action_catalog.name,
                description=action_catalog.description,
                priority=1,
                expected_effect_metric="view_to_purchase_rate",
                expected_effect_direction="increase",
                expected_effect_value=None,
                status="recommended",
                metadata={
                    "source": "recommendation_service",
                    "root_cause_id": root_cause.id if root_cause is not None else None,
                    "anomaly_id": anomaly.id,
                },
            )
            created_actions.append(action)

        return created_actions

    def _select_action(
        self,
        anomaly: SegmentAnomaly,
        root_causes: list[RootCauseCandidate],
    ) -> tuple[str, RootCauseCandidate | None] | None:
        if self._has_stockout_evidence(anomaly, root_causes):
            return STOCKOUT_ACTION_KEY, self._first_stockout_root_cause(root_causes)

        for root_cause in sorted(root_causes, key=lambda item: (item.rank_no, -item.impact_score)):
            action_key = ACTION_BY_CAUSE_KEY.get(root_cause.cause_key)
            if action_key is not None:
                return action_key, root_cause
        return None

    def _has_stockout_evidence(
        self,
        anomaly: SegmentAnomaly,
        root_causes: list[RootCauseCandidate],
    ) -> bool:
        if self._json_mentions_stockout(anomaly.evidence_json):
            return True
        return any(
            root_cause.cause_type == "stockout"
            or root_cause.cause_key == "stockout"
            or self._json_mentions_stockout(root_cause.evidence_json)
            for root_cause in root_causes
        )

    def _json_mentions_stockout(self, value: object) -> bool:
        if isinstance(value, dict):
            return any(
                self._json_mentions_stockout(key) or self._json_mentions_stockout(item)
                for key, item in value.items()
            )
        if isinstance(value, list):
            return any(self._json_mentions_stockout(item) for item in value)
        if isinstance(value, str):
            normalized = value.lower()
            return "stockout" in normalized or "stock" in normalized
        return False

    def _first_stockout_root_cause(
        self,
        root_causes: list[RootCauseCandidate],
    ) -> RootCauseCandidate | None:
        stockout_causes = [
            root_cause
            for root_cause in root_causes
            if root_cause.cause_type == "stockout"
            or root_cause.cause_key == "stockout"
            or self._json_mentions_stockout(root_cause.evidence_json)
        ]
        if not stockout_causes:
            return None
        return sorted(stockout_causes, key=lambda item: (item.rank_no, -item.impact_score))[0]

    def _primary_root_cause_id(self, root_causes: list[RootCauseCandidate]) -> int | None:
        if not root_causes:
            return None
        return sorted(root_causes, key=lambda item: (item.rank_no, -item.impact_score))[0].id

    def _build_summary(
        self,
        anomaly: SegmentAnomaly,
        root_cause: RootCauseCandidate | None,
        action_catalog: ActionCatalogItem,
    ) -> str:
        if root_cause is None:
            return f"{anomaly.metric_name} anomaly detected. Recommend {action_catalog.name}."
        return f"{root_cause.title}: recommend {action_catalog.name}."


class ExperimentService:
    def __init__(
        self,
        repository: DecisionRepository,
        *,
        config: ExperimentConfig | None = None,
    ) -> None:
        self.repository = repository
        self.config = config or ExperimentConfig()

    def sync_for_recommendation_actions(
        self,
        *,
        project_id: int,
        analysis_date: date,
        run_id: int,
    ) -> list[Experiment]:
        synced: list[Experiment] = []
        actions = self.repository.list_actions_for_experiment_sync(
            project_id=project_id,
            analysis_date=analysis_date,
        )
        for action in actions:
            existing = self.repository.get_experiment_by_recommendation_action(
                project_id=project_id,
                recommendation_action_id=action.id,
            )
            if existing is not None and existing.status == "winner_selected":
                continue

            control_content = self._resolve_control_content(project_id, action.id)
            treatment_content = self.repository.find_action_content(
                project_id=project_id,
                recommendation_action_id=action.id,
                variant_key="treatment_a",
                statuses=("generated", "approved"),
            )
            ready = control_content is not None and treatment_content is not None
            experiment = self.repository.upsert_experiment(
                project_id=project_id,
                segment_id=action.segment_id,
                recommendation_action_id=action.id,
                name=f"{action.title} experiment",
                objective_metric=OBJECTIVE_METRIC,
                target_value=self.config.target_value,
                allocation_policy="fixed_split",
                status="running" if ready else "draft",
                start_date=analysis_date,
                run_id=run_id,
            )

            control_variant = self.repository.upsert_experiment_variant(
                experiment_id=experiment.id,
                project_id=project_id,
                variant_key="control",
                name="control",
                generated_content_id=control_content.id if control_content is not None else None,
                is_control=True,
                traffic_weight=Decimal("0.5"),
                status="active",
            )
            treatment_variant = self.repository.upsert_experiment_variant(
                experiment_id=experiment.id,
                project_id=project_id,
                variant_key="treatment_a",
                name="treatment_a",
                generated_content_id=treatment_content.id if treatment_content is not None else None,
                is_control=False,
                traffic_weight=Decimal("0.5"),
                status="active",
            )

            if ready and control_content is not None and treatment_content is not None:
                self._activate_running_experiment(
                    action=action,
                    control_content=control_content,
                    treatment_content=treatment_content,
                    experiment=experiment,
                    control_variant=control_variant,
                    treatment_variant=treatment_variant,
                    run_id=run_id,
                )
            else:
                self.repository.deactivate_mappings_for_experiment(experiment_id=experiment.id)
                self.repository.update_recommendation_action_status(
                    action_id=action.id,
                    status="experiment_created",
                )
            synced.append(experiment)

        return synced

    def _resolve_control_content(
        self,
        project_id: int,
        recommendation_action_id: int,
    ) -> GeneratedContent | None:
        action_content = self.repository.find_action_content(
            project_id=project_id,
            recommendation_action_id=recommendation_action_id,
            variant_key="control",
            statuses=("generated", "approved"),
        )
        if action_content is not None:
            return action_content
        return self.repository.find_project_default_content(project_id=project_id)

    def _activate_running_experiment(
        self,
        *,
        action: RecommendationAction,
        control_content: GeneratedContent,
        treatment_content: GeneratedContent,
        experiment: Experiment,
        control_variant: ExperimentVariant,
        treatment_variant: ExperimentVariant,
        run_id: int,
    ) -> None:
        self.repository.update_recommendation_result_status(
            result_id=action.recommendation_result_id,
            status="experiment_running",
        )
        self.repository.update_recommendation_action_status(
            action_id=action.id,
            status="running",
        )
        self.repository.upsert_segment_ad_mapping(
            project_id=action.project_id,
            segment_id=action.segment_id,
            placement_key=PLACEMENT_KEY,
            experiment_id=experiment.id,
            experiment_variant_id=control_variant.id,
            generated_content_id=control_content.id,
            traffic_weight=Decimal("0.5"),
            is_active=True,
            is_winner=False,
            priority=100,
            run_id=run_id,
        )
        self.repository.upsert_segment_ad_mapping(
            project_id=action.project_id,
            segment_id=action.segment_id,
            placement_key=PLACEMENT_KEY,
            experiment_id=experiment.id,
            experiment_variant_id=treatment_variant.id,
            generated_content_id=treatment_content.id,
            traffic_weight=Decimal("0.5"),
            is_active=True,
            is_winner=False,
            priority=100,
            run_id=run_id,
        )


class WinnerDecisionService:
    def __init__(self, config: ExperimentConfig | None = None) -> None:
        self.config = config or ExperimentConfig()

    def decide(self, variants: list[ExperimentVariant]) -> WinnerDecision | None:
        candidates = [
            variant
            for variant in variants
            if variant.click_count > 0
            and variant.impression_count >= self.config.minimum_impressions
            and variant.click_count >= self.config.minimum_clicks
            and variant.conversion_rate >= self.config.target_value
        ]
        if not candidates:
            return None

        candidates.sort(
            key=lambda item: (item.conversion_rate, item.conversion_count),
            reverse=True,
        )
        winner = candidates[0]
        if len(candidates) > 1:
            runner_up = candidates[1]
            if (
                winner.conversion_rate == runner_up.conversion_rate
                and winner.conversion_count == runner_up.conversion_count
            ):
                return None
        return WinnerDecision(
            winner_variant_id=winner.id,
            winner_variant_key=winner.variant_key,
        )


class ExperimentResultUpdateService:
    def __init__(
        self,
        repository: DecisionRepository,
        result_repository: ExperimentResultRepository,
        *,
        winner_service: WinnerDecisionService | None = None,
    ) -> None:
        self.repository = repository
        self.result_repository = result_repository
        self.winner_service = winner_service or WinnerDecisionService()

    def update_running(
        self,
        *,
        project_id: int,
        analysis_date: date,
        config: ExperimentConfig | None = None,
    ) -> list[Experiment]:
        if config is not None:
            self.winner_service = WinnerDecisionService(config)

        timezone = ZoneInfo(self.repository.get_project_timezone(project_id=project_id))
        updated_experiments: list[Experiment] = []
        for experiment in self.repository.list_experiments_by_status(
            project_id=project_id,
            status="running",
        ):
            variants = self.repository.list_experiment_variants(experiment_id=experiment.id)
            window_start = datetime.combine(experiment.start_date, time.min, tzinfo=timezone)
            window_end = datetime.combine(
                analysis_date + timedelta(days=1),
                time.min,
                tzinfo=timezone,
            )
            performance_by_variant_id = self.result_repository.fetch_variant_results(
                project_id=project_id,
                experiment=experiment,
                variants=variants,
                window_start=window_start,
                window_end=window_end,
            )
            refreshed_variants: list[ExperimentVariant] = []
            for variant in variants:
                performance = performance_by_variant_id.get(
                    variant.id,
                    VariantPerformance(
                        experiment_variant_id=variant.id,
                        ad_impression_count=0,
                        ad_click_count=0,
                        attributed_purchase_count=0,
                    ),
                )
                ctr = safe_rate(
                    performance.ad_click_count,
                    performance.ad_impression_count,
                )
                conversion_rate = safe_rate(
                    performance.attributed_purchase_count,
                    performance.ad_click_count,
                )
                refreshed_variants.append(
                    self.repository.update_experiment_variant_results(
                        variant_id=variant.id,
                        impression_count=performance.ad_impression_count,
                        click_count=performance.ad_click_count,
                        conversion_count=performance.attributed_purchase_count,
                        ctr=ctr,
                        conversion_rate=conversion_rate,
                    )
                )

            winner = self.winner_service.decide(refreshed_variants)
            if winner is not None:
                winner_variant = next(
                    variant
                    for variant in refreshed_variants
                    if variant.id == winner.winner_variant_id
                )
                self.repository.set_experiment_winner(
                    experiment=experiment,
                    variants=refreshed_variants,
                    winner_variant=winner_variant,
                )
            updated_experiments.append(experiment)

        return updated_experiments


def safe_rate(numerator: int, denominator: int) -> Decimal:
    if denominator == 0:
        return Decimal("0")
    return (Decimal(numerator) / Decimal(denominator)).quantize(Decimal("0.000001"))

