from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal

from app.decision.models import (
    ActionCatalogItem,
    Experiment,
    ExperimentVariant,
    GeneratedContent,
    RecommendationAction,
    RecommendationResult,
    RootCauseCandidate,
    SegmentAdMapping,
    SegmentAnomaly,
    VariantPerformance,
)


class InMemoryDecisionRepository:
    def __init__(self) -> None:
        self.timezone = "Asia/Seoul"
        self.project_key = "demo-shop"
        self.anomalies: list[SegmentAnomaly] = []
        self.root_causes: list[RootCauseCandidate] = []
        self.action_catalog: dict[str, ActionCatalogItem] = {}
        self.results: list[RecommendationResult] = []
        self.actions: list[RecommendationAction] = []
        self.contents: list[GeneratedContent] = []
        self.experiments: list[Experiment] = []
        self.variants: list[ExperimentVariant] = []
        self.mappings: list[SegmentAdMapping] = []
        self.next_ids: dict[str, int] = {
            "result": 1,
            "action": 1,
            "experiment": 1,
            "variant": 1,
            "mapping": 1,
        }

    def add_action_catalog(self, action_key: str, *, item_id: int | None = None) -> None:
        self.action_catalog[action_key] = ActionCatalogItem(
            id=item_id or len(self.action_catalog) + 1,
            action_key=action_key,
            name=action_key.replace("_", " ").title(),
            description=f"{action_key} description",
            target_funnel_step=None,
            default_channel="banner",
            template_json={},
        )

    def add_all_action_catalog(self) -> None:
        for action_key in [
            "highlight_benefit_banner",
            "cart_coupon_banner",
            "checkout_coupon_banner",
            "alternative_product_banner",
        ]:
            self.add_action_catalog(action_key)

    def next_id(self, key: str) -> int:
        value = self.next_ids[key]
        self.next_ids[key] += 1
        return value

    def list_detected_anomalies(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> list[SegmentAnomaly]:
        return [
            anomaly
            for anomaly in self.anomalies
            if anomaly.project_id == project_id
            and anomaly.analysis_date == analysis_date
            and anomaly.status == "detected"
        ]

    def list_root_causes(self, *, anomaly_id: int) -> list[RootCauseCandidate]:
        return sorted(
            [
                root_cause
                for root_cause in self.root_causes
                if root_cause.anomaly_id == anomaly_id
            ],
            key=lambda item: (item.rank_no, -item.impact_score),
        )

    def get_active_action_catalog(self, *, action_key: str) -> ActionCatalogItem | None:
        return self.action_catalog.get(action_key)

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
    ) -> RecommendationResult:
        for result in self.results:
            if (
                result.project_id == project_id
                and result.segment_id == segment_id
                and result.analysis_date == analysis_date
                and result.anomaly_id == anomaly_id
            ):
                result.primary_root_cause_id = primary_root_cause_id
                result.summary = summary
                if result.status in {"pending_content", "no_action"}:
                    result.status = status
                result.recommendation_json = recommendation_json
                return result

        result = RecommendationResult(
            id=self.next_id("result"),
            project_id=project_id,
            segment_id=segment_id,
            anomaly_id=anomaly_id,
            primary_root_cause_id=primary_root_cause_id,
            analysis_date=analysis_date,
            summary=summary,
            status=status,
            recommendation_json=recommendation_json,
        )
        self.results.append(result)
        return result

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
    ) -> RecommendationAction:
        for action in self.actions:
            if action.recommendation_result_id == recommendation_result_id and action.action_key == action_key:
                action.action_catalog_id = action_catalog_id
                action.title = title
                action.description = description
                action.priority = priority
                action.expected_effect_metric = expected_effect_metric
                action.expected_effect_direction = expected_effect_direction
                action.expected_effect_value = expected_effect_value
                if action.status == "recommended":
                    action.status = status
                action.metadata = metadata
                return action

        action = RecommendationAction(
            id=self.next_id("action"),
            recommendation_result_id=recommendation_result_id,
            project_id=project_id,
            segment_id=segment_id,
            action_catalog_id=action_catalog_id,
            action_key=action_key,
            title=title,
            description=description,
            priority=priority,
            expected_effect_metric=expected_effect_metric,
            expected_effect_direction=expected_effect_direction,
            expected_effect_value=expected_effect_value,
            status=status,
            metadata=metadata,
        )
        self.actions.append(action)
        return action

    def list_actions_for_experiment_sync(
        self,
        *,
        project_id: int,
        analysis_date: date,
    ) -> list[RecommendationAction]:
        result_by_id = {result.id: result for result in self.results}
        return [
            action
            for action in self.actions
            if action.project_id == project_id
            and action.status in {"recommended", "content_generated", "experiment_created", "running"}
            and result_by_id[action.recommendation_result_id].analysis_date == analysis_date
        ]

    def get_recommendation_result(self, *, result_id: int) -> RecommendationResult | None:
        return next((result for result in self.results if result.id == result_id), None)

    def find_action_content(
        self,
        *,
        project_id: int,
        recommendation_action_id: int,
        variant_key: str,
        statuses: tuple[str, ...],
    ) -> GeneratedContent | None:
        matches = [
            content
            for content in self.contents
            if content.project_id == project_id
            and content.recommendation_action_id == recommendation_action_id
            and content.variant_key == variant_key
            and content.generation_status in statuses
        ]
        return matches[-1] if matches else None

    def get_experiment_by_recommendation_action(
        self,
        *,
        project_id: int,
        recommendation_action_id: int,
    ) -> Experiment | None:
        return next(
            (
                experiment
                for experiment in self.experiments
                if experiment.project_id == project_id
                and experiment.recommendation_action_id == recommendation_action_id
            ),
            None,
        )

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
    ) -> Experiment:
        experiment = self.get_experiment_by_recommendation_action(
            project_id=project_id,
            recommendation_action_id=recommendation_action_id,
        )
        if experiment is not None:
            if experiment.status != "winner_selected":
                experiment.segment_id = segment_id
                experiment.name = name
                experiment.objective_metric = objective_metric
                experiment.target_value = target_value
                experiment.allocation_policy = allocation_policy
                experiment.status = status
            return experiment

        experiment = Experiment(
            id=self.next_id("experiment"),
            project_id=project_id,
            segment_id=segment_id,
            recommendation_action_id=recommendation_action_id,
            name=name,
            objective_metric=objective_metric,
            target_value=target_value,
            allocation_policy=allocation_policy,
            status=status,
            start_date=start_date,
        )
        self.experiments.append(experiment)
        return experiment

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
    ) -> ExperimentVariant:
        for variant in self.variants:
            if variant.experiment_id == experiment_id and variant.variant_key == variant_key:
                variant.name = name
                variant.generated_content_id = generated_content_id
                variant.is_control = is_control
                variant.traffic_weight = traffic_weight
                variant.status = status
                return variant

        variant = ExperimentVariant(
            id=self.next_id("variant"),
            experiment_id=experiment_id,
            project_id=project_id,
            variant_key=variant_key,
            name=name,
            generated_content_id=generated_content_id,
            is_control=is_control,
            traffic_weight=traffic_weight,
            impression_count=0,
            click_count=0,
            conversion_count=0,
            ctr=Decimal("0"),
            conversion_rate=Decimal("0"),
            status=status,
        )
        self.variants.append(variant)
        return variant

    def deactivate_mappings_for_experiment(self, *, experiment_id: int) -> int:
        changed = 0
        for mapping in self.mappings:
            if mapping.experiment_id == experiment_id:
                mapping.is_active = False
                mapping.traffic_weight = Decimal("0")
                mapping.is_winner = False
                changed += 1
        return changed

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
    ) -> SegmentAdMapping:
        for mapping in self.mappings:
            if (
                mapping.project_id == project_id
                and mapping.segment_id == segment_id
                and mapping.placement_key == placement_key
                and mapping.experiment_variant_id == experiment_variant_id
            ):
                mapping.experiment_id = experiment_id
                mapping.generated_content_id = generated_content_id
                mapping.traffic_weight = traffic_weight
                mapping.is_active = is_active
                mapping.is_winner = is_winner
                return mapping

        mapping = SegmentAdMapping(
            id=self.next_id("mapping"),
            project_id=project_id,
            segment_id=segment_id,
            placement_key=placement_key,
            experiment_id=experiment_id,
            experiment_variant_id=experiment_variant_id,
            generated_content_id=generated_content_id,
            traffic_weight=traffic_weight,
            is_active=is_active,
            is_winner=is_winner,
        )
        self.mappings.append(mapping)
        return mapping

    def update_recommendation_result_status(self, *, result_id: int, status: str) -> None:
        result = self.get_recommendation_result(result_id=result_id)
        if result is not None:
            result.status = status

    def update_recommendation_action_status(self, *, action_id: int, status: str) -> None:
        action = next((item for item in self.actions if item.id == action_id), None)
        if action is not None:
            action.status = status

    def get_project_timezone(self, *, project_id: int) -> str:
        return self.timezone

    def get_project_key(self, *, project_id: int) -> str:
        return self.project_key

    def list_experiments_by_status(self, *, project_id: int, status: str) -> list[Experiment]:
        return [
            experiment
            for experiment in self.experiments
            if experiment.project_id == project_id and experiment.status == status
        ]

    def list_experiment_variants(self, *, experiment_id: int) -> list[ExperimentVariant]:
        return [variant for variant in self.variants if variant.experiment_id == experiment_id]

    def update_experiment_variant_results(
        self,
        *,
        variant_id: int,
        impression_count: int,
        click_count: int,
        conversion_count: int,
        ctr: Decimal,
        conversion_rate: Decimal,
    ) -> ExperimentVariant:
        variant = next(item for item in self.variants if item.id == variant_id)
        variant.impression_count = impression_count
        variant.click_count = click_count
        variant.conversion_count = conversion_count
        variant.ctr = ctr
        variant.conversion_rate = conversion_rate
        return variant

    def set_experiment_winner(
        self,
        *,
        experiment: Experiment,
        variants: list[ExperimentVariant],
        winner_variant: ExperimentVariant,
    ) -> None:
        experiment.status = "winner_selected"
        experiment.winner_variant_id = winner_variant.id
        result_id = next(
            action.recommendation_result_id
            for action in self.actions
            if action.id == experiment.recommendation_action_id
        )
        self.update_recommendation_result_status(result_id=result_id, status="winner_selected")
        self.update_recommendation_action_status(
            action_id=experiment.recommendation_action_id,
            status="lost" if winner_variant.variant_key == "control" else "won",
        )
        for variant in variants:
            is_winner = variant.id == winner_variant.id
            variant.status = "winner" if is_winner else "loser"
            variant.traffic_weight = Decimal("1") if is_winner else Decimal("0")
            for mapping in self.mappings:
                if mapping.experiment_id == experiment.id and mapping.experiment_variant_id == variant.id:
                    mapping.traffic_weight = Decimal("1") if is_winner else Decimal("0")
                    mapping.is_active = is_winner
                    mapping.is_winner = is_winner


class FakeExperimentResultRepository:
    def __init__(self) -> None:
        self.results: dict[int, VariantPerformance] = {}
        self.project_ids: list[str] = []
        self.calls: list[tuple[int, datetime, datetime]] = []

    def fetch_variant_results(
        self,
        *,
        project_id: str,
        experiment,
        variants,
        window_start: datetime,
        window_end: datetime,
    ) -> dict[int, VariantPerformance]:
        self.project_ids.append(project_id)
        self.calls.append((experiment.id, window_start, window_end))
        return {
            variant.id: self.results.get(
                variant.id,
                VariantPerformance(
                    experiment_variant_id=variant.id,
                    ad_impression_count=0,
                    ad_click_count=0,
                    attributed_purchase_count=0,
                ),
            )
            for variant in variants
        }
