from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from fastapi import HTTPException

from app.actions.schemas import ActionExperiment, RecommendedAction
from app.recommendations.schemas import (
    RecommendationApproveRequest,
    RecommendationRejectRequest,
)
from app.recommendations.service import (
    approve_recommendation_result,
    get_recommendation_result_response,
    list_active_segment_ad_mappings,
    reject_recommendation_result,
)

KST = ZoneInfo("Asia/Seoul")
WINDOW_START = datetime(2026, 6, 24, 17, 0, tzinfo=KST)
WINDOW_END = datetime(2026, 6, 24, 18, 0, tzinfo=KST)


class FakeRecommendationRepository:
    def __init__(
        self,
        result: SimpleNamespace | None = None,
        *,
        existing_experiment: SimpleNamespace | None = None,
        existing_mapping: SimpleNamespace | None = None,
        mappings: list[SimpleNamespace] | None = None,
        experiments: list[SimpleNamespace] | None = None,
        actions: list[SimpleNamespace] | None = None,
        active_creative: SimpleNamespace | None = None,
    ) -> None:
        self.result = result
        self.existing_experiment = existing_experiment
        self.existing_mapping = existing_mapping
        self.active_creative = active_creative
        self.mappings = mappings or []
        self.experiments = experiments or []
        self.actions = actions if actions is not None else (
            [recommendation_action(result=result)] if result is not None else []
        )
        self.created_experiments: list[SimpleNamespace] = []
        self.created_mappings: list[SimpleNamespace] = []
        self.updated_results: list[tuple[int, dict[str, object]]] = []
        self.updated_actions: list[tuple[int, dict[str, object]]] = []
        self.updated_mappings: list[tuple[int, dict[str, object]]] = []
        self.updated_experiments: list[tuple[int, dict[str, object]]] = []
        self.committed = False
        self.rolled_back = False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def get_recommendation_result(self, recommendation_result_id: int) -> SimpleNamespace | None:
        if self.result is None or self.result.id != recommendation_result_id:
            return None
        return self.result

    def update_recommendation_result(
        self,
        recommendation_result_id: int,
        values: dict[str, object],
    ) -> SimpleNamespace | None:
        self.updated_results.append((recommendation_result_id, values))
        if self.result is not None:
            for key, value in values.items():
                setattr(self.result, key, value)
        return self.result

    def list_recommendation_actions(
        self,
        *,
        recommendation_result_id: int,
    ) -> list[SimpleNamespace]:
        return [
            action
            for action in self.actions
            if action.recommendation_result_id == recommendation_result_id
        ]

    def update_recommendation_action(
        self,
        recommendation_action_id: int,
        values: dict[str, object],
    ) -> SimpleNamespace | None:
        self.updated_actions.append((recommendation_action_id, values))
        for action in self.actions:
            if action.id == recommendation_action_id:
                for key, value in values.items():
                    setattr(action, key, value)
                return action
        return None

    def get_experiment_by_recommendation_action(
        self,
        *,
        recommendation_action_id: int,
    ) -> SimpleNamespace | None:
        return self.existing_experiment

    def create_experiment(self, **values: object) -> SimpleNamespace:
        experiment = SimpleNamespace(id=100 + len(self.created_experiments), **values)
        self.created_experiments.append(experiment)
        return experiment

    def get_segment_ad_mapping_by_recommendation_action(
        self,
        *,
        recommendation_action_id: int,
    ) -> SimpleNamespace | None:
        return self.existing_mapping

    def create_segment_ad_mapping(self, **values: object) -> SimpleNamespace:
        mapping = SimpleNamespace(id=200 + len(self.created_mappings), **values)
        self.created_mappings.append(mapping)
        return mapping

    def get_ad_creative(self, creative_id: int) -> SimpleNamespace | None:
        if self.active_creative is not None and self.active_creative.id == creative_id:
            return self.active_creative
        return None

    def get_active_ad_creative_by_action(
        self,
        *,
        project_id: str,
        action_id: str,
    ) -> SimpleNamespace | None:
        creative = self.active_creative
        if (
            creative is not None
            and creative.project_id == project_id
            and creative.action_id == action_id
            and creative.status == "active"
            and creative.image_url
        ):
            return creative
        return None

    def list_segment_ad_mappings(
        self,
        *,
        project_id: str | None = None,
        recommendation_result_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SimpleNamespace]:
        values = self.mappings
        if project_id is not None:
            values = [mapping for mapping in values if mapping.project_id == project_id]
        if recommendation_result_id is not None:
            values = [
                mapping
                for mapping in values
                if mapping.recommendation_result_id == recommendation_result_id
            ]
        if status is not None:
            values = [mapping for mapping in values if mapping.status == status]
        return values[:limit]

    def list_active_segment_ad_mappings_with_creatives(self, project_id: str) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(mapping=mapping, creative=getattr(mapping, "creative", None))
            for mapping in self.mappings
            if mapping.project_id == project_id and mapping.status == "active"
        ]

    def update_segment_ad_mapping(
        self,
        mapping_id: int,
        values: dict[str, object],
    ) -> SimpleNamespace | None:
        self.updated_mappings.append((mapping_id, values))
        for mapping in self.mappings:
            if mapping.id == mapping_id:
                for key, value in values.items():
                    setattr(mapping, key, value)
                return mapping
        return None

    def list_experiments(
        self,
        *,
        project_id: str | None = None,
        recommendation_result_id: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[SimpleNamespace]:
        values = self.experiments
        if project_id is not None:
            values = [experiment for experiment in values if experiment.project_id == project_id]
        if recommendation_result_id is not None:
            values = [
                experiment
                for experiment in values
                if experiment.recommendation_result_id == recommendation_result_id
            ]
        if status is not None:
            values = [experiment for experiment in values if experiment.status == status]
        return values[:limit]

    def update_experiment(
        self,
        experiment_id: int,
        values: dict[str, object],
    ) -> SimpleNamespace | None:
        self.updated_experiments.append((experiment_id, values))
        for experiment in self.experiments:
            if experiment.id == experiment_id:
                for key, value in values.items():
                    setattr(experiment, key, value)
                return experiment
        return None


def recommended_action(
    *,
    action_id: str = "recommend_alternative_product",
    action_type: str = "PRODUCT",
) -> RecommendedAction:
    return RecommendedAction(
        action_id=action_id,
        action_type=action_type,
        title="대체 상품 추천",
        description="품절 상품 대신 대체 상품을 추천합니다.",
        target_step=None,
        priority_score=0.8,
        expected_impact="전환율 개선",
        rationale="테스트 추천",
        triggered_by=["cause-1"],
        execution_hint={"slot": "product_detail"},
        experiment=ActionExperiment(
            primary_metric="view_to_purchase_rate",
            guardrail_metrics=["purchase_rate"],
            variants=["control", "treatment"],
        ),
    )


def recommendation_result(
    *,
    status: str = "pending_actions",
    actions: list[RecommendedAction] | None = None,
) -> SimpleNamespace:
    recommendations = actions or [recommended_action()]
    return SimpleNamespace(
        id=1,
        project_id="loopad-demo-shop",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        baseline_start=None,
        baseline_end=None,
        segment_json={"channel": "kakao"},
        segment_hash="hash",
        status=status,
        anomaly_json={},
        root_causes_json={},
        recommendations_json={
            "recommendations": [
                recommendation.model_dump(mode="json")
                for recommendation in recommendations
            ]
        },
        policy_decision_json={"policy_id": 1, "auto_execute_enabled": True, "actions": []},
        created_at=WINDOW_START,
        updated_at=WINDOW_START,
    )


def recommendation_action(
    *,
    result: SimpleNamespace | None = None,
    action: RecommendedAction | None = None,
    status: str = "pending_review",
) -> SimpleNamespace:
    result = result or recommendation_result()
    action = action or recommended_action()
    return SimpleNamespace(
        id=10,
        project_id=result.project_id,
        recommendation_result_id=result.id,
        action_id=action.action_id,
        action_type=action.action_type,
        title=action.title,
        description=action.description,
        target_step=action.target_step,
        priority_score=action.priority_score,
        expected_impact=action.expected_impact,
        rationale=action.rationale,
        triggered_by_json=action.triggered_by,
        execution_hint_json=action.execution_hint,
        experiment_json=action.experiment.model_dump(mode="json") if action.experiment else {},
        policy_status=None,
        policy_reasons_json=[],
        policy_decision_json={},
        selected_by_strategy="policy",
        bandit_policy_id=None,
        bandit_arm_id=None,
        sampled_value=None,
        status=status,
        auto_executed_at=None,
        reviewed_by=None,
        reviewed_at=None,
        review_reason=None,
        created_at=WINDOW_START,
        updated_at=WINDOW_START,
    )


def active_creative(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": 70,
        "project_id": "loopad-demo-shop",
        "action_id": "recommend_alternative_product",
        "campaign_id": 60,
        "coupon_id": 80,
        "status": "active",
        "image_url": "https://cdn.example/ad.png",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_approve_creates_manual_experiment_and_mapping() -> None:
    repository = FakeRecommendationRepository(
        recommendation_result(),
        active_creative=active_creative(),
    )

    response = approve_recommendation_result(
        repository=repository,
        recommendation_result_id=1,
        request=RecommendationApproveRequest(
            approved_by="dashboard-user",
            action_ids=["recommend_alternative_product"],
            reason="데모 승인",
        ),
    )

    assert response.status == "experiment_running"
    assert response.approved_action_ids == ["recommend_alternative_product"]
    assert response.created_experiment_ids == [100]
    assert response.created_segment_ad_mapping_ids == [200]
    assert repository.created_experiments[0].status == "running"
    assert repository.created_mappings[0].source == "manual_approval"
    assert repository.created_mappings[0].creative_id == 70
    assert repository.created_mappings[0].campaign_id == 60
    assert repository.created_mappings[0].coupon_id == 80
    assert repository.result.status == "experiment_running"
    assert repository.committed is True


def test_approve_rejects_unknown_action_id() -> None:
    repository = FakeRecommendationRepository(recommendation_result())

    with pytest.raises(HTTPException) as exc_info:
        approve_recommendation_result(
            repository=repository,
            recommendation_result_id=1,
            request=RecommendationApproveRequest(
                approved_by="dashboard-user",
                action_ids=["unknown_action"],
            ),
        )

    assert exc_info.value.status_code == 400
    assert repository.rolled_back is True


def test_approve_rejects_invalid_status_transition() -> None:
    result = recommendation_result()
    repository = FakeRecommendationRepository(
        result,
        actions=[recommendation_action(result=result, status="stopped")],
    )

    with pytest.raises(HTTPException) as exc_info:
        approve_recommendation_result(
            repository=repository,
            recommendation_result_id=1,
            request=RecommendationApproveRequest(
                approved_by="dashboard-user",
                action_ids=["recommend_alternative_product"],
            ),
        )

    assert exc_info.value.status_code == 409


def test_approve_rejects_manual_review_action() -> None:
    manual_action = recommended_action(action_id="manual_review", action_type="REVIEW").model_copy(
        update={"experiment": None}
    )
    result = recommendation_result(actions=[manual_action])
    repository = FakeRecommendationRepository(
        result,
        actions=[recommendation_action(result=result, action=manual_action)],
    )

    with pytest.raises(HTTPException) as exc_info:
        approve_recommendation_result(
            repository=repository,
            recommendation_result_id=1,
            request=RecommendationApproveRequest(
                approved_by="dashboard-user",
                action_ids=["manual_review"],
            ),
        )

    assert exc_info.value.status_code == 400


def test_approve_reuses_existing_experiment_and_mapping() -> None:
    repository = FakeRecommendationRepository(
        recommendation_result(),
        existing_experiment=SimpleNamespace(id=777),
        existing_mapping=SimpleNamespace(id=888),
    )

    response = approve_recommendation_result(
        repository=repository,
        recommendation_result_id=1,
        request=RecommendationApproveRequest(
            approved_by="dashboard-user",
            action_ids=["recommend_alternative_product", "recommend_alternative_product"],
        ),
    )

    assert response.created_experiment_ids == [777]
    assert response.created_segment_ad_mapping_ids == [888]
    assert repository.created_experiments == []
    assert repository.created_mappings == []


def test_reject_marks_result_mappings_and_experiments() -> None:
    repository = FakeRecommendationRepository(
        recommendation_result(status="experiment_running"),
        mappings=[
            SimpleNamespace(
                id=10,
                project_id="loopad-demo-shop",
                recommendation_result_id=1,
                recommendation_action_id=10,
                status="active",
            ),
            SimpleNamespace(
                id=11,
                project_id="loopad-demo-shop",
                recommendation_result_id=1,
                recommendation_action_id=10,
                status="inactive",
            ),
        ],
        experiments=[
            SimpleNamespace(
                id=20,
                project_id="loopad-demo-shop",
                recommendation_result_id=1,
                recommendation_action_id=10,
                status="running",
            )
        ],
    )

    response = reject_recommendation_result(
        repository=repository,
        recommendation_result_id=1,
        request=RecommendationRejectRequest(
            rejected_by="dashboard-user",
            reason="쿠폰 비용 위험",
        ),
    )

    assert response.status == "dismissed"
    assert response.rejected_action_ids == ["recommend_alternative_product"]
    assert response.inactivated_segment_ad_mapping_ids == [10]
    assert response.stopped_experiment_ids == [20]
    assert repository.mappings[0].status == "inactive"
    assert repository.experiments[0].status == "stopped"
    assert repository.result.status == "dismissed"


def test_legacy_result_without_action_rows_cannot_be_approved_or_rejected() -> None:
    repository = FakeRecommendationRepository(recommendation_result(), actions=[])

    with pytest.raises(HTTPException) as approve_exc:
        approve_recommendation_result(
            repository=repository,
            recommendation_result_id=1,
            request=RecommendationApproveRequest(
                approved_by="dashboard-user",
                action_ids=["recommend_alternative_product"],
            ),
        )
    with pytest.raises(HTTPException) as reject_exc:
        reject_recommendation_result(
            repository=repository,
            recommendation_result_id=1,
            request=RecommendationRejectRequest(rejected_by="dashboard-user"),
        )

    assert approve_exc.value.status_code == 409
    assert reject_exc.value.status_code == 409
    assert repository.actions == []


def test_recommendation_detail_returns_actions_or_empty_legacy_actions() -> None:
    result = recommendation_result()
    repository = FakeRecommendationRepository(result)

    response = get_recommendation_result_response(
        repository=repository,
        recommendation_result_id=1,
    )

    assert len(response.actions) == 1
    legacy_repository = FakeRecommendationRepository(result, actions=[])
    legacy_response = get_recommendation_result_response(
        repository=legacy_repository,
        recommendation_result_id=1,
    )
    assert legacy_response.actions == []


def test_active_ad_mapping_response_only_includes_active_project_mappings() -> None:
    repository = FakeRecommendationRepository(
        mappings=[
            SimpleNamespace(
                id=99,
                project_id="loopad-demo-shop",
                segment_json={"channel": "kakao"},
                segment_hash="hash-empty-content",
                action_id="recommend_alternative_product",
                action_type="PRODUCT",
                execution_hint_json={"slot": "product_detail"},
                experiment_id=99,
                recommendation_result_id=1,
                recommendation_action_id=99,
                bandit_policy_id=None,
                bandit_arm_id=None,
                bandit_decision_id=None,
                campaign_id=None,
                creative_id=None,
                coupon_id=None,
                source="manual_approval",
                status="active",
                creative=None,
            ),
            SimpleNamespace(
                id=1,
                project_id="loopad-demo-shop",
                segment_json={"channel": "kakao"},
                segment_hash="hash-1",
                action_id="recommend_alternative_product",
                action_type="PRODUCT",
                execution_hint_json={"slot": "product_detail"},
                experiment_id=10,
                recommendation_result_id=1,
                recommendation_action_id=10,
                bandit_policy_id=30,
                bandit_arm_id=40,
                bandit_decision_id=50,
                campaign_id=60,
                creative_id=70,
                coupon_id=80,
                source="manual_approval",
                status="active",
                creative=SimpleNamespace(
                    image_url="https://cdn.example/ad.png",
                    title="대체 상품",
                    message="지금 확인해보세요",
                    landing_url="https://shop.example/products/1",
                ),
            ),
            SimpleNamespace(
                id=2,
                project_id="loopad-demo-shop",
                segment_json={},
                segment_hash="hash-2",
                action_id="limited_time_coupon",
                action_type="INCENTIVE",
                execution_hint_json={},
                experiment_id=11,
                recommendation_result_id=2,
                recommendation_action_id=11,
                bandit_policy_id=None,
                bandit_arm_id=None,
                bandit_decision_id=None,
                campaign_id=None,
                creative_id=None,
                coupon_id=None,
                source="manual_approval",
                status="inactive",
            ),
        ]
    )

    response = list_active_segment_ad_mappings(
        repository=repository,
        project_id="loopad-demo-shop",
    )

    assert len(response) == 1
    assert response[0].mapping_id == 1
    assert response[0].status == "active"
    assert response[0].recommendation_action_id == 10
    assert response[0].bandit_policy_id == 30
    assert response[0].content_url == "https://cdn.example/ad.png"
    assert response[0].title == "대체 상품"
    assert response[0].serving_weight is None
