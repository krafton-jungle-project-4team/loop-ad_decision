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
    ) -> None:
        self.result = result
        self.existing_experiment = existing_experiment
        self.existing_mapping = existing_mapping
        self.mappings = mappings or []
        self.experiments = experiments or []
        self.created_experiments: list[SimpleNamespace] = []
        self.created_mappings: list[SimpleNamespace] = []
        self.updated_results: list[tuple[int, dict[str, object]]] = []
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

    def get_experiment_by_recommendation_action(
        self,
        *,
        recommendation_result_id: int,
        action_id: str,
    ) -> SimpleNamespace | None:
        return self.existing_experiment

    def create_experiment(self, **values: object) -> SimpleNamespace:
        experiment = SimpleNamespace(id=100 + len(self.created_experiments), **values)
        self.created_experiments.append(experiment)
        return experiment

    def get_segment_ad_mapping_by_recommendation_action(
        self,
        *,
        recommendation_result_id: int,
        action_id: str,
    ) -> SimpleNamespace | None:
        return self.existing_mapping

    def create_segment_ad_mapping(self, **values: object) -> SimpleNamespace:
        mapping = SimpleNamespace(id=200 + len(self.created_mappings), **values)
        self.created_mappings.append(mapping)
        return mapping

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

    def list_active_segment_ad_mappings(self, project_id: str) -> list[SimpleNamespace]:
        return [
            mapping
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
    status: str = "pending_review",
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


def test_approve_creates_manual_experiment_and_mapping() -> None:
    repository = FakeRecommendationRepository(recommendation_result())

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
    repository = FakeRecommendationRepository(recommendation_result(status="no_action"))

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
    repository = FakeRecommendationRepository(recommendation_result(actions=[manual_action]))

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
                status="active",
            ),
            SimpleNamespace(
                id=11,
                project_id="loopad-demo-shop",
                recommendation_result_id=1,
                status="inactive",
            ),
        ],
        experiments=[
            SimpleNamespace(
                id=20,
                project_id="loopad-demo-shop",
                recommendation_result_id=1,
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

    assert response.status == "rejected"
    assert response.inactivated_segment_ad_mapping_ids == [10]
    assert response.stopped_experiment_ids == [20]
    assert repository.mappings[0].status == "inactive"
    assert repository.experiments[0].status == "stopped"
    assert repository.result.status == "rejected"


def test_active_ad_mapping_response_only_includes_active_project_mappings() -> None:
    repository = FakeRecommendationRepository(
        mappings=[
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
                source="manual_approval",
                status="active",
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
