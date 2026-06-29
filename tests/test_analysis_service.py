from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from app.actions.schemas import (
    ActionExperiment,
    ActionRecommendationRequest,
    ActionRecommendationResponse,
    RecommendedAction,
)
from app.analysis.schemas import FunnelRecommendationAnalysisRequest
from app.analysis.service import run_funnel_recommendation_analysis
from app.anomalies.schemas import (
    FunnelAnomalyEvaluation,
    FunnelAnomalyResponse,
)
from app.metrics.schemas import FunnelMetrics
from app.root_causes.schemas import RootCauseAnalysisResponse, RootCauseCandidate

KST = ZoneInfo("Asia/Seoul")
WINDOW_START = datetime(2026, 6, 24, 17, 0, tzinfo=KST)
WINDOW_END = datetime(2026, 6, 24, 18, 0, tzinfo=KST)
BASELINE_START = datetime(2026, 6, 24, 16, 0, tzinfo=KST)
BASELINE_END = datetime(2026, 6, 24, 17, 0, tzinfo=KST)


class FakePersistenceRepository:
    def __init__(
        self,
        policy: SimpleNamespace | None = None,
        *,
        existing_experiment: SimpleNamespace | None = None,
        existing_mapping: SimpleNamespace | None = None,
        active_creative: SimpleNamespace | None = None,
    ) -> None:
        self.policy = policy
        self.existing_experiment = existing_experiment
        self.existing_mapping = existing_mapping
        self.active_creative = active_creative
        self.recommendation_results: list[SimpleNamespace] = []
        self.recommendation_actions: list[SimpleNamespace] = []
        self.experiments: list[SimpleNamespace] = []
        self.mappings: list[SimpleNamespace] = []
        self.committed = False
        self.rolled_back = False

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True

    def get_automation_policy(self, project_id: str) -> SimpleNamespace | None:
        return self.policy

    def create_recommendation_result(self, **values: object) -> SimpleNamespace:
        result = SimpleNamespace(id=len(self.recommendation_results) + 1, **values)
        self.recommendation_results.append(result)
        return result

    def update_recommendation_result(
        self,
        recommendation_result_id: int,
        values: dict[str, object],
    ) -> SimpleNamespace | None:
        for result in self.recommendation_results:
            if result.id == recommendation_result_id:
                for key, value in values.items():
                    setattr(result, key, value)
                return result
        return None

    def create_recommendation_action(self, **values: object) -> SimpleNamespace:
        action = SimpleNamespace(
            id=300 + len(self.recommendation_actions),
            bandit_policy_id=None,
            bandit_arm_id=None,
            **values,
        )
        self.recommendation_actions.append(action)
        return action

    def update_recommendation_action(
        self,
        recommendation_action_id: int,
        values: dict[str, object],
    ) -> SimpleNamespace | None:
        for action in self.recommendation_actions:
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
        experiment = SimpleNamespace(id=100 + len(self.experiments), **values)
        self.experiments.append(experiment)
        return experiment

    def get_segment_ad_mapping_by_recommendation_action(
        self,
        *,
        recommendation_action_id: int,
    ) -> SimpleNamespace | None:
        return self.existing_mapping

    def create_segment_ad_mapping(self, **values: object) -> SimpleNamespace:
        mapping = SimpleNamespace(id=200 + len(self.mappings), **values)
        self.mappings.append(mapping)
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


def analysis_request() -> FunnelRecommendationAnalysisRequest:
    return FunnelRecommendationAnalysisRequest(
        project_id="loopad-demo-shop",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        top_n=5,
    )


def metrics() -> FunnelMetrics:
    return FunnelMetrics(
        product_view_sessions=1000,
        add_to_cart_sessions=100,
        checkout_start_sessions=50,
        purchase_sessions=25,
        view_to_cart_rate=0.1,
        cart_to_checkout_rate=0.5,
        checkout_to_purchase_rate=0.5,
        view_to_purchase_rate=0.025,
        view_to_cart_dropoff_rate=0.9,
        cart_to_checkout_dropoff_rate=0.5,
        checkout_to_purchase_dropoff_rate=0.5,
    )


def anomaly_evaluation() -> FunnelAnomalyEvaluation:
    return FunnelAnomalyEvaluation(
        metric="view_to_cart_rate",
        funnel_step="product_view_to_add_to_cart",
        severity="critical",
        current_value=0.1,
        baseline_value=0.3,
        delta_point=-0.2,
        relative_change=-0.67,
        drop_point=0.2,
        relative_drop=0.67,
        current_denominator=1000,
        baseline_denominator=1000,
        min_sample_size=100,
        message="view_to_cart_rate dropped.",
    )


def anomaly_response(status: str = "critical") -> FunnelAnomalyResponse:
    anomaly = anomaly_evaluation() if status != "normal" else None
    return FunnelAnomalyResponse(
        project_id="loopad-demo-shop",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        segment={"channel": "kakao", "campaign_id": None},
        status=status,
        current_metrics=metrics(),
        baseline_metrics=metrics(),
        evaluations=[anomaly_evaluation()],
        anomalies=[anomaly_evaluation()] if anomaly is not None else [],
        volume_evaluations=[],
        volume_anomalies=[],
        primary_anomaly=anomaly,
        summary_message="Funnel anomaly detected." if anomaly is not None else "No anomaly.",
    )


def root_cause_candidate() -> RootCauseCandidate:
    return RootCauseCandidate(
        rank=1,
        cause_type="channel_specific_drop",
        dimension="channel",
        value="kakao",
        metric="view_to_purchase_rate",
        funnel_step="product_view_to_purchase",
        severity="critical",
        current_value=0.1,
        baseline_value=0.3,
        drop_point=0.2,
        relative_drop=0.67,
        current_denominator=100,
        baseline_denominator=120,
        support_share=0.4,
        excess_lost_sessions=20.0,
        score=2.4,
        message="channel=kakao conversion dropped.",
    )


def root_cause_response(candidates: list[RootCauseCandidate]) -> RootCauseAnalysisResponse:
    return RootCauseAnalysisResponse(
        project_id="loopad-demo-shop",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        baseline_start=BASELINE_START,
        baseline_end=BASELINE_END,
        segment={"channel": "kakao", "campaign_id": None},
        status="critical",
        target_anomaly=None,
        total_candidates_evaluated=len(candidates),
        candidates=candidates,
        summary_message="Root causes found." if candidates else "No root causes found.",
    )


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


def manual_review_action() -> RecommendedAction:
    return recommended_action(action_id="manual_review", action_type="REVIEW").model_copy(
        update={
            "title": "운영자 수동 검토 필요",
            "description": "원인 후보가 없어 수동 검토가 필요합니다.",
            "priority_score": 0.5,
            "experiment": None,
        }
    )


def action_response(action: RecommendedAction) -> ActionRecommendationResponse:
    return ActionRecommendationResponse(
        project_id="loopad-demo-shop",
        window_start=WINDOW_START,
        window_end=WINDOW_END,
        segment={"channel": "kakao", "campaign_id": None},
        recommendations=[action],
    )


def automation_policy(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "id": 1,
        "enabled": True,
        "auto_execute_enabled": True,
        "allowed_action_ids": [],
        "allowed_action_types": ["PRODUCT"],
        "blocked_action_ids": [],
        "max_experiment_traffic_ratio": 0.2,
        "min_priority_score": 0.5,
        "max_discount_rate": None,
        "max_daily_coupon_budget": None,
        "max_message_per_user_per_day": None,
        "stop_loss_relative_drop": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_no_anomaly_saves_no_action_result() -> None:
    repository = FakePersistenceRepository()

    response = run_funnel_recommendation_analysis(
        request=analysis_request(),
        metrics_repository=object(),
        root_cause_repository=object(),
        persistence_repository=repository,
        anomaly_calculator=lambda request, repo: anomaly_response("normal"),
        root_cause_calculator=lambda request, repo: (_ for _ in ()).throw(AssertionError()),
        action_recommender=lambda request: (_ for _ in ()).throw(AssertionError()),
    )

    assert response.status == "no_action"
    assert response.recommendation_result_id == 1
    assert repository.recommendation_results[0].status == "no_action"
    assert repository.recommendation_results[0].policy_decision_json["actions"] == []
    assert repository.committed is True


def test_anomaly_without_root_cause_uses_manual_review_and_pending_review() -> None:
    repository = FakePersistenceRepository(policy=automation_policy(allowed_action_types=["REVIEW"]))
    captured_action_request: list[ActionRecommendationRequest] = []

    def action_recommender(request: ActionRecommendationRequest) -> ActionRecommendationResponse:
        captured_action_request.append(request)
        return action_response(manual_review_action())

    response = run_funnel_recommendation_analysis(
        request=analysis_request(),
        metrics_repository=object(),
        root_cause_repository=object(),
        persistence_repository=repository,
        anomaly_calculator=lambda request, repo: anomaly_response(),
        root_cause_calculator=lambda request, repo: root_cause_response([]),
        action_recommender=action_recommender,
    )

    assert response.status == "pending_actions"
    assert response.recommended_actions[0].action_id == "manual_review"
    assert repository.recommendation_actions[0].action_id == "manual_review"
    assert repository.recommendation_actions[0].status == "pending_review"
    assert captured_action_request[0].causes[0].cause_type == "UNEXPLAINED_FUNNEL_ANOMALY"
    assert repository.recommendation_results[0].root_causes_json["synthetic_causes"][0][
        "cause_type"
    ] == "UNEXPLAINED_FUNNEL_ANOMALY"


def test_policy_blocked_when_no_recommendations_auto_execute() -> None:
    repository = FakePersistenceRepository(
        policy=automation_policy(allowed_action_ids=["pause_out_of_stock_ads"], allowed_action_types=[])
    )

    response = run_funnel_recommendation_analysis(
        request=analysis_request(),
        metrics_repository=object(),
        root_cause_repository=object(),
        persistence_repository=repository,
        anomaly_calculator=lambda request, repo: anomaly_response(),
        root_cause_calculator=lambda request, repo: root_cause_response([root_cause_candidate()]),
        action_recommender=lambda request: action_response(recommended_action()),
    )

    assert response.status == "pending_actions"
    assert response.blocked_actions[0].action_id == "recommend_alternative_product"
    assert repository.recommendation_actions[0].status == "policy_blocked"
    assert repository.recommendation_results[0].policy_decision_json["actions"][0][
        "status"
    ] == "blocked"
    assert repository.experiments == []
    assert repository.mappings == []


def test_auto_executed_action_creates_experiment_and_mapping() -> None:
    repository = FakePersistenceRepository(
        policy=automation_policy(),
        active_creative=active_creative(),
    )

    response = run_funnel_recommendation_analysis(
        request=analysis_request(),
        metrics_repository=object(),
        root_cause_repository=object(),
        persistence_repository=repository,
        anomaly_calculator=lambda request, repo: anomaly_response(),
        root_cause_calculator=lambda request, repo: root_cause_response([root_cause_candidate()]),
        action_recommender=lambda request: action_response(recommended_action()),
    )

    assert response.status == "experiment_running"
    assert response.auto_executed_action_ids == ["recommend_alternative_product"]
    assert response.created_experiment_ids == [100]
    assert response.created_segment_ad_mapping_ids == [200]
    assert repository.recommendation_results[0].status == "experiment_running"
    assert repository.recommendation_results[0].policy_decision_json["actions"][0][
        "status"
    ] == "auto_executed"
    assert repository.experiments[0].traffic_split_json == {"control": 0.8, "treatment": 0.2}
    assert repository.experiments[0].recommendation_action_id == 300
    assert repository.mappings[0].source == "auto_policy"
    assert repository.mappings[0].status == "active"
    assert repository.mappings[0].recommendation_action_id == 300
    assert repository.mappings[0].creative_id == 70
    assert repository.mappings[0].campaign_id == 60
    assert repository.mappings[0].coupon_id == 80


def test_existing_experiment_and_mapping_are_reused_by_recommendation_action() -> None:
    repository = FakePersistenceRepository(
        policy=automation_policy(),
        existing_experiment=SimpleNamespace(id=777),
        existing_mapping=SimpleNamespace(id=888),
    )

    response = run_funnel_recommendation_analysis(
        request=analysis_request(),
        metrics_repository=object(),
        root_cause_repository=object(),
        persistence_repository=repository,
        anomaly_calculator=lambda request, repo: anomaly_response(),
        root_cause_calculator=lambda request, repo: root_cause_response([root_cause_candidate()]),
        action_recommender=lambda request: action_response(recommended_action()),
    )

    assert response.status == "experiment_running"
    assert response.created_experiment_ids == [777]
    assert response.created_segment_ad_mapping_ids == [888]
    assert repository.experiments == []
    assert repository.mappings == []
