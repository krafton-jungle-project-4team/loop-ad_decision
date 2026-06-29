import pytest
from sqlalchemy.dialects import postgresql
from pydantic import ValidationError

from app.core.config import Settings
from app.db.postgres import build_postgres_url, create_postgres_engine
from app.persistence.models import (
    AdCreative,
    AutomationPolicy,
    Experiment,
    RecommendationAction,
    RecommendationResult,
    SegmentAdMapping,
)
from app.persistence.repository import (
    PostgresRepository,
    build_segment_hash,
    canonical_segment_json,
)


class CapturingSession:
    def __init__(self) -> None:
        self.statement = None

    def execute(self, statement):
        self.statement = statement
        return []


def settings_with_required_env(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "loopad_env": "local",
        "loopad_service_id": "decision-api",
        "port": 8000,
        "loopad_aurora_host": "postgres",
        "loopad_aurora_port": 5432,
        "loopad_aurora_database": "loopad",
        "loopad_aurora_username": "postgres",
        "loopad_aurora_password": "secret",
        "loopad_clickhouse_url": "http://clickhouse:8123",
        "loopad_clickhouse_username": "default",
        "loopad_data_storage_bucket": "bucket",
        "loopad_genai_generated_assets_prefix": "genai/generated/",
        "loopad_openai_api_key": "sk-test",
    }
    values.update(overrides)
    return Settings(**values)


def test_canonical_segment_json_removes_null_values_and_sorts_keys() -> None:
    canonical = canonical_segment_json(
        {
            "product_id": "sku-1",
            "channel": None,
            "age_group": "30s",
        }
    )

    assert canonical == '{"age_group":"30s","product_id":"sku-1"}'


def test_build_segment_hash_uses_canonical_segment_json() -> None:
    first_hash = build_segment_hash(
        {
            "product_id": "sku-1",
            "channel": None,
            "age_group": "30s",
        }
    )
    second_hash = build_segment_hash(
        {
            "age_group": "30s",
            "product_id": "sku-1",
        }
    )

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_settings_requires_loopad_env_contract() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_settings_validates_decision_service_id() -> None:
    with pytest.raises(ValidationError):
        settings_with_required_env(loopad_service_id="dashboard-api")


def test_build_postgres_url_uses_aurora_settings() -> None:
    settings = settings_with_required_env(
        loopad_aurora_host="postgres",
        loopad_aurora_port=5433,
        loopad_aurora_username="loopad",
        loopad_aurora_password="secret",
        loopad_aurora_database="decision",
    )

    assert build_postgres_url(settings) == "postgresql+psycopg://loopad:***@postgres:5433/decision"


def test_postgres_engine_is_created_lazily() -> None:
    create_postgres_engine.cache_clear()

    assert create_postgres_engine.cache_info().currsize == 0


def test_orm_models_include_contract_columns_without_content_url_column() -> None:
    assert "bandit_decision_summary_json" in RecommendationResult.__table__.columns
    assert "summary_message" in RecommendationResult.__table__.columns

    recommendation_action_columns = RecommendationAction.__table__.columns
    for column_name in (
        "recommendation_result_id",
        "policy_status",
        "policy_reasons_json",
        "policy_decision_json",
        "status",
        "bandit_policy_id",
        "bandit_arm_id",
    ):
        assert column_name in recommendation_action_columns

    experiment_columns = Experiment.__table__.columns
    assert "recommendation_action_id" in experiment_columns
    assert "bandit_policy_id" in experiment_columns
    assert "bandit_arm_id" in experiment_columns
    assert any(
        constraint.name == "uq_experiments_recommendation_action"
        for constraint in Experiment.__table__.constraints
    )

    mapping_columns = SegmentAdMapping.__table__.columns
    for column_name in (
        "recommendation_action_id",
        "bandit_policy_id",
        "bandit_arm_id",
        "bandit_decision_id",
        "campaign_id",
        "creative_id",
        "coupon_id",
    ):
        assert column_name in mapping_columns
    assert any(
        constraint.name == "uq_segment_ad_mappings_recommendation_action"
        for constraint in SegmentAdMapping.__table__.constraints
    )

    assert "image_url" in AdCreative.__table__.columns
    assert "content_url" not in AdCreative.__table__.columns


def test_automation_policy_matches_contract_columns() -> None:
    columns = AutomationPolicy.__table__.columns
    for column_name in (
        "project_id",
        "enabled",
        "auto_execute_enabled",
        "allowed_action_ids",
        "allowed_action_types",
        "blocked_action_ids",
        "max_experiment_traffic_ratio",
        "min_priority_score",
        "max_discount_rate",
        "max_daily_coupon_budget",
        "max_message_per_user_per_day",
        "stop_loss_relative_drop",
    ):
        assert column_name in columns
    assert columns["project_id"].unique is True


def test_active_mapping_query_filters_expired_mappings_and_inactive_creatives() -> None:
    session = CapturingSession()
    repository = PostgresRepository(session)

    assert repository.list_active_segment_ad_mappings_with_creatives("loopad-demo-shop") == []
    compiled_sql = str(
        session.statement.compile(
            dialect=postgresql.dialect(),
            compile_kwargs={"literal_binds": True},
        )
    )

    assert "segment_ad_mappings.status = 'active'" in compiled_sql
    assert "segment_ad_mappings.expires_at IS NULL" in compiled_sql
    assert "segment_ad_mappings.expires_at >" in compiled_sql
    assert "segment_ad_mappings.creative_id IS NULL" in compiled_sql
    assert "ad_creatives.status = 'active'" in compiled_sql
    assert "ad_creatives.project_id = segment_ad_mappings.project_id" in compiled_sql
