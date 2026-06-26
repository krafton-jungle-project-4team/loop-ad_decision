from pathlib import Path

from app.persistence.models import AnalysisJob


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_clickhouse_init_uses_new_events_schema_without_seed_data() -> None:
    sql = read_text("scripts/clickhouse/clickhouse-init.sql")

    assert "CREATE TABLE IF NOT EXISTS events" in sql
    assert "experiment_id String DEFAULT ''" in sql
    assert "variant_id LowCardinality(String) DEFAULT ''" in sql
    assert "action_id String DEFAULT ''" in sql
    assert "mapping_id String DEFAULT ''" in sql
    assert "ad_id String DEFAULT ''" in sql
    assert "creative_id String DEFAULT ''" in sql
    assert "properties_json String DEFAULT ''" in sql
    assert "CREATE MATERIALIZED VIEW IF NOT EXISTS mv_experiment_metrics_daily" in sql
    assert "INSERT INTO" not in sql


def test_clickhouse_seed_loads_ga4_csv_into_new_events_schema() -> None:
    sql = read_text("scripts/clickhouse/clickhouse-seed.sql")

    assert "INSERT INTO events" in sql
    assert "FROM file(" in sql
    assert "'ga4_exports/ga4_events_*.csv'" in sql
    assert "CSVWithNames" in sql
    assert "'' AS experiment_id" in sql
    assert "'' AS variant_id" in sql
    assert "'' AS action_id" in sql
    assert "'' AS mapping_id" in sql
    assert "'' AS ad_id" in sql
    assert "'' AS creative_id" in sql
    assert "'{}' AS properties_json" in sql
    assert "CREATE TABLE" not in sql


def test_postgres_scripts_separate_schema_and_seed_data() -> None:
    init_sql = read_text("scripts/postgres/postgres-init.sql")
    seed_sql = read_text("scripts/postgres/postgres-seed.sql")

    assert "CREATE TABLE IF NOT EXISTS projects" in init_sql
    assert "CREATE TABLE IF NOT EXISTS automation_policies" in init_sql
    assert "CREATE TABLE IF NOT EXISTS segment_ad_mappings" in init_sql
    assert "INSERT INTO" not in init_sql

    assert "INSERT INTO projects" in seed_sql
    assert "INSERT INTO automation_policies" in seed_sql
    assert "INSERT INTO action_catalog" in seed_sql
    assert "CREATE TABLE" not in seed_sql


def test_postgres_init_analysis_jobs_matches_sqlalchemy_model() -> None:
    init_sql = read_text("scripts/postgres/postgres-init.sql")
    columns = AnalysisJob.__table__.columns

    assert "CREATE TABLE IF NOT EXISTS analysis_jobs" in init_sql
    assert "project_id VARCHAR(128) NOT NULL" in init_sql
    assert "status VARCHAR(32) NOT NULL DEFAULT 'queued'" in init_sql
    assert "request_json JSONB NOT NULL DEFAULT '{}'::jsonb" in init_sql
    assert "REFERENCES recommendation_results(id) ON DELETE SET NULL" in init_sql
    assert "error_message TEXT" in init_sql
    assert "attempts BIGINT NOT NULL DEFAULT 0" in init_sql
    assert "max_attempts BIGINT NOT NULL DEFAULT 1" in init_sql
    assert "locked_at TIMESTAMPTZ" in init_sql
    assert "started_at TIMESTAMPTZ" in init_sql
    assert "finished_at TIMESTAMPTZ" in init_sql
    assert "idx_analysis_jobs_project_id" in init_sql
    assert "idx_analysis_jobs_status" in init_sql
    assert "idx_analysis_jobs_recommendation_result" in init_sql
    assert "idx_analysis_jobs_status_created" in init_sql

    assert columns["project_id"].type.length == 128
    assert columns["status"].type.length == 32
    assert columns["status"].server_default.arg == "queued"
    assert columns["attempts"].server_default.arg.text == "0"
    assert columns["max_attempts"].server_default.arg.text == "1"
    assert columns["recommendation_result_id"].foreign_keys


def test_compose_mounts_database_init_and_csv_seed_paths() -> None:
    compose = read_text("docker-compose.yml")

    assert "./scripts/postgres:/docker-entrypoint-initdb.d:ro" in compose
    assert "./scripts/clickhouse:/docker-entrypoint-initdb.d:ro" in compose
    assert "./ga4_exports:/var/lib/clickhouse/user_files/ga4_exports:ro" in compose
    assert "decision-worker:" in compose
    assert 'command: ["python", "-m", "app.analysis.worker"]' in compose


def test_server_env_contract_uses_loopad_env_without_fallbacks() -> None:
    compose = read_text("docker-compose.yml")
    dockerfile = read_text("Dockerfile")
    env_example = read_text(".env.example")

    assert ":-" not in compose
    assert "CLICKHOUSE_HOST" not in compose
    assert "POSTGRES_HOST" not in compose
    assert "POSTGRES_PASSWORD" not in env_example
    assert "LOOPAD_ENV" in env_example
    assert "LOOPAD_SERVICE_ID=decision-api" in env_example
    assert "LOOPAD_CLICKHOUSE_URL=http://clickhouse:8123" in env_example
    assert 'os.environ[\\"PORT\\"]' in dockerfile
    assert 'CMD ["python", "-m", "app.main"]' in dockerfile


def test_deploy_workflow_uses_infra_ecs_reusable_workflow() -> None:
    workflow = read_text(".github/workflows/deploy.yml")

    assert "krafton-jungle-project-4team/loop-ad_infra/.github/workflows/ecs-deploy.yml@main" in workflow
    assert "service_name: decision-api" in workflow
    assert "ecr_repository: loop-ad/decision-api" in workflow
    assert "ecs_cluster: dev-loop-ad-cluster" in workflow
    assert "ecs_service: dev-decision-api" in workflow
    assert "container_name: decision-api" in workflow
