from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def test_compose_excludes_local_database_services_and_init_paths() -> None:
    compose = read_text("docker-compose.yml")

    assert "\n  postgres:" not in compose
    assert "\n  clickhouse:" not in compose
    assert "depends_on:" not in compose
    assert "./scripts/postgres" not in compose
    assert "./scripts/clickhouse" not in compose
    assert "./ga4_exports" not in compose
    assert "postgres-data" not in compose
    assert "clickhouse-data" not in compose
    assert "clickhouse-logs" not in compose
    assert "decision-worker:" in compose
    assert 'command: ["python", "-m", "app.analysis.worker"]' in compose


def test_server_env_contract_uses_loopad_env_without_fallbacks() -> None:
    compose = read_text("docker-compose.yml")
    dockerfile = read_text("Dockerfile")
    env_example = read_text(".env.example")

    assert ":-" not in compose
    assert "CLICKHOUSE_HOST" not in compose
    assert "POSTGRES_HOST" not in compose
    assert "CLICKHOUSE_HOST" not in env_example
    assert "POSTGRES_HOST" not in env_example
    assert "POSTGRES_PASSWORD" not in env_example
    assert "LOOPAD_ENV" in env_example
    assert "LOOPAD_SERVICE_ID=decision-api" in env_example
    assert "LOOPAD_RUNTIME" not in env_example
    assert "LOOPAD_RUNTIME" not in compose
    assert "LOOPAD_AURORA_HOST=localhost" in env_example
    assert "LOOPAD_AURORA_PORT=15432" in env_example
    assert "LOOPAD_AURORA_DATABASE=loopad" in env_example
    assert "LOOPAD_AURORA_USERNAME=loopad" in env_example
    assert "LOOPAD_AURORA_PASSWORD=loopad" in env_example
    assert "LOOPAD_CLICKHOUSE_URL=http://localhost:18123" in env_example
    assert "LOOPAD_CLICKHOUSE_USERNAME=default" in env_example
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
