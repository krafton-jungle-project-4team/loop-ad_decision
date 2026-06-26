from pathlib import Path


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


def test_compose_mounts_database_init_and_csv_seed_paths() -> None:
    compose = read_text("docker-compose.yml")

    assert "./scripts/postgres:/docker-entrypoint-initdb.d:ro" in compose
    assert "./scripts/clickhouse:/docker-entrypoint-initdb.d:ro" in compose
    assert "./ga4_exports:/var/lib/clickhouse/user_files/ga4_exports:ro" in compose
