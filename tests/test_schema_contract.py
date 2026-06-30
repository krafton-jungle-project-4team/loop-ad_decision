from pathlib import Path


def test_active_ad_serving_rules_exposes_tracking_payload_fields() -> None:
    schema = Path("schema.sql").read_text(encoding="utf-8")
    view_sql = schema.split("CREATE OR REPLACE VIEW active_ad_serving_rules AS", 1)[1]
    view_sql = view_sql.split("WHERE m.is_active = true", 1)[0]

    assert "p.project_key" in view_sql
    assert "m.id AS mapping_id" in view_sql
    assert "JOIN projects p" in view_sql
    assert "ON p.id = m.project_id" in view_sql
    assert "m.experiment_id" in view_sql
    assert "m.experiment_variant_id" in view_sql
    assert "m.generated_content_id" in view_sql
    assert "ev.variant_key" in view_sql
