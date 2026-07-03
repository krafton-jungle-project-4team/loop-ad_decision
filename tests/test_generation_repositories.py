from app.repositories.content_candidates import CONTENT_CANDIDATE_COLUMNS
from app.repositories.generation_runs import GENERATION_RUN_COLUMNS


def test_generation_run_repository_columns_match_data_source_contract() -> None:
    assert GENERATION_RUN_COLUMNS == (
        "generation_id",
        "project_id",
        "campaign_id",
        "promotion_id",
        "analysis_id",
        "content_option_count",
        "operator_instruction",
        "prompt_context_json",
        "report_json",
        "status",
        "created_at",
        "updated_at",
    )


def test_content_candidate_repository_columns_match_data_source_contract() -> None:
    assert CONTENT_CANDIDATE_COLUMNS == (
        "content_id",
        "content_option_id",
        "project_id",
        "campaign_id",
        "promotion_id",
        "analysis_id",
        "generation_id",
        "segment_id",
        "segment_name",
        "channel",
        "subject",
        "preheader",
        "title",
        "body",
        "cta",
        "message",
        "image_prompt",
        "landing_url",
        "reason_summary",
        "data_evidence_json",
        "message_strategy",
        "payload_json",
        "status",
        "approved_at",
        "approved_by",
        "created_at",
        "updated_at",
    )
