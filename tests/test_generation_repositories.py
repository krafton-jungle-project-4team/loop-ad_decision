from app.generation.repositories import (
    CONTENT_CANDIDATE_COLUMNS,
    GENERATION_RUN_COLUMNS,
    ContentCandidateRecord,
    ContentCandidateRepository,
    GenerationInputRepository,
    GenerationRunRecord,
    GenerationRunRepository,
)
from app.generation.schemas import ContentChannel, GenerationRequest


class FakeCursor:
    def __init__(
        self,
        *,
        fetchone_result: dict[str, object] | None = None,
        fetchall_result: list[dict[str, object]] | None = None,
    ) -> None:
        self.fetchone_result = fetchone_result
        self.fetchall_result = fetchall_result or []
        self.executed: list[tuple[str, dict[str, object] | None]] = []

    def __enter__(self) -> "FakeCursor":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def execute(self, query: str, params: dict[str, object] | None = None) -> None:
        self.executed.append((query, params))

    def fetchone(self) -> dict[str, object] | None:
        return self.fetchone_result

    def fetchall(self) -> list[dict[str, object]]:
        return self.fetchall_result


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self.cursor_instance = cursor
        self.row_factories: list[object] = []

    def cursor(self, *, row_factory: object = None) -> FakeCursor:
        self.row_factories.append(row_factory)
        return self.cursor_instance


def test_generation_run_repository_columns_match_data_source_contract() -> None:
    assert GENERATION_RUN_COLUMNS == (
        "generation_id",
        "analysis_id",
        "project_id",
        "campaign_id",
        "promotion_id",
        "content_option_count",
        "operator_instruction",
        "input_json",
        "output_json",
        "generation_report_json",
        "status",
        "created_at",
        "updated_at",
    )


def test_content_candidate_repository_columns_match_data_source_contract() -> None:
    assert CONTENT_CANDIDATE_COLUMNS == (
        "content_id",
        "content_option_id",
        "generation_id",
        "analysis_id",
        "project_id",
        "campaign_id",
        "promotion_id",
        "segment_id",
        "channel",
        "subject",
        "preheader",
        "title",
        "body",
        "cta",
        "message",
        "image_prompt",
        "image_url",
        "landing_url",
        "generation_prompt",
        "reason_summary",
        "data_evidence_json",
        "message_strategy",
        "metadata_json",
        "status",
        "created_at",
        "updated_at",
    )


def test_generation_run_repository_create_executes_insert() -> None:
    cursor = FakeCursor(fetchone_result={"generation_id": "generation_banner_001"})
    repository = GenerationRunRepository(FakeConnection(cursor))

    result = repository.create(
        GenerationRunRecord(
            generation_id="generation_banner_001",
            analysis_id="analysis_banner_001",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            content_option_count=2,
            operator_instruction=None,
            input_json={"analysis_id": "analysis_banner_001"},
            output_json={"content_candidate_ids": ["content_banner_001"]},
            generation_report_json={"content_candidate_count": 2},
            status="completed",
        )
    )

    assert result == {"generation_id": "generation_banner_001"}
    query, params = cursor.executed[0]
    assert "INSERT INTO generation_runs" in query
    assert params is not None
    assert params["generation_id"] == "generation_banner_001"
    assert params["input_json"].obj == {"analysis_id": "analysis_banner_001"}
    assert params["output_json"].obj == {
        "content_candidate_ids": ["content_banner_001"]
    }
    assert params["generation_report_json"].obj == {"content_candidate_count": 2}


def test_generation_run_repository_lists_ids_by_promotion() -> None:
    cursor = FakeCursor(
        fetchall_result=[
            {"generation_id": "generation_banner_001"},
            {"generation_id": "generation_banner_001_run_2"},
        ]
    )
    repository = GenerationRunRepository(FakeConnection(cursor))

    result = repository.list_ids_by_promotion("promo_banner_001")

    assert result == ["generation_banner_001", "generation_banner_001_run_2"]
    query, params = cursor.executed[0]
    assert "FROM generation_runs" in query
    assert "promotion_id = %(promotion_id)s" in query
    assert params == {"promotion_id": "promo_banner_001"}


def test_content_candidate_repository_create_executes_insert() -> None:
    cursor = FakeCursor(fetchone_result={"content_id": "content_banner_001"})
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.create(
        ContentCandidateRecord(
            content_id="content_banner_001",
            content_option_id="banner_option_001",
            generation_id="generation_banner_001",
            analysis_id="analysis_banner_001",
            project_id="hotel-client-a",
            campaign_id="camp_summer_2026",
            promotion_id="promo_banner_001",
            segment_id="seg_repeat_hotel_no_booking",
            channel=ContentChannel.ONSITE_BANNER,
            title="Book this weekend's rooms",
            body="Compare refundable summer offers before rooms run out.",
            cta="View hotel deals",
            image_prompt="bright modern hotel room, summer travel banner",
            image_url="https://gen-ai.asset.dev.loop-ad.org/generated-assets/content_banner_001.png",
            landing_url="https://demo-stay.example.com/summer",
            generation_prompt="Create an onsite banner.",
            data_evidence_json={"segment_id": "seg_repeat_hotel_no_booking"},
            metadata_json={"content_id": "content_banner_001"},
        )
    )

    assert result == {"content_id": "content_banner_001"}
    query, params = cursor.executed[0]
    assert "INSERT INTO content_candidates" in query
    assert params is not None
    assert params["content_id"] == "content_banner_001"
    assert params["channel"] == "onsite_banner"
    assert params["image_url"] == (
        "https://gen-ai.asset.dev.loop-ad.org/generated-assets/content_banner_001.png"
    )
    assert params["data_evidence_json"].obj == {
        "segment_id": "seg_repeat_hotel_no_booking"
    }
    assert params["metadata_json"].obj == {"content_id": "content_banner_001"}


def test_content_candidate_repository_lists_by_generation() -> None:
    rows = [{"content_id": "content_banner_001"}]
    cursor = FakeCursor(fetchall_result=rows)
    repository = ContentCandidateRepository(FakeConnection(cursor))

    result = repository.list_by_generation("generation_banner_001")

    assert result == rows
    query, params = cursor.executed[0]
    assert "FROM content_candidates" in query
    assert params == {"generation_id": "generation_banner_001"}


def test_generation_input_repository_reads_confirmed_target_segments() -> None:
    cursor = FakeCursor(
        fetchone_result={
            "project_id": "hotel-client-a",
            "campaign_id": "camp_summer_2026",
            "promotion_id": "promo_banner_001",
            "channel": "onsite_banner",
            "goal_metric": "booking_conversion_rate",
            "goal_target_value": "0.030000",
            "goal_basis": "all_segments",
            "message_brief": "Drive hotel booking conversion.",
            "landing_url": "https://demo-stay.example.com/summer",
        },
        fetchall_result=[
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_ai_repeat_hotel",
                "segment_name": "AI suggested repeat hotel viewers",
                "content_brief_json": {
                    "message_direction": "Highlight refundable hotel stays.",
                    "keywords": ["refundable stays", "hotel deals"],
                },
                "data_evidence_json": {
                    "source": "ai_suggested",
                    "booking_conversion_rate": "0.018",
                    "top_common_features": ["same_hotel_repeat_view"],
                    "sample_ratio": "0.018000",
                },
                "segment_vector_id": "segvec_ai_repeat_hotel_v1",
                "estimated_size": 1342,
                "priority": "high",
                "segment_source": "ai_suggested",
                "query_preview_id": "seg_query_preview_001",
                "natural_language_query": "repeat hotel viewers without booking",
                "generated_sql": "SELECT user_id FROM hotel_detail_events",
                "segment_sample_size": 1342,
                "segment_sample_ratio": "0.018000",
            },
            {
                "analysis_id": "analysis_banner_001",
                "promotion_id": "promo_banner_001",
                "segment_id": "seg_manual_family_trip",
                "segment_name": "Manual family trip segment",
                "content_brief_json": {
                    "message_direction": "Promote family-friendly rooms.",
                    "keywords": ["family rooms"],
                },
                "data_evidence_json": {"source": "manual_rule"},
                "segment_vector_id": "segvec_manual_family_trip_v1",
                "estimated_size": 820,
                "priority": "medium",
                "segment_source": "manual_rule",
                "query_preview_id": None,
                "natural_language_query": "family hotel trip planners",
                "generated_sql": None,
                "segment_sample_size": 820,
                "segment_sample_ratio": "0.011000",
            },
        ],
    )
    repository = GenerationInputRepository(FakeConnection(cursor))
    request = GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=2,
        operator_instruction=None,
    )

    promotion = repository.get_promotion_input(request)
    target_segments = repository.list_target_segment_inputs(request)

    assert promotion is not None
    assert promotion.channel == ContentChannel.ONSITE_BANNER
    assert promotion.landing_url == "https://demo-stay.example.com/summer"
    assert [segment.segment_id for segment in target_segments] == [
        "seg_ai_repeat_hotel",
        "seg_manual_family_trip",
    ]
    assert target_segments[0].source == "ai_suggested"
    assert target_segments[1].source == "manual_rule"
    assert target_segments[0].content_brief_json["booking_conversion_rate"] == "0.018"
    assert target_segments[0].natural_language_query == (
        "repeat hotel viewers without booking"
    )
    assert target_segments[0].generated_sql == (
        "SELECT user_id FROM hotel_detail_events"
    )
    assert target_segments[0].query_preview_id == "seg_query_preview_001"

    executed_sql = "\n".join(query for query, _params in cursor.executed)
    assert "FROM promotion_target_segments" in executed_sql
    assert "LEFT JOIN segment_definitions" in executed_sql
    assert "promotion_segment_suggestions" not in executed_sql
