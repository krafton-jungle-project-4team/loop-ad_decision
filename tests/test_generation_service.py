from app.generation.repositories import (
    ContentCandidateRecord,
    GenerationRunRecord,
)
from app.generation.schemas import (
    ContentChannel,
    GenerationRequest,
)
from app.generation.service import GenerationService


class FakeGenerationRunRepository:
    def __init__(self) -> None:
        self.saved: list[GenerationRunRecord] = []

    def create(self, record: GenerationRunRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"generation_id": record.generation_id}


class FakeContentCandidateRepository:
    def __init__(self) -> None:
        self.saved: list[ContentCandidateRecord] = []

    def create(self, record: ContentCandidateRecord) -> dict[str, object]:
        self.saved.append(record)
        return {"content_id": record.content_id}


def generation_request(
    *,
    content_option_count: int = 2,
    operator_instruction: str | None = "Make the banner direct and concise.",
) -> GenerationRequest:
    return GenerationRequest(
        project_id="hotel-client-a",
        campaign_id="camp_summer_2026",
        promotion_id="promo_banner_001",
        analysis_id="analysis_banner_001",
        content_option_count=content_option_count,
        operator_instruction=operator_instruction,
    )


def test_generation_service_persists_run_and_content_candidates() -> None:
    generation_run_repository = FakeGenerationRunRepository()
    content_candidate_repository = FakeContentCandidateRepository()
    service = GenerationService(
        generation_run_repository=generation_run_repository,
        content_candidate_repository=content_candidate_repository,
    )

    response = service.generate(generation_request(content_option_count=2))

    assert response.generation_id == "generation_banner_001"
    assert response.promotion_id == "promo_banner_001"
    assert response.status == "completed"
    assert len(response.content_candidates) == 2

    assert len(generation_run_repository.saved) == 1
    generation_run = generation_run_repository.saved[0]
    assert generation_run.generation_id == response.generation_id
    assert generation_run.project_id == "hotel-client-a"
    assert generation_run.status == "completed"
    assert generation_run.input_json == {
        "project_id": "hotel-client-a",
        "campaign_id": "camp_summer_2026",
        "promotion_id": "promo_banner_001",
        "analysis_id": "analysis_banner_001",
        "content_option_count": 2,
        "operator_instruction": "Make the banner direct and concise.",
    }
    assert generation_run.output_json == {
        "content_candidate_ids": [
            "content_banner_repeat_hotel_001",
            "content_banner_repeat_hotel_002",
        ],
    }
    assert generation_run.generation_report_json == {
        "status": "completed",
        "content_candidate_count": 2,
    }

    assert len(content_candidate_repository.saved) == 2
    first_candidate = content_candidate_repository.saved[0]
    assert first_candidate.content_id == "content_banner_repeat_hotel_001"
    assert first_candidate.content_option_id == "banner_repeat_hotel_option_001"
    assert first_candidate.generation_id == response.generation_id
    assert first_candidate.project_id == "hotel-client-a"
    assert first_candidate.channel == ContentChannel.ONSITE_BANNER
    assert first_candidate.generation_prompt
    assert first_candidate.metadata_json["content_id"] == first_candidate.content_id
    assert first_candidate.metadata_json["channel"] == "onsite_banner"


def test_generation_service_can_generate_response_without_repositories() -> None:
    service = GenerationService()

    response = service.generate(generation_request(content_option_count=1))

    assert response.generation_id == "generation_banner_001"
    assert len(response.content_candidates) == 1
    assert response.content_candidates[0].content_option_id == (
        "banner_repeat_hotel_option_001"
    )
