from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import status

from app.contents.ai.gemini_image_provider import GeminiImageProvider
from app.contents.ai.mock_image_provider import MockImageProvider
from app.contents.contents_service import ContentsService
from app.contents.errors import ContentGenerationError
from app.contents.schemas import GenerateContentRequest
from app.contents.storage.s3_asset_storage_service import StorageObject


_DEFAULT = object()


class RecordingStorage:
    def __init__(self) -> None:
        self.uploaded: dict[str, StorageObject] = {}

    def public_url_for_key(self, key: str) -> str:
        return f"https://cdn.example.com/{key}"

    def upload_objects(self, objects: list[StorageObject]) -> None:
        self.uploaded.update({item.key: item for item in objects})


class FakePostgresRepository:
    def __init__(
        self,
        *,
        recommendation_action: object = _DEFAULT,
        action_catalog_item: object = _DEFAULT,
        mapping: object = _DEFAULT,
        ad_creative: object = _DEFAULT,
    ) -> None:
        self.recommendation_action = (
            default_recommendation_action()
            if recommendation_action is _DEFAULT
            else recommendation_action
        )
        self.action_catalog_item = (
            default_action_catalog_item()
            if action_catalog_item is _DEFAULT
            else action_catalog_item
        )
        self.mapping = default_mapping() if mapping is _DEFAULT else mapping
        self.ad_creative = default_ad_creative() if ad_creative is _DEFAULT else ad_creative
        self.committed = False
        self.rolled_back = False
        self.updated_urls: list[str] = []

    def get_recommendation_action_by_result_action(
        self,
        *,
        recommendation_result_id: int,
        action_id: str,
    ) -> SimpleNamespace | None:
        action = self.recommendation_action
        if action is None:
            return None
        if action.recommendation_result_id == recommendation_result_id and action.action_id == action_id:
            return action
        return None

    def get_active_action_catalog_item(self, action_id: str) -> SimpleNamespace | None:
        action = self.action_catalog_item
        if action is None:
            return None
        if action.action_id == action_id and action.status == "active":
            return action
        return None

    def get_content_generation_target(
        self,
        *,
        recommendation_action_id: int,
        project_id: str,
    ) -> SimpleNamespace | None:
        mapping = self.mapping
        creative = self.ad_creative
        if mapping is None or creative is None:
            return None
        if mapping.recommendation_action_id != recommendation_action_id:
            return None
        if mapping.project_id != project_id:
            return None
        if mapping.status != "active":
            return None
        if mapping.expires_at is not None and mapping.expires_at <= datetime.now(UTC):
            return None
        if mapping.creative_id != creative.id:
            return None
        if creative.status != "active":
            return None
        if creative.project_id != mapping.project_id:
            return None
        return SimpleNamespace(mapping=mapping, creative=creative)

    def update_ad_creative_image_url(
        self,
        creative_id: int,
        image_url: str,
    ) -> SimpleNamespace | None:
        creative = self.ad_creative
        if creative is None or creative.id != creative_id:
            return None
        creative.image_url = image_url
        self.updated_urls.append(image_url)
        return creative

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        self.rolled_back = True


def default_recommendation_action() -> SimpleNamespace:
    return SimpleNamespace(
        id=10,
        project_id="google-ga4-demo-commerce",
        recommendation_result_id=1,
        action_id="act_free_shipping_coupon",
        action_type="coupon",
        title="Offer free shipping through Kakao",
        description="Send a fresh food free-shipping coupon to Kakao-inbound users.",
        target_step="checkout",
        expected_impact="Recover fresh food checkout drop",
        execution_hint_json={"coupon_code": "CPN_FREE_SHIPPING"},
    )


def default_action_catalog_item() -> SimpleNamespace:
    return SimpleNamespace(
        action_id="act_free_shipping_coupon",
        action_type="coupon",
        title="Free shipping coupon",
        description="Offer free shipping.",
        target_step="checkout",
        expected_impact="Recover checkout drop",
        execution_hint_json={"coupon_type": "free_shipping"},
        status="active",
    )


def default_mapping(**overrides: object) -> SimpleNamespace:
    values = {
        "id": 20,
        "project_id": "google-ga4-demo-commerce",
        "recommendation_action_id": 10,
        "creative_id": 1,
        "status": "active",
        "expires_at": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def default_ad_creative(**overrides: object) -> SimpleNamespace:
    values = {
        "id": 1,
        "project_id": "google-ga4-demo-commerce",
        "action_id": "act_free_shipping_coupon",
        "creative_type": "message",
        "title": "Offer free shipping through Kakao",
        "message": "Fresh food free shipping coupon.",
        "image_url": None,
        "landing_url": "https://shop.example.com/",
        "payload_json": {"delivery_channel": "kakao"},
        "status": "active",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def create_service(
    repository: FakePostgresRepository,
    storage: RecordingStorage,
) -> ContentsService:
    return ContentsService(
        repository=repository,
        image_provider=MockImageProvider(),
        storage=storage,
        asset_key_prefix="genai/generated/",
    )


def test_generate_content_uses_recommendation_action_mapping_and_updates_final_url() -> None:
    repository = FakePostgresRepository()
    storage = RecordingStorage()
    service = create_service(repository, storage)

    response = service.generate_content(
        GenerateContentRequest(
            recommendation_result_id=1,
            action_id="act_free_shipping_coupon",
        )
    )

    assert response.creative_id == "1"
    assert response.action_id == "act_free_shipping_coupon"
    assert response.recommendation_action_id == 10
    assert response.mapping_id == 20
    assert response.content_url.endswith("/final/banner.png")
    assert response.content_url.startswith("https://cdn.example.com/genai/generated/")
    assert repository.ad_creative.image_url == response.content_url
    assert repository.committed is True
    assert len(storage.uploaded) == 6
    assert all(key.startswith("genai/generated/ad-creatives/") for key in storage.uploaded)
    assert any(key.endswith("/source/creative-brief.json") for key in storage.uploaded)
    assert any(key.endswith("/source/copy.json") for key in storage.uploaded)
    assert any(key.endswith("/assets/background.png") for key in storage.uploaded)
    assert any(key.endswith("/source/composition.svg") for key in storage.uploaded)
    assert any(key.endswith("/source/manifest.json") for key in storage.uploaded)
    assert any(key.endswith("/final/banner.png") for key in storage.uploaded)
    final_banner_key = next(
        key for key in storage.uploaded if key.endswith("/final/banner.png")
    )
    assert storage.uploaded[final_banner_key].body.startswith(b"\x89PNG\r\n\x1a\n")


def test_generate_content_returns_404_when_active_mapping_is_missing() -> None:
    repository = FakePostgresRepository(mapping=None)
    service = create_service(repository, RecordingStorage())

    with pytest.raises(ContentGenerationError) as exc_info:
        service.generate_content(
            GenerateContentRequest(
                recommendation_result_id=1,
                action_id="act_free_shipping_coupon",
            )
        )

    assert exc_info.value.code == "SEGMENT_AD_MAPPING_NOT_FOUND"
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert repository.rolled_back is True


@pytest.mark.parametrize(
    "mapping",
    [
        default_mapping(status="inactive"),
        default_mapping(expires_at=datetime.now(UTC) - timedelta(seconds=1)),
    ],
)
def test_generate_content_does_not_use_inactive_or_expired_mapping(
    mapping: SimpleNamespace,
) -> None:
    repository = FakePostgresRepository(mapping=mapping)
    service = create_service(repository, RecordingStorage())

    with pytest.raises(ContentGenerationError) as exc_info:
        service.generate_content(
            GenerateContentRequest(
                recommendation_result_id=1,
                action_id="act_free_shipping_coupon",
            )
        )

    assert exc_info.value.code == "SEGMENT_AD_MAPPING_NOT_FOUND"
    assert repository.rolled_back is True


@pytest.mark.parametrize(
    "creative",
    [
        default_ad_creative(status="inactive"),
        default_ad_creative(project_id="other-project"),
    ],
)
def test_generate_content_does_not_use_inactive_or_project_mismatched_creative(
    creative: SimpleNamespace,
) -> None:
    repository = FakePostgresRepository(ad_creative=creative)
    service = create_service(repository, RecordingStorage())

    with pytest.raises(ContentGenerationError) as exc_info:
        service.generate_content(
            GenerateContentRequest(
                recommendation_result_id=1,
                action_id="act_free_shipping_coupon",
            )
        )

    assert exc_info.value.code == "SEGMENT_AD_MAPPING_NOT_FOUND"
    assert repository.rolled_back is True


def test_generate_content_reuses_existing_url_when_force_false() -> None:
    repository = FakePostgresRepository(
        ad_creative=default_ad_creative(
            image_url="https://cdn.example.com/existing/final/banner.png",
        )
    )
    storage = RecordingStorage()
    service = create_service(repository, storage)

    response = service.generate_content(
        GenerateContentRequest(
            recommendation_result_id=1,
            action_id="act_free_shipping_coupon",
            force=False,
        )
    )

    assert response.content_url == "https://cdn.example.com/existing/final/banner.png"
    assert response.recommendation_action_id == 10
    assert response.mapping_id == 20
    assert storage.uploaded == {}
    assert repository.committed is False


def test_generate_content_regenerates_existing_url_when_force_true() -> None:
    repository = FakePostgresRepository(
        ad_creative=default_ad_creative(
            image_url="https://cdn.example.com/existing/final/banner.png",
        )
    )
    storage = RecordingStorage()
    service = create_service(repository, storage)

    response = service.generate_content(
        GenerateContentRequest(
            recommendation_result_id=1,
            action_id="act_free_shipping_coupon",
            force=True,
        )
    )

    assert response.content_url != "https://cdn.example.com/existing/final/banner.png"
    assert "/generations/gen_" in response.content_url
    assert response.content_url.endswith("/final/banner.png")
    assert repository.ad_creative.image_url == response.content_url
    assert repository.committed is True
    assert len(storage.uploaded) == 6


def test_generate_content_raises_required_code_when_recommendation_action_is_missing() -> None:
    repository = FakePostgresRepository(recommendation_action=None)
    service = create_service(repository, RecordingStorage())

    with pytest.raises(ContentGenerationError) as exc_info:
        service.generate_content(
            GenerateContentRequest(
                recommendation_result_id=1,
                action_id="act_free_shipping_coupon",
            )
        )

    assert exc_info.value.code == "RECOMMENDATION_ACTION_NOT_FOUND"
    assert exc_info.value.status_code == status.HTTP_404_NOT_FOUND
    assert repository.rolled_back is True


def test_gemini_provider_parses_interactions_image_response() -> None:
    provider = GeminiImageProvider(api_key=None, model="gemini-3.1-flash-image")

    image = provider._parse_image_response(
        b'{"output_image":{"data":"iVBORw0KGgo=","mime_type":"image/png"}}'
    )

    assert image.body == b"\x89PNG\r\n\x1a\n"
    assert image.content_type == "image/png"
    assert image.provider_name == "gemini"
    assert image.model == "gemini-3.1-flash-image"
