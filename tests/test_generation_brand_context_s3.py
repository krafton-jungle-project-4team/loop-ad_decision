from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import pytest
from structlog.testing import capture_logs

from app.generation.brand_context import BrandGuardrails
from app.generation.brand_context_s3 import S3BrandContextLoader
from app.generation.errors import PermanentGenerationError
from app.generation.schemas import ContentChannel


BUCKET = "loopad-test-data"
PREFIX = "brand-context/"
PROJECT_ID = "demo_project"
CURRENT_KEY = f"{PREFIX}{PROJECT_ID}/current.json"
MANIFEST_KEY = f"{PREFIX}{PROJECT_ID}/manifests/v1/manifest.json"
BRAND_KIT_KEY = f"{PREFIX}{PROJECT_ID}/brand-kits/v1/brand-kit.json"
BRAND_VOICE_KEY = (
    f"{PREFIX}{PROJECT_ID}/guidelines/brand-voice/v1/content.md"
)
PHOTO_RECIPE_KEY = (
    f"{PREFIX}{PROJECT_ID}/guidelines/photo-recipe/v1/content.md"
)
HOME_HERO_KEY = f"{PREFIX}{PROJECT_ID}/assets/home-hero/v1/original.jpg"
OFFER_CATALOG_KEY = (
    f"{PREFIX}{PROJECT_ID}/catalogs/black-friday-hotels/v2/catalog.json"
)
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
MARKDOWN_CONTENT_TYPE = "text/markdown; charset=utf-8"


class FakeS3Error(RuntimeError):
    def __init__(self, code: str, status_code: int) -> None:
        self.response = {
            "Error": {"Code": code},
            "ResponseMetadata": {"HTTPStatusCode": status_code},
        }
        super().__init__(code)


class FakeS3Client:
    def __init__(self, objects: dict[str, tuple[bytes, str]]) -> None:
        self.objects = objects
        self.get_calls: list[str] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        assert Bucket == BUCKET
        self.get_calls.append(Key)
        value = self.objects.get(Key)
        if value is None:
            raise FakeS3Error("NoSuchKey", 404)
        body, content_type = value
        return {
            "Body": io.BytesIO(body),
            "ContentLength": len(body),
            "ContentType": content_type,
        }


def test_s3_loader_resolves_pointer_and_loads_required_context() -> None:
    s3_client = FakeS3Client(bundle_objects())
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=s3_client,
    )

    snapshot = loader.resolve_snapshot(project_id=PROJECT_ID)

    assert snapshot is not None
    assert snapshot.context_version == "v1"
    assert snapshot.manifest_key == MANIFEST_KEY
    assert snapshot.guide_version == "v1"
    assert snapshot.asset_manifest_version == "v1"
    assert s3_client.get_calls == [CURRENT_KEY, MANIFEST_KEY]

    documents = loader.load_documents(
        project_id=PROJECT_ID,
        snapshot=snapshot,
        channel=ContentChannel.ONSITE_BANNER,
    )

    assert [document.source_id for document in documents] == [
        "brand-voice",
        "photo-recipe",
        "home-hero",
    ]
    assert documents[-1].s3_key == HOME_HERO_KEY
    assert "StayLoop 대표 호텔 이미지" in documents[-1].document_text
    guardrails = BrandGuardrails.from_documents(documents)
    assert guardrails.forbidden_terms == ("최저가 보장", "100% 만족")
    assert guardrails.approved_colours == ("#1668e3", "#ffffff")
    assert s3_client.get_calls == [
        CURRENT_KEY,
        MANIFEST_KEY,
        BRAND_KIT_KEY,
        BRAND_VOICE_KEY,
        PHOTO_RECIPE_KEY,
        HOME_HERO_KEY,
    ]


def test_s3_loader_loads_only_sms_applicable_guides_without_asset() -> None:
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=FakeS3Client(bundle_objects()),
    )
    snapshot = loader.resolve_snapshot(project_id=PROJECT_ID)
    assert snapshot is not None

    documents = loader.load_documents(
        project_id=PROJECT_ID,
        snapshot=snapshot,
        channel=ContentChannel.SMS,
    )

    assert [document.source_id for document in documents] == ["brand-voice"]


def test_s3_loader_loads_verified_offer_catalog_with_public_image_path() -> None:
    catalog_bytes = _json_bytes(
        {
            "schema_version": "stayloop.promotion-price-catalog.v1",
            "project_id": PROJECT_ID,
            "catalog_id": "black-friday-hotels",
            "catalog_version": "v2",
            "promotion_label": "제주·오키나와 블랙프라이데이",
            "price_basis": "one_room_one_night",
            "currency": "KRW",
            "hotels": [
                {
                    "hotel_id": "jeju-ocean-breeze-006",
                    "hotel_name": "Jeju Ocean Breeze Resort",
                    "destination_id": "jeju",
                    "currency": "KRW",
                    "sale_price_per_night": 278000,
                    "original_price_per_night": 342000,
                    "discount_rate_percent": 19,
                }
            ],
        }
    )
    hotel_image_bytes = b"hotel-image"
    hotel_image_key = (
        f"{PREFIX}{PROJECT_ID}/assets/"
        "hotel-jeju-ocean-breeze-006-hero/v2/original.png"
    )
    objects = bundle_objects(
        manifest_patch={
            "context_version": "v1",
            "catalogs": [
                {
                    "catalog_id": "black-friday-hotels",
                    **reference(
                        OFFER_CATALOG_KEY,
                        catalog_bytes,
                        JSON_CONTENT_TYPE,
                        version="v2",
                    ),
                    "required": True,
                    "applies_to": ["email"],
                }
            ],
            "assets": [
                {
                    "asset_id": "hotel-jeju-ocean-breeze-006-hero",
                    **reference(
                        hotel_image_key,
                        hotel_image_bytes,
                        "image/png",
                        version="v2",
                    ),
                    "role": "hotel",
                    "active": True,
                    "advertising_use": "demo_only",
                    "frontend_path": (
                        "/stayloop/promotions/jeju-resort-exterior.png"
                    ),
                    "entity_refs": [
                        {
                            "type": "hotel",
                            "id": "jeju-ocean-breeze-006",
                            "usage": "primary",
                        },
                        {"type": "destination", "id": "jeju"},
                    ],
                }
            ],
        }
    )
    objects[OFFER_CATALOG_KEY] = (catalog_bytes, JSON_CONTENT_TYPE)
    objects[hotel_image_key] = (hotel_image_bytes, "image/png")
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=FakeS3Client(objects),
    )
    snapshot = loader.resolve_snapshot(project_id=PROJECT_ID)
    assert snapshot is not None

    catalog = loader.load_offer_catalog(
        project_id=PROJECT_ID,
        snapshot=snapshot,
    )

    assert catalog is not None
    assert catalog["catalog_version"] == "v2"
    assert catalog["hotels"] == [
        {
            "offer_id": "jeju-ocean-breeze-006",
            "hotel_name": "Jeju Ocean Breeze Resort",
            "destination_id": "jeju",
            "currency": "KRW",
            "sale_price_per_night": 278000,
            "original_price_per_night": 342000,
            "discount_rate_percent": 19,
            "image_path": "/stayloop/promotions/jeju-resort-exterior.png",
            "asset_id": "hotel-jeju-ocean-breeze-006-hero",
        }
    ]
    assert OFFER_CATALOG_KEY in loader._s3_client.get_calls


def test_s3_loader_returns_none_when_project_pointer_is_absent() -> None:
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=FakeS3Client({}),
    )

    with capture_logs() as logs:
        assert loader.resolve_snapshot(project_id=PROJECT_ID) is None

    completed = next(
        record
        for record in logs
        if record["event"] == "provider_request_completed"
    )
    assert completed["provider"] == "aws_s3"
    assert completed["endpoint"] == "get_object"
    assert completed["outcome"] == "not_found"


def test_s3_loader_accepts_json_content_type_without_charset() -> None:
    objects = bundle_objects()
    objects[CURRENT_KEY] = (objects[CURRENT_KEY][0], "application/json")
    objects[MANIFEST_KEY] = (objects[MANIFEST_KEY][0], "application/json")
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=FakeS3Client(objects),
    )

    assert loader.resolve_snapshot(project_id=PROJECT_ID) is not None


def test_s3_loader_rejects_manifest_checksum_mismatch() -> None:
    objects = bundle_objects()
    pointer = json.loads(objects[CURRENT_KEY][0])
    pointer["manifest_sha256"] = "0" * 64
    objects[CURRENT_KEY] = (_json_bytes(pointer), JSON_CONTENT_TYPE)
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=FakeS3Client(objects),
    )

    with pytest.raises(PermanentGenerationError) as exc_info:
        loader.resolve_snapshot(project_id=PROJECT_ID)

    assert exc_info.value.code == "brand_context_manifest_checksum_mismatch"


def test_s3_loader_rejects_cross_project_manifest_reference() -> None:
    objects = bundle_objects(
        manifest_patch={
            "brand_kit": reference(
                "brand-context/other_project/brand-kits/v1/brand-kit.json",
                _json_bytes({"brand": {"name": "StayLoop"}}),
                JSON_CONTENT_TYPE,
                version="v1",
            )
        }
    )
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=FakeS3Client(objects),
    )

    with pytest.raises(PermanentGenerationError) as exc_info:
        loader.resolve_snapshot(project_id=PROJECT_ID)

    assert exc_info.value.code == "brand_context_key_invalid"


def test_s3_loader_rejects_loaded_object_checksum_mismatch() -> None:
    objects = bundle_objects()
    objects[BRAND_VOICE_KEY] = (b"changed", MARKDOWN_CONTENT_TYPE)
    loader = S3BrandContextLoader(
        bucket_name=BUCKET,
        base_prefix=PREFIX,
        s3_client=FakeS3Client(objects),
    )
    snapshot = loader.resolve_snapshot(project_id=PROJECT_ID)
    assert snapshot is not None

    with pytest.raises(PermanentGenerationError) as exc_info:
        loader.load_documents(
            project_id=PROJECT_ID,
            snapshot=snapshot,
            channel=ContentChannel.ONSITE_BANNER,
        )

    assert exc_info.value.code in {
        "brand_context_object_size_mismatch",
        "brand_context_object_checksum_mismatch",
    }


def bundle_objects(
    *,
    manifest_patch: dict[str, Any] | None = None,
) -> dict[str, tuple[bytes, str]]:
    brand_kit_bytes = _json_bytes(
        {
            "schemaVersion": "stayloop.brand_kit.v1",
            "brand": {
                "id": "stayloop",
                "name": "StayLoop",
                "category": "hotel_booking",
                "locale": "ko-KR",
            },
            "color": {
                "primary": "#1668e3",
                "surface": "#ffffff",
                "duplicate": "#1668e3",
            },
        }
    )
    brand_voice_bytes = (
        "# StayLoop Brand Voice\n\n"
        "## 기본 원칙\n\n- 짧고 분명하게 말한다.\n\n"
        "## 금지 문구\n\n- \"최저가 보장\"\n- \"100% 만족\"\n\n"
        "## 이메일\n\n- 차분하게 안내한다.\n"
    ).encode("utf-8")
    photo_recipe_bytes = (
        "# StayLoop Photo Recipe\n\n"
        "## 공통 기준\n\n- 이미지에 글자를 굽지 않는다.\n"
    ).encode("utf-8")
    home_hero_bytes = b"home-hero-image"
    manifest: dict[str, Any] = {
        "schema_version": "loopad.brand-context-manifest.v1",
        "project_id": PROJECT_ID,
        "brand_id": "stayloop",
        "context_version": "v1",
        "created_at": "2026-07-13T00:00:00Z",
        "brand_kit": reference(
            BRAND_KIT_KEY,
            brand_kit_bytes,
            JSON_CONTENT_TYPE,
            version="v1",
        ),
        "guidelines": [
            {
                "guide_id": "brand-voice",
                **reference(
                    BRAND_VOICE_KEY,
                    brand_voice_bytes,
                    MARKDOWN_CONTENT_TYPE,
                    version="v1",
                ),
                "required": True,
                "applies_to": ["email", "onsite_banner", "sms"],
            },
            {
                "guide_id": "photo-recipe",
                **reference(
                    PHOTO_RECIPE_KEY,
                    photo_recipe_bytes,
                    MARKDOWN_CONTENT_TYPE,
                    version="v1",
                ),
                "required": True,
                "applies_to": ["email", "onsite_banner"],
            },
        ],
        "assets": [
            {
                "asset_id": "home-hero",
                **reference(
                    HOME_HERO_KEY,
                    home_hero_bytes,
                    "image/jpeg",
                    version="v1",
                ),
                "role": "hero",
                "active": True,
                "advertising_use": "demo_only",
                "alt_text": "StayLoop 대표 호텔 이미지",
                "tags": ["hotel", "hero"],
            }
        ],
        "catalogs": [],
    }
    if manifest_patch:
        manifest.update(manifest_patch)
    manifest_bytes = _json_bytes(manifest)
    pointer_bytes = _json_bytes(
        {
            "schema_version": "loopad.brand-context-pointer.v1",
            "project_id": PROJECT_ID,
            "context_version": "v1",
            "manifest_key": MANIFEST_KEY,
            "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        }
    )
    return {
        CURRENT_KEY: (pointer_bytes, JSON_CONTENT_TYPE),
        MANIFEST_KEY: (manifest_bytes, JSON_CONTENT_TYPE),
        BRAND_KIT_KEY: (brand_kit_bytes, JSON_CONTENT_TYPE),
        BRAND_VOICE_KEY: (brand_voice_bytes, MARKDOWN_CONTENT_TYPE),
        PHOTO_RECIPE_KEY: (photo_recipe_bytes, MARKDOWN_CONTENT_TYPE),
        HOME_HERO_KEY: (home_hero_bytes, "image/jpeg"),
    }


def reference(
    key: str,
    body: bytes,
    content_type: str,
    **values: Any,
) -> dict[str, Any]:
    return {
        **values,
        "s3_key": key,
        "sha256": hashlib.sha256(body).hexdigest(),
        "content_type": content_type,
        "byte_size": len(body),
    }


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
