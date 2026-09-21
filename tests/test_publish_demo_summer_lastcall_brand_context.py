from __future__ import annotations

import hashlib
import io
import json
from typing import Any

import pytest
from botocore.exceptions import ClientError

from scripts.publish_demo_summer_lastcall_brand_context import (
    BASE_CATALOG_ID,
    BASE_CATALOG_VERSION,
    BASE_OFFER_SET_ID,
    JSON_CONTENT_TYPE,
    POINTER_KEY,
    PROJECT_ID,
    TARGET_CATALOG_ID,
    TARGET_CATALOG_KEY,
    TARGET_CATALOG_VERSION,
    TARGET_HOTEL_IDS,
    TARGET_MANIFEST_KEY,
    TARGET_OFFER_SET_ID,
    build_publication_bundle,
    publish_bundle,
    verify_published_bundle,
)


BUCKET = "test-bucket"
BASE_HOTEL_IDS = (
    "jeju-ocean-breeze-006",
    "jeju-aewol-sunset-007",
    "jeju-halla-garden-008",
    "jeju-seogwipo-cliff-009",
    "okinawa-naha-terrace-017",
    "okinawa-chatan-sunset-018",
    "okinawa-ishigaki-blue-019",
    "okinawa-yomitan-coast-020",
)


class FakeS3Client:
    def __init__(self, objects: dict[str, tuple[bytes, str]]) -> None:
        self.objects = dict(objects)
        self.put_calls: list[dict[str, Any]] = []

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        assert Bucket == BUCKET or Bucket == "validation"
        value = self.objects.get(Key)
        if value is None:
            raise _client_error("NoSuchKey", 404, "GetObject")
        body, content_type = value
        return {
            "Body": io.BytesIO(body),
            "ContentType": content_type,
            "ContentLength": len(body),
            "ETag": f'"{hashlib.md5(body).hexdigest()}"',
        }

    def put_object(self, **values: Any) -> dict[str, Any]:
        assert values["Bucket"] == BUCKET
        key = values["Key"]
        body = values["Body"]
        if values.get("IfNoneMatch") == "*" and key in self.objects:
            raise _client_error("PreconditionFailed", 412, "PutObject")
        if_match = values.get("IfMatch")
        if if_match is not None:
            current = self.objects.get(key)
            assert current is not None
            current_etag = f'"{hashlib.md5(current[0]).hexdigest()}"'
            if current_etag != if_match:
                raise _client_error("PreconditionFailed", 412, "PutObject")
        self.objects[key] = (body, values["ContentType"])
        self.put_calls.append(dict(values))
        return {"ETag": f'"{hashlib.md5(body).hexdigest()}"'}


def test_publication_builds_both_offer_sets_and_four_discounted_hotels() -> None:
    s3_client = FakeS3Client(source_objects())

    bundle = build_publication_bundle(s3_client, bucket_name=BUCKET)

    assert [item["offer_set_id"] for item in bundle.target_manifest["offer_sets"]] == [
        BASE_OFFER_SET_ID,
        TARGET_OFFER_SET_ID,
    ]
    assert [item["catalog_id"] for item in bundle.target_manifest["catalogs"]] == [
        BASE_CATALOG_ID,
        TARGET_CATALOG_ID,
    ]
    assert len(json.loads(bundle.base_catalog_bytes)["hotels"]) == 8
    hotels = bundle.target_catalog["hotels"]
    assert [hotel["hotel_id"] for hotel in hotels] == list(TARGET_HOTEL_IDS)
    assert [hotel["original_price_per_night"] for hotel in hotels] == [
        342000,
        264000,
        286000,
        362000,
    ]
    assert [hotel["promotion_price_per_night"] for hotel in hotels] == [
        278000,
        214000,
        232000,
        318000,
    ]
    assert [hotel["sale_price_per_night"] for hotel in hotels] == [
        250200,
        192600,
        208800,
        286200,
    ]
    assert [hotel["discount_rate_percent"] for hotel in hotels] == [
        19,
        19,
        19,
        12,
    ]
    assert all(
        hotel["additional_discount_rate_percent"] == 10
        for hotel in hotels
    )
    assert bundle.target_pointer["manifest_sha256"] == hashlib.sha256(
        bundle.target_manifest_bytes
    ).hexdigest()


def test_publication_writes_pointer_last_and_loader_resolves_target() -> None:
    s3_client = FakeS3Client(source_objects())
    bundle = build_publication_bundle(s3_client, bucket_name=BUCKET)

    assert publish_bundle(s3_client, bucket_name=BUCKET, bundle=bundle)

    assert [call["Key"] for call in s3_client.put_calls] == [
        TARGET_CATALOG_KEY,
        TARGET_MANIFEST_KEY,
        POINTER_KEY,
    ]
    assert s3_client.put_calls[-1]["IfMatch"] == bundle.source_pointer_etag
    assert verify_published_bundle(s3_client, bucket_name=BUCKET) == {
        "context_version": "v4",
        "manifest_sha256": hashlib.sha256(
            bundle.target_manifest_bytes
        ).hexdigest(),
        "offer_set_id": TARGET_OFFER_SET_ID,
        "catalog_id": TARGET_CATALOG_ID,
        "catalog_version": TARGET_CATALOG_VERSION,
        "offer_count": 4,
    }


def test_publication_is_idempotent_for_existing_immutable_objects() -> None:
    s3_client = FakeS3Client(source_objects())
    bundle = build_publication_bundle(s3_client, bucket_name=BUCKET)
    publish_bundle(s3_client, bucket_name=BUCKET, bundle=bundle)
    first_put_count = len(s3_client.put_calls)

    second_bundle = build_publication_bundle(s3_client, bucket_name=BUCKET)

    assert not publish_bundle(
        s3_client,
        bucket_name=BUCKET,
        bundle=second_bundle,
    )
    assert len(s3_client.put_calls) == first_put_count


def test_publication_recovers_when_valid_target_exists_before_pointer_switch() -> None:
    objects = source_objects()
    initial_client = FakeS3Client(objects)
    generated = build_publication_bundle(initial_client, bucket_name=BUCKET)

    existing_catalog = json.loads(generated.target_catalog_bytes)
    existing_catalog["publication_note"] = "uploaded before pointer activation"
    existing_catalog_bytes = _json_bytes(existing_catalog)
    existing_manifest = json.loads(generated.target_manifest_bytes)
    existing_manifest["publication_note"] = "uploaded before pointer activation"
    target_reference = next(
        reference
        for reference in existing_manifest["catalogs"]
        if reference["catalog_id"] == TARGET_CATALOG_ID
    )
    target_reference["sha256"] = hashlib.sha256(existing_catalog_bytes).hexdigest()
    target_reference["byte_size"] = len(existing_catalog_bytes)
    existing_manifest_bytes = _json_bytes(existing_manifest)
    objects[TARGET_CATALOG_KEY] = (existing_catalog_bytes, JSON_CONTENT_TYPE)
    objects[TARGET_MANIFEST_KEY] = (existing_manifest_bytes, JSON_CONTENT_TYPE)
    s3_client = FakeS3Client(objects)

    recovered = build_publication_bundle(s3_client, bucket_name=BUCKET)

    assert recovered.target_catalog_bytes == existing_catalog_bytes
    assert recovered.target_manifest_bytes == existing_manifest_bytes
    assert publish_bundle(s3_client, bucket_name=BUCKET, bundle=recovered)
    assert [call["Key"] for call in s3_client.put_calls] == [POINTER_KEY]
    assert verify_published_bundle(s3_client, bucket_name=BUCKET)[
        "manifest_sha256"
    ] == hashlib.sha256(existing_manifest_bytes).hexdigest()


def test_publication_rejects_missing_target_hotel() -> None:
    objects = source_objects()
    pointer = json.loads(objects[POINTER_KEY][0])
    manifest_key = pointer["manifest_key"]
    manifest = json.loads(objects[manifest_key][0])
    base_key = manifest["catalogs"][0]["s3_key"]
    base_catalog = json.loads(objects[base_key][0])
    base_catalog["hotels"] = [
        hotel
        for hotel in base_catalog["hotels"]
        if hotel["hotel_id"] != TARGET_HOTEL_IDS[-1]
    ]
    base_bytes = _json_bytes(base_catalog)
    objects[base_key] = (base_bytes, JSON_CONTENT_TYPE)
    manifest["catalogs"][0]["sha256"] = hashlib.sha256(base_bytes).hexdigest()
    manifest["catalogs"][0]["byte_size"] = len(base_bytes)
    manifest_bytes = _json_bytes(manifest)
    objects[manifest_key] = (manifest_bytes, JSON_CONTENT_TYPE)
    pointer["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    objects[POINTER_KEY] = (_json_bytes(pointer), JSON_CONTENT_TYPE)

    with pytest.raises(ValueError, match="missing target hotel"):
        build_publication_bundle(FakeS3Client(objects), bucket_name=BUCKET)


def source_objects() -> dict[str, tuple[bytes, str]]:
    base_catalog_key = (
        "brand-context/demo_project/catalogs/"
        "black-friday-hotels/v2/catalog.json"
    )
    base_catalog_bytes = _json_bytes(
        {
            "schema_version": "stayloop.promotion-price-catalog.v1",
            "project_id": PROJECT_ID,
            "catalog_id": BASE_CATALOG_ID,
            "catalog_version": BASE_CATALOG_VERSION,
            "promotion_id": "jeju-okinawa-black-friday-demo",
            "promotion_label": "제주·오키나와 여름 특가",
            "source": "stayloop-demo-frontend-fixture",
            "source_file": "src/data/hotels.ts",
            "trust_scope": "demo_presentation_only",
            "price_basis": "one_room_one_night",
            "taxes_and_fees_included": True,
            "currency": "KRW",
            "hotels": [
                _hotel(
                    "jeju-ocean-breeze-006",
                    "Jeju Ocean Breeze Resort",
                    "jeju",
                    342000,
                    278000,
                    19,
                ),
                _hotel(
                    "jeju-aewol-sunset-007",
                    "Aewol Sunset Villa",
                    "jeju",
                    264000,
                    214000,
                    19,
                ),
                _hotel(
                    "jeju-halla-garden-008",
                    "Halla Garden Hotel",
                    "jeju",
                    198000,
                    154000,
                    22,
                ),
                _hotel(
                    "jeju-seogwipo-cliff-009",
                    "Seogwipo Cliff Stay",
                    "jeju",
                    248000,
                    196000,
                    21,
                ),
                _hotel(
                    "okinawa-naha-terrace-017",
                    "Naha Island Terrace",
                    "okinawa",
                    286000,
                    232000,
                    19,
                ),
                _hotel(
                    "okinawa-chatan-sunset-018",
                    "Chatan Sunset Bay Resort",
                    "okinawa",
                    362000,
                    318000,
                    12,
                ),
                _hotel(
                    "okinawa-ishigaki-blue-019",
                    "Ishigaki Blue Sky Resort",
                    "okinawa",
                    246000,
                    196000,
                    20,
                ),
                _hotel(
                    "okinawa-yomitan-coast-020",
                    "Yomitan Coast Hotel",
                    "okinawa",
                    218000,
                    174000,
                    20,
                ),
            ],
        }
    )
    source_manifest_key = (
        "brand-context/demo_project/manifests/v2/manifest.json"
    )
    source_manifest_bytes = _json_bytes(
        {
            "schema_version": "loopad.brand-context-manifest.v1",
            "project_id": PROJECT_ID,
            "brand_id": "stayloop",
            "context_version": "v2",
            "created_at": "2026-07-18T00:00:00Z",
            "brand_kit": _reference(
                "brand-context/demo_project/brand-kits/v2/brand-kit.json",
                b"brand-kit",
            ),
            "guidelines": [],
            "assets": [
                _hotel_asset(hotel_id)
                for hotel_id in BASE_HOTEL_IDS
            ],
            "catalogs": [
                {
                    "catalog_id": BASE_CATALOG_ID,
                    "version": BASE_CATALOG_VERSION,
                    "s3_key": base_catalog_key,
                    "sha256": hashlib.sha256(base_catalog_bytes).hexdigest(),
                    "content_type": JSON_CONTENT_TYPE,
                    "byte_size": len(base_catalog_bytes),
                    "required": True,
                    "applies_to": ["email", "onsite_banner", "sms"],
                    "claim_scope": ["hotel_name", "sale_price"],
                    "trust_scope": "demo_presentation_only",
                }
            ],
        }
    )
    pointer_bytes = _json_bytes(
        {
            "schema_version": "loopad.brand-context-pointer.v1",
            "project_id": PROJECT_ID,
            "context_version": "v2",
            "manifest_key": source_manifest_key,
            "manifest_sha256": hashlib.sha256(source_manifest_bytes).hexdigest(),
        }
    )
    return {
        POINTER_KEY: (pointer_bytes, JSON_CONTENT_TYPE),
        source_manifest_key: (source_manifest_bytes, JSON_CONTENT_TYPE),
        base_catalog_key: (base_catalog_bytes, JSON_CONTENT_TYPE),
    }


def _hotel(
    hotel_id: str,
    hotel_name: str,
    destination_id: str,
    regular_price: int,
    promotion_price: int,
    discount_rate: int,
) -> dict[str, Any]:
    return {
        "hotel_id": hotel_id,
        "hotel_name": hotel_name,
        "destination_id": destination_id,
        "currency": "KRW",
        "sale_price_per_night": promotion_price,
        "original_price_per_night": regular_price,
        "discount_amount": regular_price - promotion_price,
        "discount_rate_percent": discount_rate,
    }


def _reference(key: str, body: bytes) -> dict[str, Any]:
    return {
        "version": "v2",
        "s3_key": key,
        "sha256": hashlib.sha256(body).hexdigest(),
        "content_type": JSON_CONTENT_TYPE,
        "byte_size": len(body),
    }


def _hotel_asset(hotel_id: str) -> dict[str, Any]:
    body = f"image:{hotel_id}".encode()
    return {
        "asset_id": f"{hotel_id}-primary",
        "version": "v2",
        "s3_key": f"brand-context/demo_project/assets/v2/{hotel_id}.webp",
        "sha256": hashlib.sha256(body).hexdigest(),
        "content_type": "image/webp",
        "byte_size": len(body),
        "active": True,
        "advertising_use": "approved",
        "role": "hotel",
        "frontend_path": f"/images/hotels/{hotel_id}.webp",
        "entity_refs": [
            {
                "type": "hotel",
                "id": hotel_id,
                "usage": "primary",
            }
        ],
    }


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _client_error(code: str, status: int, operation: str) -> ClientError:
    return ClientError(
        {
            "Error": {"Code": code, "Message": code},
            "ResponseMetadata": {"HTTPStatusCode": status},
        },
        operation,
    )
