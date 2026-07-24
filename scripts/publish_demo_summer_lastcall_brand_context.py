from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import boto3
from botocore.exceptions import ClientError

from app.generation.brand_context_s3 import S3BrandContextLoader


PROJECT_ID = "demo_project"
BASE_PREFIX = "brand-context/"
POINTER_KEY = f"{BASE_PREFIX}{PROJECT_ID}/current.json"
BASE_CATALOG_ID = "black-friday-hotels"
BASE_CATALOG_VERSION = "v2"
BASE_OFFER_SET_ID = "summer-base"
TARGET_CONTEXT_VERSION = "v4"
TARGET_CATALOG_ID = "black-friday-hotels-lastcall"
TARGET_CATALOG_VERSION = "v4"
TARGET_OFFER_SET_ID = "summer-lastcall"
TARGET_DEAL_CODE = "summer-lastcall"
TARGET_CREATED_AT = "2026-07-24T00:00:00Z"
TARGET_CATALOG_KEY = (
    f"{BASE_PREFIX}{PROJECT_ID}/catalogs/"
    f"{TARGET_CATALOG_ID}/{TARGET_CATALOG_VERSION}/catalog.json"
)
TARGET_MANIFEST_KEY = (
    f"{BASE_PREFIX}{PROJECT_ID}/manifests/"
    f"{TARGET_CONTEXT_VERSION}/manifest.json"
)
JSON_CONTENT_TYPE = "application/json; charset=utf-8"
SHOP_ORIGIN = "https://demo-shoppingmall.dev.loop-ad.org"
TARGET_LANDING_URL = f"{SHOP_ORIGIN}/search?deal={TARGET_DEAL_CODE}"
TARGET_HOTEL_IDS = (
    "jeju-ocean-breeze-006",
    "jeju-aewol-sunset-007",
    "okinawa-naha-terrace-017",
    "okinawa-chatan-sunset-018",
)


@dataclass(frozen=True)
class PublicationBundle:
    source_pointer_etag: str
    source_pointer_bytes: bytes
    base_catalog_key: str
    base_catalog_bytes: bytes
    target_catalog: Mapping[str, Any]
    target_catalog_bytes: bytes
    target_manifest: Mapping[str, Any]
    target_manifest_bytes: bytes
    target_pointer: Mapping[str, Any]
    target_pointer_bytes: bytes


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_hex(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def build_lastcall_catalog(base_catalog: Mapping[str, Any]) -> dict[str, Any]:
    _require_equal(base_catalog, "project_id", PROJECT_ID)
    _require_equal(base_catalog, "catalog_id", BASE_CATALOG_ID)
    _require_equal(base_catalog, "catalog_version", BASE_CATALOG_VERSION)

    source_hotels = _mapping_list(base_catalog.get("hotels"))
    hotels_by_id = {
        _required_text(hotel.get("hotel_id"), "hotel.hotel_id"): hotel
        for hotel in source_hotels
    }
    if len(hotels_by_id) != len(source_hotels):
        raise ValueError("base catalog contains duplicate hotel_id values")

    target_hotels: list[dict[str, Any]] = []
    for hotel_id in TARGET_HOTEL_IDS:
        source = hotels_by_id.get(hotel_id)
        if source is None:
            raise ValueError(f"base catalog is missing target hotel: {hotel_id}")
        regular_price = _required_positive_int(
            source.get("original_price_per_night"),
            f"{hotel_id}.original_price_per_night",
        )
        promotion_price = _required_positive_int(
            source.get("sale_price_per_night"),
            f"{hotel_id}.sale_price_per_night",
        )
        base_discount_rate = _required_percentage(
            source.get("discount_rate_percent"),
            f"{hotel_id}.discount_rate_percent",
        )
        final_price = (promotion_price * 9 + 5) // 10
        target_hotels.append(
            {
                "hotel_id": hotel_id,
                "hotel_name": _required_text(
                    source.get("hotel_name"),
                    f"{hotel_id}.hotel_name",
                ),
                "destination_id": _required_text(
                    source.get("destination_id"),
                    f"{hotel_id}.destination_id",
                ),
                "currency": _required_text(
                    source.get("currency"),
                    f"{hotel_id}.currency",
                ),
                "sale_price_per_night": final_price,
                "original_price_per_night": regular_price,
                "promotion_price_per_night": promotion_price,
                "discount_amount": regular_price - promotion_price,
                "discount_rate_percent": base_discount_rate,
                "additional_discount_rate_percent": 10,
                "destination_url": (
                    f"{SHOP_ORIGIN}/hotel/{hotel_id}?deal={TARGET_DEAL_CODE}"
                ),
            }
        )

    return {
        "schema_version": "stayloop.promotion-price-catalog.v1",
        "project_id": PROJECT_ID,
        "catalog_id": TARGET_CATALOG_ID,
        "catalog_version": TARGET_CATALOG_VERSION,
        "offer_set_id": TARGET_OFFER_SET_ID,
        "promotion_id": base_catalog.get("promotion_id"),
        "promotion_label": "제주·오키나와 여름 막바지 10% 추가 할인",
        "source": base_catalog.get("source"),
        "source_file": base_catalog.get("source_file"),
        "source_catalog": {
            "catalog_id": BASE_CATALOG_ID,
            "catalog_version": BASE_CATALOG_VERSION,
        },
        "trust_scope": base_catalog.get("trust_scope"),
        "price_basis": base_catalog.get("price_basis"),
        "taxes_and_fees_included": base_catalog.get("taxes_and_fees_included"),
        "currency": base_catalog.get("currency"),
        "deal_code": TARGET_DEAL_CODE,
        "landing_url": TARGET_LANDING_URL,
        "summary": {
            "hotel_count": len(target_hotels),
            "minimum_sale_price_per_night": min(
                hotel["sale_price_per_night"] for hotel in target_hotels
            ),
            "maximum_discount_rate_percent": 10,
        },
        "usage_rules": [
            (
                "Advertising copy may use these values literally for the "
                "StayLoop demo presentation."
            ),
            "Do not alter, estimate, or recalculate a listed price.",
            (
                "All amounts are Korean won per room per night with taxes "
                "and fees included."
            ),
            (
                "The original price is the regular price, the promotion "
                "price is the snapshotted base offer, and the sale price is "
                "the final summer-lastcall price."
            ),
        ],
        "hotels": target_hotels,
    }


def build_target_manifest(
    source_manifest: Mapping[str, Any],
    *,
    target_catalog_bytes: bytes,
) -> dict[str, Any]:
    _require_equal(source_manifest, "project_id", PROJECT_ID)
    base_reference = _find_base_catalog_reference(source_manifest)
    base_reference["offer_set_id"] = BASE_OFFER_SET_ID
    base_reference["required"] = False

    target_reference = {
        "catalog_id": TARGET_CATALOG_ID,
        "offer_set_id": TARGET_OFFER_SET_ID,
        "version": TARGET_CATALOG_VERSION,
        "s3_key": TARGET_CATALOG_KEY,
        "sha256": sha256_hex(target_catalog_bytes),
        "content_type": JSON_CONTENT_TYPE,
        "byte_size": len(target_catalog_bytes),
        "required": True,
        "applies_to": list(base_reference.get("applies_to") or ["email"]),
        "claim_scope": list(base_reference.get("claim_scope") or []),
        "trust_scope": base_reference.get("trust_scope"),
    }

    manifest = copy.deepcopy(dict(source_manifest))
    manifest["context_version"] = TARGET_CONTEXT_VERSION
    manifest["created_at"] = TARGET_CREATED_AT
    manifest["offer_sets"] = [
        {
            "offer_set_id": BASE_OFFER_SET_ID,
            "catalog_id": BASE_CATALOG_ID,
            "catalog_version": BASE_CATALOG_VERSION,
            "landing_url": f"{SHOP_ORIGIN}/search",
        },
        {
            "offer_set_id": TARGET_OFFER_SET_ID,
            "catalog_id": TARGET_CATALOG_ID,
            "catalog_version": TARGET_CATALOG_VERSION,
            "landing_url": TARGET_LANDING_URL,
        },
    ]
    manifest["catalogs"] = [base_reference, target_reference]
    return manifest


def build_publication_bundle(s3_client: Any, *, bucket_name: str) -> PublicationBundle:
    pointer_bytes, pointer_content_type, pointer_etag = _read_object(
        s3_client,
        bucket_name=bucket_name,
        key=POINTER_KEY,
    )
    _require_json_content_type(pointer_content_type, POINTER_KEY)
    source_pointer = _json_object(pointer_bytes, POINTER_KEY)
    _require_equal(source_pointer, "project_id", PROJECT_ID)
    source_manifest_key = _required_text(
        source_pointer.get("manifest_key"),
        "pointer.manifest_key",
    )
    source_manifest_bytes, manifest_content_type, _ = _read_object(
        s3_client,
        bucket_name=bucket_name,
        key=source_manifest_key,
    )
    _require_json_content_type(manifest_content_type, source_manifest_key)
    expected_manifest_sha = _required_text(
        source_pointer.get("manifest_sha256"),
        "pointer.manifest_sha256",
    )
    if sha256_hex(source_manifest_bytes) != expected_manifest_sha:
        raise ValueError("source manifest checksum does not match current pointer")
    source_manifest = _json_object(source_manifest_bytes, source_manifest_key)

    base_reference = _find_base_catalog_reference(source_manifest)
    base_catalog_key = _required_text(
        base_reference.get("s3_key"),
        "base catalog s3_key",
    )
    base_catalog_bytes, base_content_type, _ = _read_object(
        s3_client,
        bucket_name=bucket_name,
        key=base_catalog_key,
    )
    _require_json_content_type(base_content_type, base_catalog_key)
    if len(base_catalog_bytes) != base_reference.get("byte_size"):
        raise ValueError("base catalog byte size does not match manifest")
    if sha256_hex(base_catalog_bytes) != base_reference.get("sha256"):
        raise ValueError("base catalog checksum does not match manifest")
    base_catalog = _json_object(base_catalog_bytes, base_catalog_key)

    target_catalog = build_lastcall_catalog(base_catalog)
    target_catalog_bytes = canonical_json_bytes(target_catalog)
    target_manifest = build_target_manifest(
        source_manifest,
        target_catalog_bytes=target_catalog_bytes,
    )
    target_manifest_bytes = canonical_json_bytes(target_manifest)
    existing_target = _load_compatible_existing_target(
        s3_client,
        bucket_name=bucket_name,
        expected_catalog=target_catalog,
    )
    if existing_target is not None:
        (
            target_catalog,
            target_catalog_bytes,
            target_manifest,
            target_manifest_bytes,
        ) = existing_target
    target_pointer = {
        "schema_version": "loopad.brand-context-pointer.v1",
        "project_id": PROJECT_ID,
        "context_version": TARGET_CONTEXT_VERSION,
        "manifest_key": TARGET_MANIFEST_KEY,
        "manifest_sha256": sha256_hex(target_manifest_bytes),
    }
    target_pointer_bytes = canonical_json_bytes(target_pointer)
    bundle = PublicationBundle(
        source_pointer_etag=pointer_etag,
        source_pointer_bytes=pointer_bytes,
        base_catalog_key=base_catalog_key,
        base_catalog_bytes=base_catalog_bytes,
        target_catalog=target_catalog,
        target_catalog_bytes=target_catalog_bytes,
        target_manifest=target_manifest,
        target_manifest_bytes=target_manifest_bytes,
        target_pointer=target_pointer,
        target_pointer_bytes=target_pointer_bytes,
    )
    validate_publication_bundle(bundle)
    return bundle


def validate_publication_bundle(bundle: PublicationBundle) -> None:
    objects = {
        POINTER_KEY: (bundle.target_pointer_bytes, JSON_CONTENT_TYPE),
        TARGET_MANIFEST_KEY: (bundle.target_manifest_bytes, JSON_CONTENT_TYPE),
        bundle.base_catalog_key: (bundle.base_catalog_bytes, JSON_CONTENT_TYPE),
        TARGET_CATALOG_KEY: (bundle.target_catalog_bytes, JSON_CONTENT_TYPE),
    }
    loader = S3BrandContextLoader(
        bucket_name="validation",
        base_prefix=BASE_PREFIX,
        s3_client=_MemoryS3Client(objects),
    )
    snapshot = loader.resolve_snapshot(project_id=PROJECT_ID)
    if snapshot is None:
        raise ValueError("target pointer did not resolve")
    base_catalog = loader.load_offer_catalog(
        project_id=PROJECT_ID,
        snapshot=snapshot,
        offer_set_id=BASE_OFFER_SET_ID,
    )
    target_catalog = loader.load_offer_catalog(
        project_id=PROJECT_ID,
        snapshot=snapshot,
        offer_set_id=TARGET_OFFER_SET_ID,
    )
    if base_catalog is None or base_catalog.get("catalog_id") != BASE_CATALOG_ID:
        raise ValueError("summer-base did not resolve to the base catalog")
    if (
        target_catalog is None
        or target_catalog.get("catalog_id") != TARGET_CATALOG_ID
        or target_catalog.get("catalog_version") != TARGET_CATALOG_VERSION
    ):
        raise ValueError("summer-lastcall did not resolve to the target catalog")
    offer_ids = [hotel.get("offer_id") for hotel in target_catalog["hotels"]]
    if offer_ids != list(TARGET_HOTEL_IDS):
        raise ValueError("target catalog hotel order or membership is invalid")


def publish_bundle(
    s3_client: Any,
    *,
    bucket_name: str,
    bundle: PublicationBundle,
) -> bool:
    _put_immutable(
        s3_client,
        bucket_name=bucket_name,
        key=TARGET_CATALOG_KEY,
        body=bundle.target_catalog_bytes,
    )
    _put_immutable(
        s3_client,
        bucket_name=bucket_name,
        key=TARGET_MANIFEST_KEY,
        body=bundle.target_manifest_bytes,
    )
    if bundle.source_pointer_bytes == bundle.target_pointer_bytes:
        return False
    s3_client.put_object(
        Bucket=bucket_name,
        Key=POINTER_KEY,
        Body=bundle.target_pointer_bytes,
        ContentType=JSON_CONTENT_TYPE,
        IfMatch=bundle.source_pointer_etag,
    )
    return True


def verify_published_bundle(s3_client: Any, *, bucket_name: str) -> dict[str, Any]:
    loader = S3BrandContextLoader(
        bucket_name=bucket_name,
        base_prefix=BASE_PREFIX,
        s3_client=s3_client,
    )
    snapshot = loader.resolve_snapshot(project_id=PROJECT_ID)
    if snapshot is None:
        raise ValueError("published brand context pointer was not found")
    catalog = loader.load_offer_catalog(
        project_id=PROJECT_ID,
        snapshot=snapshot,
        offer_set_id=TARGET_OFFER_SET_ID,
    )
    if catalog is None:
        raise ValueError("published summer-lastcall catalog was not found")
    return {
        "context_version": snapshot.context_version,
        "manifest_sha256": snapshot.manifest_sha256,
        "offer_set_id": catalog.get("offer_set_id"),
        "catalog_id": catalog.get("catalog_id"),
        "catalog_version": catalog.get("catalog_version"),
        "offer_count": len(catalog.get("hotels") or []),
    }


class _MemoryS3Client:
    def __init__(self, objects: Mapping[str, tuple[bytes, str]]) -> None:
        self._objects = dict(objects)

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        del Bucket
        body, content_type = self._objects[Key]
        return {
            "Body": io.BytesIO(body),
            "ContentType": content_type,
            "ContentLength": len(body),
        }


def _put_immutable(
    s3_client: Any,
    *,
    bucket_name: str,
    key: str,
    body: bytes,
) -> None:
    try:
        s3_client.put_object(
            Bucket=bucket_name,
            Key=key,
            Body=body,
            ContentType=JSON_CONTENT_TYPE,
            IfNoneMatch="*",
        )
        return
    except ClientError as exc:
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if status != 412 and code not in {"PreconditionFailed", "412"}:
            raise

    existing, content_type, _ = _read_object(
        s3_client,
        bucket_name=bucket_name,
        key=key,
    )
    _require_json_content_type(content_type, key)
    if existing != body:
        raise ValueError(f"immutable S3 object already exists with different bytes: {key}")


def _load_compatible_existing_target(
    s3_client: Any,
    *,
    bucket_name: str,
    expected_catalog: Mapping[str, Any],
) -> tuple[
    Mapping[str, Any],
    bytes,
    Mapping[str, Any],
    bytes,
] | None:
    catalog_object = _read_optional_object(
        s3_client,
        bucket_name=bucket_name,
        key=TARGET_CATALOG_KEY,
    )
    manifest_object = _read_optional_object(
        s3_client,
        bucket_name=bucket_name,
        key=TARGET_MANIFEST_KEY,
    )
    if catalog_object is None and manifest_object is None:
        return None
    if catalog_object is None or manifest_object is None:
        raise ValueError(
            "summer-lastcall target publication is incomplete; "
            "catalog and manifest must either both exist or both be absent"
        )

    catalog_bytes, catalog_content_type, _ = catalog_object
    manifest_bytes, manifest_content_type, _ = manifest_object
    _require_json_content_type(catalog_content_type, TARGET_CATALOG_KEY)
    _require_json_content_type(manifest_content_type, TARGET_MANIFEST_KEY)
    catalog = _json_object(catalog_bytes, TARGET_CATALOG_KEY)
    manifest = _json_object(manifest_bytes, TARGET_MANIFEST_KEY)

    if _target_catalog_execution_view(catalog) != _target_catalog_execution_view(
        expected_catalog
    ):
        raise ValueError(
            "existing summer-lastcall catalog does not match the expected offer set"
        )
    _validate_existing_target_manifest(
        manifest,
        catalog_bytes=catalog_bytes,
    )
    return catalog, catalog_bytes, manifest, manifest_bytes


def _validate_existing_target_manifest(
    manifest: Mapping[str, Any],
    *,
    catalog_bytes: bytes,
) -> None:
    _require_equal(manifest, "project_id", PROJECT_ID)
    _require_equal(manifest, "context_version", TARGET_CONTEXT_VERSION)
    references = [
        reference
        for reference in _mapping_list(manifest.get("catalogs"))
        if reference.get("catalog_id") == TARGET_CATALOG_ID
        and reference.get("version") == TARGET_CATALOG_VERSION
    ]
    if len(references) != 1:
        raise ValueError("target manifest must contain exactly one lastcall catalog")
    reference = references[0]
    _require_equal(reference, "s3_key", TARGET_CATALOG_KEY)
    _require_equal(reference, "sha256", sha256_hex(catalog_bytes))
    _require_equal(reference, "byte_size", len(catalog_bytes))
    _require_json_content_type(
        _required_text(reference.get("content_type"), "target catalog content_type"),
        TARGET_CATALOG_KEY,
    )

    offer_sets = [
        offer_set
        for offer_set in _mapping_list(manifest.get("offer_sets"))
        if offer_set.get("offer_set_id") == TARGET_OFFER_SET_ID
    ]
    if len(offer_sets) != 1:
        raise ValueError("target manifest must contain exactly one lastcall offer set")
    offer_set = offer_sets[0]
    _require_equal(offer_set, "catalog_id", TARGET_CATALOG_ID)
    _require_equal(offer_set, "catalog_version", TARGET_CATALOG_VERSION)
    _require_equal(offer_set, "landing_url", TARGET_LANDING_URL)


def _target_catalog_execution_view(catalog: Mapping[str, Any]) -> dict[str, Any]:
    hotels = _mapping_list(catalog.get("hotels"))
    return {
        "schema_version": catalog.get("schema_version"),
        "project_id": catalog.get("project_id"),
        "catalog_id": catalog.get("catalog_id"),
        "catalog_version": catalog.get("catalog_version"),
        "offer_set_id": catalog.get("offer_set_id"),
        "currency": catalog.get("currency"),
        "deal_code": catalog.get("deal_code"),
        "landing_url": catalog.get("landing_url"),
        "hotels": [
            {
                key: hotel.get(key)
                for key in (
                    "hotel_id",
                    "hotel_name",
                    "destination_id",
                    "currency",
                    "original_price_per_night",
                    "sale_price_per_night",
                    "discount_amount",
                    "discount_rate_percent",
                    "destination_url",
                )
            }
            for hotel in hotels
        ],
    }


def _read_optional_object(
    s3_client: Any,
    *,
    bucket_name: str,
    key: str,
) -> tuple[bytes, str, str] | None:
    try:
        return _read_object(
            s3_client,
            bucket_name=bucket_name,
            key=key,
        )
    except ClientError as exc:
        status = int(exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode", 0))
        code = str(exc.response.get("Error", {}).get("Code", ""))
        if status == 404 or code in {"NoSuchKey", "NotFound", "404"}:
            return None
        raise


def _read_object(
    s3_client: Any,
    *,
    bucket_name: str,
    key: str,
) -> tuple[bytes, str, str]:
    response = s3_client.get_object(Bucket=bucket_name, Key=key)
    body = response["Body"].read()
    if not isinstance(body, bytes):
        raise ValueError(f"S3 object body was not bytes: {key}")
    content_type = str(response.get("ContentType") or "")
    etag = str(response.get("ETag") or "")
    return body, content_type, etag


def _find_base_catalog_reference(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    matches = [
        copy.deepcopy(dict(reference))
        for reference in _mapping_list(manifest.get("catalogs"))
        if reference.get("catalog_id") == BASE_CATALOG_ID
        and reference.get("version") == BASE_CATALOG_VERSION
    ]
    if len(matches) != 1:
        raise ValueError("manifest must contain exactly one base v2 catalog reference")
    return matches[0]


def _mapping_list(value: object) -> list[Mapping[str, Any]]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _json_object(value: bytes, label: str) -> Mapping[str, Any]:
    parsed = json.loads(value)
    if not isinstance(parsed, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return parsed


def _required_text(value: object, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    return text


def _required_positive_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _required_percentage(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if value < 0 or value > 100:
        raise ValueError(f"{label} must be between 0 and 100")
    return value


def _require_equal(value: Mapping[str, Any], key: str, expected: object) -> None:
    if value.get(key) != expected:
        raise ValueError(f"{key} must be {expected!r}")


def _require_json_content_type(value: str, key: str) -> None:
    if value.split(";", maxsplit=1)[0].strip().lower() != "application/json":
        raise ValueError(f"S3 object content type was not JSON: {key}")


def _summary(bundle: PublicationBundle) -> dict[str, Any]:
    return {
        "project_id": PROJECT_ID,
        "context_version": TARGET_CONTEXT_VERSION,
        "catalog_key": TARGET_CATALOG_KEY,
        "catalog_sha256": sha256_hex(bundle.target_catalog_bytes),
        "manifest_key": TARGET_MANIFEST_KEY,
        "manifest_sha256": sha256_hex(bundle.target_manifest_bytes),
        "pointer_key": POINTER_KEY,
        "offer_sets": [BASE_OFFER_SET_ID, TARGET_OFFER_SET_ID],
        "offers": [
            {
                "hotel_id": hotel["hotel_id"],
                "original_price_per_night": hotel["original_price_per_night"],
                "promotion_price_per_night": hotel[
                    "promotion_price_per_night"
                ],
                "sale_price_per_night": hotel["sale_price_per_night"],
            }
            for hotel in bundle.target_catalog["hotels"]
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Publish the immutable demo summer-lastcall Brand Context bundle."
        )
    )
    parser.add_argument("--bucket", required=True)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write immutable objects and update current.json. Default is dry-run.",
    )
    args = parser.parse_args()

    s3_client = boto3.client("s3")
    bundle = build_publication_bundle(s3_client, bucket_name=args.bucket)
    print(json.dumps(_summary(bundle), ensure_ascii=False, indent=2))
    if not args.apply:
        print("dry-run: no S3 objects were changed")
        return 0

    pointer_updated = publish_bundle(
        s3_client,
        bucket_name=args.bucket,
        bundle=bundle,
    )
    verification = verify_published_bundle(s3_client, bucket_name=args.bucket)
    print(
        json.dumps(
            {
                "pointer_updated": pointer_updated,
                "verification": verification,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
