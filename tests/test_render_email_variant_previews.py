from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from scripts.render_email_variant_previews import (
    load_preview_bundle,
    render_local_previews,
)


PROJECT_ID = "demo_project"
CATALOG_ID = "black-friday-hotels"
LANDING_URL = "https://demo-shoppingmall.dev.loop-ad.org/promotions/summer"


def test_local_v2_bundle_renders_three_provider_free_email_previews(
    tmp_path: Path,
) -> None:
    bundle_root = tmp_path / "bundle"
    _write_preview_bundle(bundle_root)

    bundle = load_preview_bundle(
        bundle_root=bundle_root,
        project_id=PROJECT_ID,
        catalog_id=CATALOG_ID,
    )
    output_dir = tmp_path / "previews"
    report = render_local_previews(
        bundle=bundle,
        project_id=PROJECT_ID,
        landing_url=LANDING_URL,
        output_dir=output_dir,
    )

    assert bundle.snapshot.context_version == "v2"
    assert len(bundle.offer_catalog["hotels"]) == 8
    assert report["generation_status"] == "completed"
    assert report["candidate_count"] == 3
    assert report["catalog_hotel_count"] == 8
    assert report["external_provider_calls"] == 0
    assert [candidate["variant"] for candidate in report["candidates"]] == [
        "editorial",
        "offer_cards",
        "comparison",
    ]
    assert {
        candidate["variant"]: candidate["local_images_verified"]
        for candidate in report["candidates"]
    } == {
        "editorial": 3,
        "offer_cards": 8,
        "comparison": 4,
    }

    expected_outputs = {
        output_dir / "index.html",
        output_dir / "report.json",
        output_dir / "01-editorial.production.html",
        output_dir / "01-editorial.local.html",
        output_dir / "02-eight-offers.production.html",
        output_dir / "02-eight-offers.local.html",
        output_dir / "03-comparison.production.html",
        output_dir / "03-comparison.local.html",
    }
    assert all(path.is_file() for path in expected_outputs)
    assert len(list((output_dir / "assets").iterdir())) == 8
    for candidate in report["candidates"]:
        assert Path(candidate["canonical_output"]).is_file()
        assert Path(candidate["local_output"]).is_file()
        assert candidate["artifact_status"] == "published"
        assert candidate["image_generation_status"] == "completed"

    saved_report = json.loads((output_dir / "report.json").read_text("utf-8"))
    assert saved_report["external_provider_calls"] == 0
    assert [item["variant"] for item in saved_report["candidates"]] == [
        "editorial",
        "offer_cards",
        "comparison",
    ]


def _write_preview_bundle(bundle_root: Path) -> None:
    project_root = bundle_root / "brand-context" / PROJECT_ID
    brand_kit = _json_bytes(
        {
            "brand": {
                "name": "StayLoop",
                "category": "travel",
                "locale": "ko-KR",
            },
            "colors": {"primary": "#0F55C8"},
        }
    )
    guide = "# StayLoop email guide\n\nUse concise, trustworthy travel copy.\n".encode()
    brand_kit_ref = _write_reference(
        bundle_root,
        "brand-context/demo_project/brand-kits/v2/brand-kit.json",
        brand_kit,
        content_type="application/json; charset=utf-8",
    )
    guide_ref = _write_reference(
        bundle_root,
        "brand-context/demo_project/guidelines/v2/email.md",
        guide,
        content_type="text/markdown; charset=utf-8",
    )

    assets: list[dict[str, Any]] = []
    hotels: list[dict[str, Any]] = []
    for index in range(8):
        destination_id = "jeju" if index < 4 else "okinawa"
        hotel_id = f"{destination_id}-hotel-{index + 1:02d}"
        asset_id = f"{hotel_id}-hero"
        asset_key = f"brand-context/demo_project/assets/v2/{asset_id}.png"
        asset_ref = _write_reference(
            bundle_root,
            asset_key,
            f"fixture-image-{index}".encode(),
            content_type="image/png",
        )
        assets.append(
            {
                **asset_ref,
                "asset_id": asset_id,
                "version": "v2",
                "active": True,
                "advertising_use": "approved",
                "role": "hotel",
                "frontend_path": f"/stayloop/v2/{asset_id}.png",
                "alt_text": f"StayLoop hotel {index + 1}",
                "tags": [destination_id, "hotel"],
                "entity_refs": [
                    {
                        "type": "hotel",
                        "id": hotel_id,
                        "usage": "primary",
                    },
                    {"type": "destination", "id": destination_id},
                ],
            }
        )
        hotels.append(
            {
                "hotel_id": hotel_id,
                "hotel_name": f"StayLoop Hotel {index + 1}",
                "destination_id": destination_id,
                "currency": "KRW",
                "sale_price_per_night": 180_000 + index * 10_000,
                "original_price_per_night": 220_000 + index * 10_000,
                "discount_rate_percent": 10 + index,
            }
        )

    catalog_ref = _write_reference(
        bundle_root,
        "brand-context/demo_project/catalogs/v2/offers.json",
        _json_bytes(
            {
                "schema_version": "stayloop.promotion-price-catalog.v1",
                "project_id": PROJECT_ID,
                "catalog_id": CATALOG_ID,
                "catalog_version": "v2",
                "promotion_label": "StayLoop summer stays",
                "currency": "KRW",
                "price_basis": "one_room_one_night",
                "hotels": hotels,
            }
        ),
        content_type="application/json; charset=utf-8",
    )
    manifest_key = "brand-context/demo_project/manifests/v2/manifest.json"
    manifest = _json_bytes(
        {
            "schema_version": "loopad.brand-context-manifest.v1",
            "project_id": PROJECT_ID,
            "context_version": "v2",
            "brand_kit": {**brand_kit_ref, "version": "v2"},
            "guidelines": [
                {
                    **guide_ref,
                    "guide_id": "email-guide",
                    "version": "v2",
                    "required": True,
                    "applies_to": ["email"],
                }
            ],
            "assets": assets,
            "catalogs": [
                {
                    **catalog_ref,
                    "catalog_id": CATALOG_ID,
                    "version": "v2",
                    "required": True,
                    "applies_to": ["email"],
                }
            ],
        }
    )
    _write_object(bundle_root, manifest_key, manifest)
    _write_object(
        bundle_root,
        "brand-context/demo_project/current.json",
        _json_bytes(
            {
                "schema_version": "loopad.brand-context-pointer.v1",
                "project_id": PROJECT_ID,
                "context_version": "v2",
                "manifest_key": manifest_key,
                "manifest_sha256": hashlib.sha256(manifest).hexdigest(),
            }
        ),
    )
    assert project_root.is_dir()


def _write_reference(
    bundle_root: Path,
    key: str,
    body: bytes,
    *,
    content_type: str,
) -> dict[str, Any]:
    _write_object(bundle_root, key, body)
    return {
        "s3_key": key,
        "sha256": hashlib.sha256(body).hexdigest(),
        "byte_size": len(body),
        "content_type": content_type,
    }


def _write_object(bundle_root: Path, key: str, body: bytes) -> None:
    path = bundle_root / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)


def _json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
