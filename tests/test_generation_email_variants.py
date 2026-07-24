from app.generation.artifacts import render_email_html, source_for_channel
from app.generation.email_variants import (
    COMPARISON_VARIANT,
    EDITORIAL_VARIANT,
    OFFER_CARDS_VARIANT,
    build_email_creative_extensions,
    reusable_catalog_image_url,
)
from app.generation.prompt_builder import PromotionOfferLink
from app.generation.schemas import ContentChannel, EmailHtmlSource


LANDING_URL = "https://demo-shoppingmall.dev.loop-ad.org/promotions/black-friday"
HOTEL_IDS = (
    "jeju-ocean-breeze-006",
    "jeju-aewol-sunset-007",
    "jeju-seongsan-morning-008",
    "jeju-halla-garden-009",
    "okinawa-naha-terrace-017",
    "okinawa-chatan-sunset-018",
    "okinawa-onna-coral-019",
    "okinawa-ishigaki-sky-020",
)


def test_editorial_variant_builds_two_destination_sections() -> None:
    extensions = build_email_creative_extensions(
        option_index=1,
        landing_url=LANDING_URL,
        offer_links=_offer_links(),
        offer_catalog=_offer_catalog(),
    )

    assert extensions["variant_type"] == EDITORIAL_VARIANT
    assert extensions["template_version"] == "email.editorial.v3"
    assert len(extensions["featured_offers"]) == 2
    assert [
        offer["destination_id"] for offer in extensions["featured_offers"]
    ] == ["jeju", "okinawa"]
    assert extensions["hero_image_url"] == (
        "https://demo-shoppingmall.dev.loop-ad.org/"
        "stayloop/promotions/hotel-2.png"
    )
    assert extensions["link_targets"] == [
        {"placeholder": "{{redirect_url}}", "target_type": "promotion"}
    ]
    assert reusable_catalog_image_url(extensions) == extensions["hero_image_url"]

    content_values = _content_values(extensions)
    content_values["image_url"] = reusable_catalog_image_url(extensions)
    rendered = render_email_html(content_values)
    source = source_for_channel(
        channel=ContentChannel.EMAIL,
        content_values=content_values,
    )
    EmailHtmlSource.model_validate(source)

    assert "JEJU · OKINAWA SUMMER EDIT" in rendered
    assert "바다와 오름 사이,<br>천천히 시작하는 하루" in rendered
    assert "투명한 바다 곁에서,<br>오래 머무는 휴식" in rendered
    assert "background:#151515" in rendered
    assert "background:#222222" in rendered
    assert "background:#f8fafc" not in rendered
    assert "StayLoop Hotel 1" in rendered
    assert "StayLoop Hotel 5" in rendered
    assert extensions["hero_image_url"] in rendered
    assert "https://gen-ai.asset.dev.loop-ad.org/genai/hero.png" not in rendered
    assert "height:210px;object-fit:cover" not in rendered
    assert "{{offer_redirect_url_1}}" not in rendered


def test_offer_card_variant_builds_eight_redirect_targets_and_email_html() -> None:
    extensions = build_email_creative_extensions(
        option_index=2,
        landing_url=LANDING_URL,
        offer_links=_offer_links(),
        offer_catalog=_offer_catalog(),
    )

    assert extensions["variant_type"] == OFFER_CARDS_VARIANT
    assert extensions["template_version"] == "email.offer-cards.v3"
    assert len(extensions["offers"]) == 8
    assert len(extensions["link_targets"]) == 9
    assert extensions["link_targets"][0] == {
        "placeholder": "{{redirect_url}}",
        "target_type": "promotion",
    }
    assert extensions["link_targets"][-1] == {
        "placeholder": "{{offer_redirect_url_8}}",
        "target_type": "offer",
        "offer_id": "okinawa-ishigaki-sky-020",
        "destination_url": (
            "https://demo-shoppingmall.dev.loop-ad.org/"
            "hotel/okinawa-ishigaki-sky-020"
        ),
    }
    assert reusable_catalog_image_url(extensions) == (
        "https://demo-shoppingmall.dev.loop-ad.org/"
        "stayloop/promotions/hotel-1.png"
    )
    assert reusable_catalog_image_url(extensions) == extensions["offers"][0][
        "image_url"
    ]
    content_values = _content_values(extensions)

    rendered = render_email_html(content_values)
    source = source_for_channel(
        channel=ContentChannel.EMAIL,
        content_values=content_values,
    )
    EmailHtmlSource.model_validate(source)

    assert rendered.count("전체 프로모션 보기") == 9
    assert "제주 추천" in rendered
    assert "오키나와 추천" in rendered
    assert "278,000원" in rendered
    assert "342,000원" in rendered
    assert "https://demo-shoppingmall.dev.loop-ad.org/stayloop/" in rendered
    assert "https://gen-ai.asset.dev.loop-ad.org/genai/hero.png" not in rendered
    for index in range(1, 9):
        assert f"{{{{offer_redirect_url_{index}}}}}" in rendered
        assert f"{{{{offer_redirect_url_{index}}}}}" in source[
            "required_placeholders"
        ]


def test_comparison_variant_builds_four_rows_and_primary_redirect_only() -> None:
    extensions = build_email_creative_extensions(
        option_index=3,
        landing_url=LANDING_URL,
        offer_links=_offer_links(),
        offer_catalog=_offer_catalog(),
    )

    assert extensions["variant_type"] == COMPARISON_VARIANT
    assert extensions["template_version"] == "email.comparison.v1"
    assert len(extensions["comparison_offers"]) == 4
    assert [
        offer["destination_id"] for offer in extensions["comparison_offers"]
    ] == ["jeju", "jeju", "okinawa", "okinawa"]
    assert extensions["link_targets"] == [
        {"placeholder": "{{redirect_url}}", "target_type": "promotion"}
    ]
    assert "offers" not in extensions
    assert reusable_catalog_image_url(extensions) == (
        "https://demo-shoppingmall.dev.loop-ad.org/"
        "stayloop/promotions/hotel-1.png"
    )
    assert reusable_catalog_image_url(extensions) == extensions[
        "comparison_offers"
    ][0]["image_url"]

    content_values = _content_values(extensions)
    rendered = render_email_html(content_values)
    source = source_for_channel(
        channel=ContentChannel.EMAIL,
        content_values=content_values,
    )
    EmailHtmlSource.model_validate(source)

    assert rendered.count("PICK 0") == 4
    assert "가격과 위치를 한눈에 비교해보세요" in rendered
    assert "최대 19% 할인" in rendered
    assert "숙소 확인하기" not in rendered
    assert "{{offer_redirect_url_1}}" not in rendered
    assert source["required_placeholders"] == [
        "{{redirect_url}}",
        "{{open_pixel_url}}",
        "{{unsubscribe_url}}",
    ]


def test_offer_card_variant_preserves_catalog_deal_query_for_offer_redirects() -> None:
    landing_url = (
        "https://demo-shoppingmall.dev.loop-ad.org/"
        "search?deal=summer-lastcall"
    )
    extensions = build_email_creative_extensions(
        option_index=2,
        landing_url=landing_url,
        offer_links=_offer_links()[:4],
        offer_catalog=_offer_catalog(deal_code="summer-lastcall"),
    )

    assert extensions["link_targets"][0] == {
        "placeholder": "{{redirect_url}}",
        "target_type": "promotion",
    }
    selected_offers = extensions["offers"]
    assert isinstance(selected_offers, list)
    assert len(selected_offers) == 4
    offer_targets = extensions["link_targets"][1:]
    assert len(offer_targets) == 4
    for index, (offer, target) in enumerate(
        zip(selected_offers, offer_targets, strict=True),
        start=1,
    ):
        expected_destination_url = (
            "https://demo-shoppingmall.dev.loop-ad.org/"
            f"hotel/{offer['offer_id']}?deal=summer-lastcall"
        )
        assert offer["destination_url"] == expected_destination_url
        assert target == {
            "placeholder": f"{{{{offer_redirect_url_{index}}}}}",
            "target_type": "offer",
            "offer_id": offer["offer_id"],
            "destination_url": expected_destination_url,
        }

    content_values = _content_values(extensions)
    content_values["landing_url"] = landing_url
    rendered = render_email_html(content_values)

    # Dispatch wraps these placeholders in attributed redirect URLs. The raw
    # destination, including its deal query, remains in link_targets above.
    assert 'href="{{redirect_url}}"' in rendered
    for index in range(1, 5):
        assert f'{{{{offer_redirect_url_{index}}}}}' in rendered
    assert rendered.count("전체 프로모션 보기") == 5


def _content_values(extensions: dict[str, object]) -> dict[str, object]:
    return {
        "subject": "제주·오키나와 여름 숙소 추천",
        "preheader": "8개 숙소의 데모 특가를 확인하세요.",
        "body": "제주와 오키나와 숙소를 가격과 할인율로 비교해 보세요.",
        "cta": "전체 프로모션 보기",
        "image_prompt": "hotel booking promotion",
        "image_url": "https://gen-ai.asset.dev.loop-ad.org/genai/hero.png",
        "landing_url": LANDING_URL,
        **extensions,
    }


def _offer_links() -> tuple[PromotionOfferLink, ...]:
    return tuple(
        PromotionOfferLink(
            offer_id=hotel_id,
            destination_url=(
                f"https://demo-shoppingmall.dev.loop-ad.org/hotel/{hotel_id}"
            ),
        )
        for hotel_id in HOTEL_IDS
    )


def _offer_catalog(*, deal_code: str | None = None) -> dict[str, object]:
    hotels = []
    for index, hotel_id in enumerate(HOTEL_IDS):
        destination_id = "jeju" if hotel_id.startswith("jeju-") else "okinawa"
        hotel = {
            "offer_id": hotel_id,
            "hotel_name": f"StayLoop Hotel {index + 1}",
            "destination_id": destination_id,
            "currency": "KRW",
            "sale_price_per_night": 278000 - index * 10000,
            "original_price_per_night": 342000 - index * 10000,
            "discount_rate_percent": 19,
            "image_path": f"/stayloop/promotions/hotel-{index + 1}.png",
            "asset_id": f"hotel-{index + 1}-hero",
        }
        if deal_code is not None:
            hotel["destination_url"] = (
                "https://demo-shoppingmall.dev.loop-ad.org/"
                f"hotel/{hotel_id}?deal={deal_code}"
            )
        hotels.append(hotel)
    catalog = {
        "schema_version": "stayloop.promotion-price-catalog.v1",
        "catalog_id": "black-friday-hotels",
        "catalog_version": "v2",
        "promotion_label": "제주·오키나와 블랙프라이데이",
        "currency": "KRW",
        "price_basis": "one_room_one_night",
        "hotels": hotels,
    }
    if deal_code is not None:
        catalog["deal_code"] = deal_code
        catalog["landing_url"] = (
            "https://demo-shoppingmall.dev.loop-ad.org/"
            f"search?deal={deal_code}"
        )
    return catalog
