from app.generation.artifacts import render_email_html, source_for_channel
from app.generation.email_variants import (
    OFFER_CARDS_VARIANT,
    TEXT_POSTER_VARIANT,
    VISUAL_POSTER_VARIANT,
    build_email_creative_extensions,
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


def test_offer_card_variant_builds_eight_redirect_targets_and_email_html() -> None:
    extensions = build_email_creative_extensions(
        option_index=1,
        landing_url=LANDING_URL,
        offer_links=_offer_links(),
        offer_catalog=_offer_catalog(),
    )

    assert extensions["variant_type"] == OFFER_CARDS_VARIANT
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
    content_values = {
        "subject": "제주·오키나와 블랙프라이데이",
        "preheader": "8개 숙소의 데모 특가를 확인하세요.",
        "body": "제주 4곳과 오키나와 4곳을 한 번에 비교해 보세요.",
        "cta": "전체 프로모션 보기",
        "image_prompt": "hotel booking promotion",
        "image_url": "https://gen-ai.asset.dev.loop-ad.org/genai/hero.png",
        "landing_url": LANDING_URL,
        **extensions,
    }

    rendered = render_email_html(content_values)
    source = source_for_channel(
        channel=ContentChannel.EMAIL,
        content_values=content_values,
    )
    EmailHtmlSource.model_validate(source)

    assert rendered.count("숙소 확인하기") == 8
    assert "278,000원" in rendered
    assert "342,000원" in rendered
    assert "https://demo-shoppingmall.dev.loop-ad.org/stayloop/" in rendered
    for index in range(1, 9):
        assert f"{{{{offer_redirect_url_{index}}}}}" in rendered
        assert f"{{{{offer_redirect_url_{index}}}}}" in source[
            "required_placeholders"
        ]


def test_poster_variants_keep_only_the_primary_redirect() -> None:
    visual = build_email_creative_extensions(
        option_index=2,
        landing_url=LANDING_URL,
        offer_links=_offer_links(),
        offer_catalog=_offer_catalog(),
    )
    text = build_email_creative_extensions(
        option_index=3,
        landing_url=LANDING_URL,
        offer_links=_offer_links(),
        offer_catalog=_offer_catalog(),
    )

    assert visual["variant_type"] == VISUAL_POSTER_VARIANT
    assert text["variant_type"] == TEXT_POSTER_VARIANT
    assert visual["link_targets"] == [
        {"placeholder": "{{redirect_url}}", "target_type": "promotion"}
    ]
    assert text["link_targets"] == visual["link_targets"]
    assert "offers" not in visual
    assert "offers" not in text


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


def _offer_catalog() -> dict[str, object]:
    hotels = []
    for index, hotel_id in enumerate(HOTEL_IDS):
        destination_id = "jeju" if hotel_id.startswith("jeju-") else "okinawa"
        hotels.append(
            {
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
        )
    return {
        "schema_version": "stayloop.promotion-price-catalog.v1",
        "catalog_id": "black-friday-hotels",
        "catalog_version": "v2",
        "promotion_label": "제주·오키나와 블랙프라이데이",
        "currency": "KRW",
        "price_basis": "one_room_one_night",
        "hotels": hotels,
    }
