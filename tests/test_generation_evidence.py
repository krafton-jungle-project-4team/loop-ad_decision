from __future__ import annotations

import pytest

from app.generation.evidence import (
    EvidenceReferenceError,
    EvidenceResolver,
    verified_hotel_benefits,
)


def test_evidence_resolver_resolves_audience_and_nested_hotel_references() -> None:
    resolver = EvidenceResolver(
        audience_evidence={
            "primary_signals": ["near_checkin"],
            "promotion_matched_features": ["free_cancellation"],
        },
        hotel_profile={
            "booking_policy": {"free_cancellation": True},
            "verified_benefits": ["late_checkout"],
        },
    )

    assert resolver.resolve("primary_signals[0]") == "near_checkin"
    assert resolver.resolve("promotion_matched_features[0]") == (
        "free_cancellation"
    )
    assert resolver.resolve("hotel_profile.booking_policy.free_cancellation") is True
    assert resolver.resolve("hotel_profile.verified_benefits[0]") == "late_checkout"


@pytest.mark.parametrize(
    "reference",
    (
        "primary_signals[1]",
        "hotel_profile.booking_policy.breakfast_included",
        "hotel_profile.verified_benefits[2]",
        "hotel_profile[bad]",
        "unknown_evidence[0]",
    ),
)
def test_evidence_resolver_rejects_invalid_or_missing_references(
    reference: str,
) -> None:
    resolver = EvidenceResolver(
        audience_evidence={"primary_signals": ["near_checkin"]},
        hotel_profile={
            "booking_policy": {"free_cancellation": True},
            "verified_benefits": ["late_checkout"],
        },
    )

    with pytest.raises(EvidenceReferenceError):
        resolver.resolve(reference)


def test_verified_hotel_benefits_requires_true_policy_or_explicit_verified_list() -> None:
    resolver = EvidenceResolver(
        audience_evidence={
            "promotion_matched_features": [
                "free_cancellation",
                "breakfast_included",
            ]
        },
        hotel_profile={
            "booking_policy": {"free_cancellation": False},
            "meal_policy": {"breakfast_included": True},
            "verified_benefits": ["late_checkout"],
        },
    )

    assert verified_hotel_benefits(resolver) == [
        (
            "breakfast_included",
            "hotel_profile.meal_policy.breakfast_included",
        ),
        ("late_checkout", "hotel_profile.verified_benefits[0]"),
    ]
