from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


AUDIENCE_REFERENCE_PATTERN = re.compile(
    r"^(?P<section>primary_signals|promotion_matched_features)"
    r"\[(?P<index>[0-9]+)\]$"
)
HOTEL_REFERENCE_PREFIX = "hotel_profile"
HOTEL_PATH_TOKEN_PATTERN = re.compile(
    r"(?:\.(?P<key>[A-Za-z_][A-Za-z0-9_]*))|(?:\[(?P<index>[0-9]+)\])"
)


class EvidenceReferenceError(ValueError):
    """Raised when a strategy evidence reference cannot be resolved."""


@dataclass(frozen=True)
class EvidenceResolver:
    audience_evidence: Mapping[str, Any]
    hotel_profile: Mapping[str, Any] | None

    def resolve(self, reference: str) -> Any:
        audience_match = AUDIENCE_REFERENCE_PATTERN.fullmatch(reference)
        if audience_match is not None:
            return self._resolve_audience_reference(
                section=audience_match.group("section"),
                index=int(audience_match.group("index")),
                reference=reference,
            )
        if reference.startswith(f"{HOTEL_REFERENCE_PREFIX}."):
            return self._resolve_hotel_reference(reference)
        raise EvidenceReferenceError(f"unsupported evidence reference: {reference}")

    def validate_all(self, references: Sequence[str]) -> None:
        for reference in references:
            self.resolve(reference)

    def _resolve_audience_reference(
        self,
        *,
        section: str,
        index: int,
        reference: str,
    ) -> Any:
        values = self.audience_evidence.get(section)
        if not isinstance(values, Sequence) or isinstance(values, str):
            raise EvidenceReferenceError(
                f"evidence reference does not point to a sequence: {reference}"
            )
        if index >= len(values):
            raise EvidenceReferenceError(
                f"evidence reference index is out of range: {reference}"
            )
        value = values[index]
        _validate_resolved_value(value, reference=reference)
        return value

    def _resolve_hotel_reference(self, reference: str) -> Any:
        if self.hotel_profile is None:
            raise EvidenceReferenceError(
                f"hotel profile is not available for reference: {reference}"
            )
        suffix = reference[len(HOTEL_REFERENCE_PREFIX) :]
        tokens = list(HOTEL_PATH_TOKEN_PATTERN.finditer(suffix))
        if not tokens or "".join(token.group(0) for token in tokens) != suffix:
            raise EvidenceReferenceError(f"invalid hotel evidence path: {reference}")

        value: Any = self.hotel_profile
        for token in tokens:
            key = token.group("key")
            raw_index = token.group("index")
            if key is not None:
                if not isinstance(value, Mapping) or key not in value:
                    raise EvidenceReferenceError(
                        f"hotel evidence path is missing: {reference}"
                    )
                value = value[key]
                continue
            if raw_index is None or not isinstance(value, Sequence) or isinstance(
                value,
                str,
            ):
                raise EvidenceReferenceError(
                    f"hotel evidence path is not indexable: {reference}"
                )
            index = int(raw_index)
            if index >= len(value):
                raise EvidenceReferenceError(
                    f"hotel evidence index is out of range: {reference}"
                )
            value = value[index]

        _validate_resolved_value(value, reference=reference)
        return value


def verified_hotel_benefits(
    resolver: EvidenceResolver,
) -> list[tuple[str, str]]:
    benefits: list[tuple[str, str]] = []
    boolean_benefit_paths = (
        (
            "free_cancellation",
            "hotel_profile.booking_policy.free_cancellation",
        ),
        (
            "breakfast_included",
            "hotel_profile.meal_policy.breakfast_included",
        ),
    )
    for benefit, reference in boolean_benefit_paths:
        try:
            value = resolver.resolve(reference)
        except EvidenceReferenceError:
            continue
        if value is True:
            benefits.append((benefit, reference))

    verified_benefits = None
    if resolver.hotel_profile is not None:
        verified_benefits = resolver.hotel_profile.get("verified_benefits")
    if isinstance(verified_benefits, Sequence) and not isinstance(
        verified_benefits,
        str,
    ):
        for index, value in enumerate(verified_benefits):
            if not isinstance(value, str) or not value.strip():
                continue
            benefits.append(
                (
                    value.strip(),
                    f"hotel_profile.verified_benefits[{index}]",
                )
            )
    return list(dict.fromkeys(benefits))


def _validate_resolved_value(value: Any, *, reference: str) -> None:
    if value is None:
        raise EvidenceReferenceError(f"evidence reference resolved to null: {reference}")
    if isinstance(value, str) and not value.strip():
        raise EvidenceReferenceError(
            f"evidence reference resolved to empty text: {reference}"
        )
    if isinstance(value, (Mapping, Sequence)) and not isinstance(value, str):
        if not value:
            raise EvidenceReferenceError(
                f"evidence reference resolved to an empty value: {reference}"
            )
