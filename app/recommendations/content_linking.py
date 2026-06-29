from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class MappingContentIds:
    campaign_id: int | None = None
    creative_id: int | None = None
    coupon_id: int | None = None


def resolve_mapping_content_ids(
    *,
    repository: Any,
    project_id: str,
    action_id: str,
    execution_hint_json: dict[str, Any] | None,
) -> MappingContentIds:
    hint = execution_hint_json or {}
    campaign_id = get_hint_int(hint, "campaign_id")
    creative_id = get_hint_int(hint, "creative_id", "ad_creative_id")
    coupon_id = get_hint_int(hint, "coupon_id")

    creative = None
    if creative_id is not None:
        creative = repository.get_ad_creative(creative_id)
        if not is_usable_creative(creative, project_id):
            creative = None
            creative_id = None

    if creative is None:
        creative = repository.get_active_ad_creative_by_action(
            project_id=project_id,
            action_id=action_id,
        )

    if creative is None:
        return MappingContentIds(
            campaign_id=campaign_id,
            creative_id=creative_id,
            coupon_id=coupon_id,
        )

    return MappingContentIds(
        campaign_id=campaign_id or getattr(creative, "campaign_id", None),
        creative_id=getattr(creative, "id", creative_id),
        coupon_id=coupon_id or getattr(creative, "coupon_id", None),
    )


def get_hint_int(hint: dict[str, Any], *keys: str) -> int | None:
    for key in keys:
        value = coerce_int(hint.get(key))
        if value is not None:
            return value
    return None


def coerce_int(value: Any) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdecimal():
        return int(value)
    return None


def is_usable_creative(creative: Any, project_id: str) -> bool:
    if creative is None:
        return False
    return (
        getattr(creative, "project_id", None) == project_id
        and getattr(creative, "status", None) == "active"
        and bool(getattr(creative, "image_url", None))
    )
