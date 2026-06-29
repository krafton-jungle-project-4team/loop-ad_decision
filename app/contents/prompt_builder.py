from __future__ import annotations

from typing import Any

from app.contents.types import RecommendationActionTarget


BLOCKED_PROMPT_KEYS = {
    "email",
    "event_id",
    "event_time",
    "external_user_id",
    "ip",
    "name",
    "phone",
    "raw_event",
    "raw_events",
    "session_id",
    "user_id",
}


class ContentPromptBuilder:
    """Builds a privacy-safe prompt payload from aggregated decision data."""

    def build(self, target: RecommendationActionTarget, variant_key: str) -> dict[str, Any]:
        return {
            "variant_key": variant_key,
            "project_id": str(target.project_id),
            "analysis_date": str(target.analysis_date),
            "segment": {
                "id": target.segment.id,
                "segment_key": target.segment.segment_key,
                "name": target.segment.name,
                "description": target.segment.description,
                "attributes": self._sanitize(target.segment.attributes),
            },
            "action": {
                "action_key": target.action_key,
                "action_type": target.action_type,
                "title": target.action_title,
                "description": target.action_description,
            },
            "metrics": self._sanitize(target.metrics),
            "root_cause": self._sanitize(target.root_cause),
        }

    def _sanitize(self, value: Any) -> Any:
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for key, nested_value in value.items():
                normalized_key = str(key).strip().lower()
                if normalized_key in BLOCKED_PROMPT_KEYS:
                    continue
                sanitized[key] = self._sanitize(nested_value)
            return sanitized
        if isinstance(value, list):
            return [self._sanitize(item) for item in value]
        return value
