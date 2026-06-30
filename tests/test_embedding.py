from __future__ import annotations

from app.decision import embedding
from app.decision.embedding import (
    BEHAVIOR_RESERVED_START,
    FUTURE_RESERVED_START,
    VECTOR_DIMENSION,
    embed_segment,
    embed_user,
)


def attrs() -> dict[str, str]:
    return {
        "age_group": "30s",
        "gender": " Male ",
        "device_type": "Mobile Web",
        "acquisition_channel": "Kakao",
        "primary_category": "Fresh Food",
    }


def test_vectorizer_is_deterministic_and_64_dimensions() -> None:
    first = embed_user(attrs())
    second = embed_user(attrs())

    assert first == second
    assert len(first) == VECTOR_DIMENSION


def test_reserved_behavior_and_future_slots_are_zero() -> None:
    vector = embed_user(attrs())

    assert vector[BEHAVIOR_RESERVED_START:] == [0.0] * (VECTOR_DIMENSION - BEHAVIOR_RESERVED_START)
    assert vector[FUTURE_RESERVED_START:] == [0.0] * (VECTOR_DIMENSION - FUTURE_RESERVED_START)


def test_user_and_segment_use_same_vector_space_after_normalization() -> None:
    assert embed_user(attrs()) == embed_segment(
        {
            "age_group": "30s",
            "gender": "male",
            "device": "mobile web",
            "channel": "kakao",
            "category": "fresh food",
        }
    )


def test_unknown_values_are_skipped_and_zero_vector_is_guarded() -> None:
    vector = embed_user(
        {
            "age_group": "",
            "gender": "unknown",
            "device_type": None,
            "acquisition_channel": "n/a",
            "primary_category": "(not set)",
        }
    )

    assert vector == [0.0] * VECTOR_DIMENSION


def test_vectorizer_imports_shared_analysis_normalizer() -> None:
    assert embedding.normalize_dimensions.__module__ == "app.analysis.segments"
