from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping, Sequence

from app.analysis.repositories import BookingTrainingRecord, HotelMarketingProfileRecord


FEATURE_NAMES = (
    "bias",
    "mobile_ratio",
    "package_ratio",
    "stay_nights_scaled",
    "near_checkin_score",
)

TRAINING_ITERATIONS = 80
LEARNING_RATE = 0.35
L2_REGULARIZATION = 0.002


@dataclass(frozen=True)
class BookingPropensityPrediction:
    probability: float
    model_version: str
    feature_values: Mapping[str, float]
    training_sample_count: int


@dataclass(frozen=True)
class BookingPropensityModel:
    coefficients: tuple[float, ...]
    training_sample_count: int
    model_version: str = "booking_propensity_logistic_v1"

    def predict_profile(
        self,
        profile: HotelMarketingProfileRecord | None,
    ) -> BookingPropensityPrediction | None:
        if profile is None:
            return None

        features = _features_from_profile(profile.profile_json)
        probability = _sigmoid(
            sum(
                coefficient * feature
                for coefficient, feature in zip(self.coefficients, features)
            )
        )
        return BookingPropensityPrediction(
            probability=round(probability, 6),
            model_version=self.model_version,
            feature_values={
                name: round(value, 6)
                for name, value in zip(FEATURE_NAMES, features)
                if name != "bias"
            },
            training_sample_count=self.training_sample_count,
        )


def train_booking_propensity_model(
    records: Sequence[BookingTrainingRecord],
) -> BookingPropensityModel | None:
    usable_records = [
        record
        for record in records
        if record.event_count > 0 and 0 <= record.booking_count <= record.event_count
    ]
    if not usable_records:
        return None

    coefficients = [0.0] * len(FEATURE_NAMES)
    total_weight = sum(record.event_count for record in usable_records)
    for _ in range(TRAINING_ITERATIONS):
        gradients = [0.0] * len(coefficients)
        for record in usable_records:
            features = _features_from_training_record(record)
            label = record.booking_count / record.event_count
            prediction = _sigmoid(
                sum(
                    coefficient * feature
                    for coefficient, feature in zip(coefficients, features)
                )
            )
            weight = record.event_count / total_weight
            error = prediction - label
            for index, feature in enumerate(features):
                gradients[index] += weight * error * feature

        for index, gradient in enumerate(gradients):
            regularization = 0.0 if index == 0 else L2_REGULARIZATION * coefficients[index]
            coefficients[index] -= LEARNING_RATE * (gradient + regularization)

    return BookingPropensityModel(
        coefficients=tuple(round(coefficient, 8) for coefficient in coefficients),
        training_sample_count=len(usable_records),
    )


def _features_from_training_record(record: BookingTrainingRecord) -> tuple[float, ...]:
    return (
        1.0,
        _clamp01(record.is_mobile),
        _clamp01(record.is_package),
        _scale_stay_nights(record.stay_nights),
        _near_checkin_score(record.days_until_checkin),
    )


def _features_from_profile(profile_json: Mapping[str, object]) -> tuple[float, ...]:
    return (
        1.0,
        _clamp01(_float_value(profile_json.get("mobile_ratio"))),
        _clamp01(_float_value(profile_json.get("package_ratio"))),
        _scale_stay_nights(_float_value(profile_json.get("avg_stay_nights"))),
        _near_checkin_score(
            _float_value(profile_json.get("avg_days_until_checkin"), default=60.0),
        ),
    )


def _scale_stay_nights(value: float) -> float:
    return _clamp01(value / 14.0)


def _near_checkin_score(value: float) -> float:
    return 1.0 - _clamp01(value / 60.0)


def _sigmoid(value: float) -> float:
    if value >= 0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def _float_value(value: object, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
