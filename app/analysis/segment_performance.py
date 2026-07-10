from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence


CONTEXTUAL_BOOKING_MODEL_VERSION = "dec.contextual-booking-calibration.v1"
DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "contextual_booking_calibration_v1.json"
)
MODEL_CANDIDATE_TYPES = (
    "intent_matched",
    "target_destination_affinity",
    "funnel_recovery",
    "benefit_value_seeker",
    "promotion_responsive",
    "general_destination_explorer",
)
NUMERIC_FEATURE_NAMES = (
    "promotion_condition_match",
    "destination_context_required",
    "destination_match_user_rate",
    "destination_match_event_rate",
    "eligible_destination_match_user_rate",
    "hotel_detail_view_user_rate",
    "booking_start_user_rate",
    "booking_complete_user_rate",
    "funnel_recovery_user_rate",
    "benefit_user_rate",
    "promotion_response_user_rate",
    "sample_reliability",
)
MODEL_FEATURE_NAMES = NUMERIC_FEATURE_NAMES + tuple(
    f"candidate_type__{candidate_type}"
    for candidate_type in MODEL_CANDIDATE_TYPES
)


@dataclass(frozen=True, slots=True)
class SegmentPerformanceFeatures:
    candidate_type: str
    promotion_condition_match: float
    destination_context_required: bool
    destination_match_user_rate: float
    destination_match_event_rate: float
    eligible_destination_match_user_rate: float
    hotel_detail_view_user_rate: float
    booking_start_user_rate: float
    booking_complete_user_rate: float
    funnel_recovery_user_rate: float
    benefit_user_rate: float
    promotion_response_user_rate: float
    sample_reliability: float

    def model_values(self) -> dict[str, float]:
        values = {
            "promotion_condition_match": _clamp01(
                self.promotion_condition_match
            ),
            "destination_context_required": float(
                self.destination_context_required
            ),
            "destination_match_user_rate": _clamp01(
                self.destination_match_user_rate
            ),
            "destination_match_event_rate": _clamp01(
                self.destination_match_event_rate
            ),
            "eligible_destination_match_user_rate": _clamp01(
                self.eligible_destination_match_user_rate
            ),
            "hotel_detail_view_user_rate": _clamp01(
                self.hotel_detail_view_user_rate
            ),
            "booking_start_user_rate": _clamp01(
                self.booking_start_user_rate
            ),
            "booking_complete_user_rate": _clamp01(
                self.booking_complete_user_rate
            ),
            "funnel_recovery_user_rate": _clamp01(
                self.funnel_recovery_user_rate
            ),
            "benefit_user_rate": _clamp01(self.benefit_user_rate),
            "promotion_response_user_rate": _clamp01(
                self.promotion_response_user_rate
            ),
            "sample_reliability": _clamp01(self.sample_reliability),
        }
        values.update(
            {
                f"candidate_type__{candidate_type}": float(
                    self.candidate_type == candidate_type
                )
                for candidate_type in MODEL_CANDIDATE_TYPES
            }
        )
        return values

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> SegmentPerformanceFeatures:
        return cls(
            candidate_type=str(payload.get("candidate_type", "unknown")),
            promotion_condition_match=float(
                payload.get("promotion_condition_match", 0.0) or 0.0
            ),
            destination_context_required=bool(
                payload.get("destination_context_required", False)
            ),
            destination_match_user_rate=float(
                payload.get("destination_match_user_rate", 0.0) or 0.0
            ),
            destination_match_event_rate=float(
                payload.get("destination_match_event_rate", 0.0) or 0.0
            ),
            eligible_destination_match_user_rate=float(
                payload.get("eligible_destination_match_user_rate", 0.0) or 0.0
            ),
            hotel_detail_view_user_rate=float(
                payload.get("hotel_detail_view_user_rate", 0.0) or 0.0
            ),
            booking_start_user_rate=float(
                payload.get("booking_start_user_rate", 0.0) or 0.0
            ),
            booking_complete_user_rate=float(
                payload.get("booking_complete_user_rate", 0.0) or 0.0
            ),
            funnel_recovery_user_rate=float(
                payload.get("funnel_recovery_user_rate", 0.0) or 0.0
            ),
            benefit_user_rate=float(payload.get("benefit_user_rate", 0.0) or 0.0),
            promotion_response_user_rate=float(
                payload.get("promotion_response_user_rate", 0.0) or 0.0
            ),
            sample_reliability=float(
                payload.get("sample_reliability", 0.0) or 0.0
            ),
        )


class SegmentPerformancePredictor(Protocol):
    version: str
    method: str
    calibration_status: str

    def predict(self, features: SegmentPerformanceFeatures) -> float:
        ...

    def metadata(self) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True, slots=True)
class CalibrationTrainingExample:
    features: SegmentPerformanceFeatures
    success_count: int
    sample_size: int

    @property
    def outcome_rate(self) -> float:
        if self.sample_size <= 0:
            return 0.0
        return _clamp01(self.success_count / self.sample_size)


@dataclass(frozen=True, slots=True)
class LogisticSegmentPerformanceModel:
    intercept: float
    coefficients: tuple[float, ...]
    feature_means: tuple[float, ...]
    feature_scales: tuple[float, ...]
    training_metadata: Mapping[str, Any]
    feature_names: tuple[str, ...] = MODEL_FEATURE_NAMES
    version: str = CONTEXTUAL_BOOKING_MODEL_VERSION
    method: str = "temporal_holdout_logistic_calibration"
    calibration_status: str = "calibrated"

    def __post_init__(self) -> None:
        expected = len(self.feature_names)
        if not all(
            len(values) == expected
            for values in (
                self.coefficients,
                self.feature_means,
                self.feature_scales,
            )
        ):
            raise ValueError("calibration model dimensions do not match feature_names")
        if any(scale <= 0 for scale in self.feature_scales):
            raise ValueError("calibration feature scales must be positive")

    def predict(self, features: SegmentPerformanceFeatures) -> float:
        raw_values = features.model_values()
        linear = self.intercept
        for index, feature_name in enumerate(self.feature_names):
            standardized = (
                raw_values.get(feature_name, 0.0) - self.feature_means[index]
            ) / self.feature_scales[index]
            linear += self.coefficients[index] * standardized
        return _sigmoid(linear)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "model_version": self.version,
            "method": self.method,
            "calibration_status": self.calibration_status,
            **dict(self.training_metadata),
        }

    def to_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "method": self.method,
            "calibration_status": self.calibration_status,
            "feature_names": list(self.feature_names),
            "intercept": self.intercept,
            "coefficients": list(self.coefficients),
            "feature_means": list(self.feature_means),
            "feature_scales": list(self.feature_scales),
            "training_metadata": dict(self.training_metadata),
        }

    @classmethod
    def from_json(
        cls,
        payload: Mapping[str, Any],
    ) -> LogisticSegmentPerformanceModel:
        version = str(payload.get("version", ""))
        if version != CONTEXTUAL_BOOKING_MODEL_VERSION:
            raise ValueError(f"unsupported calibration model version: {version!r}")
        feature_names = tuple(str(value) for value in payload["feature_names"])
        if feature_names != MODEL_FEATURE_NAMES:
            raise ValueError("calibration model feature contract does not match runtime")
        return cls(
            intercept=float(payload["intercept"]),
            coefficients=tuple(float(value) for value in payload["coefficients"]),
            feature_means=tuple(float(value) for value in payload["feature_means"]),
            feature_scales=tuple(float(value) for value in payload["feature_scales"]),
            training_metadata=dict(payload.get("training_metadata", {})),
            feature_names=feature_names,
        )


class ContextualBookingHeuristicPredictor:
    version = "dec.contextual-booking-heuristic.v1"
    method = "destination_context_heuristic"
    calibration_status = "uncalibrated_fallback"

    def predict(self, features: SegmentPerformanceFeatures) -> float:
        destination_signal = (
            0.7 * features.destination_match_user_rate
            + 0.3 * features.destination_match_event_rate
        )
        intent_signal = (
            0.30 * features.hotel_detail_view_user_rate
            + 0.25 * features.funnel_recovery_user_rate
            + 0.20 * features.booking_start_user_rate
            + 0.15 * features.benefit_user_rate
            + 0.10 * features.promotion_response_user_rate
        )
        predicted = (
            0.005
            + 0.08 * destination_signal
            + 0.04 * features.eligible_destination_match_user_rate
            + 0.025 * intent_signal
            + 0.02 * features.promotion_condition_match
        )
        if features.destination_context_required and destination_signal <= 0:
            predicted *= 0.2
        return _clamp01(predicted)

    def metadata(self) -> Mapping[str, Any]:
        return {
            "model_version": self.version,
            "method": self.method,
            "calibration_status": self.calibration_status,
        }


def fit_logistic_segment_performance_model(
    examples: Sequence[CalibrationTrainingExample],
    *,
    training_metadata: Mapping[str, Any] | None = None,
    iterations: int = 4000,
    learning_rate: float = 0.08,
    l2_penalty: float = 0.05,
) -> LogisticSegmentPerformanceModel:
    valid = [
        example
        for example in examples
        if example.sample_size > 0
        and 0 <= example.success_count <= example.sample_size
    ]
    if len(valid) < 2:
        raise ValueError("at least two calibration examples are required")
    if iterations <= 0 or learning_rate <= 0 or l2_penalty < 0:
        raise ValueError("invalid calibration optimizer settings")

    rows = [example.features.model_values() for example in valid]
    sample_weights = [math.sqrt(example.sample_size) for example in valid]
    total_weight = sum(sample_weights)
    means: list[float] = []
    scales: list[float] = []
    for feature_name in MODEL_FEATURE_NAMES:
        values = [row.get(feature_name, 0.0) for row in rows]
        mean = sum(
            value * weight
            for value, weight in zip(values, sample_weights, strict=True)
        ) / total_weight
        variance = sum(
            weight * (value - mean) ** 2
            for value, weight in zip(values, sample_weights, strict=True)
        ) / total_weight
        means.append(mean)
        scales.append(max(math.sqrt(variance), 1e-6))

    matrix = [
        [
            (row.get(feature_name, 0.0) - means[index]) / scales[index]
            for index, feature_name in enumerate(MODEL_FEATURE_NAMES)
        ]
        for row in rows
    ]
    outcomes = [example.outcome_rate for example in valid]
    total_success = sum(example.success_count for example in valid)
    total_sample = sum(example.sample_size for example in valid)
    base_rate = (total_success + 0.5) / (total_sample + 1.0)
    intercept = _logit(base_rate)
    coefficients = [0.0] * len(MODEL_FEATURE_NAMES)

    for iteration in range(iterations):
        intercept_gradient = 0.0
        coefficient_gradients = [0.0] * len(coefficients)
        for row, outcome, weight in zip(
            matrix,
            outcomes,
            sample_weights,
            strict=True,
        ):
            prediction = _sigmoid(
                intercept
                + sum(
                    coefficient * value
                    for coefficient, value in zip(
                        coefficients,
                        row,
                        strict=True,
                    )
                )
            )
            error = (prediction - outcome) * weight
            intercept_gradient += error
            for index, value in enumerate(row):
                coefficient_gradients[index] += error * value

        step = learning_rate / math.sqrt(1.0 + iteration / 250.0)
        intercept -= step * intercept_gradient / total_weight
        for index in range(len(coefficients)):
            regularized_gradient = (
                coefficient_gradients[index] / total_weight
                + l2_penalty * coefficients[index]
            )
            coefficients[index] -= step * regularized_gradient

    metadata = {
        "training_example_count": len(valid),
        "training_candidate_user_observation_count": total_sample,
        "training_contextual_booking_observation_count": total_success,
        "training_contextual_booking_observation_rate": _clamp01(
            total_success / max(total_sample, 1)
        ),
        **dict(training_metadata or {}),
    }
    return LogisticSegmentPerformanceModel(
        intercept=intercept,
        coefficients=tuple(coefficients),
        feature_means=tuple(means),
        feature_scales=tuple(scales),
        training_metadata=metadata,
    )


def load_segment_performance_model(
    path: Path,
) -> LogisticSegmentPerformanceModel:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration model must be a JSON object")
    return LogisticSegmentPerformanceModel.from_json(payload)


def build_segment_performance_predictor(
    model_path: str | Path | None = None,
) -> SegmentPerformancePredictor:
    path = Path(model_path).expanduser() if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        return load_segment_performance_model(path)
    if model_path is not None:
        raise ValueError(f"segment performance model not found: {path}")
    return ContextualBookingHeuristicPredictor()


def write_segment_performance_model(
    model: LogisticSegmentPerformanceModel,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(model.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sigmoid(value: float) -> float:
    if value >= 0:
        exponent = math.exp(-min(value, 700.0))
        return 1.0 / (1.0 + exponent)
    exponent = math.exp(max(value, -700.0))
    return exponent / (1.0 + exponent)


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-9), 1.0 - 1e-9)
    return math.log(clipped / (1.0 - clipped))


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))
