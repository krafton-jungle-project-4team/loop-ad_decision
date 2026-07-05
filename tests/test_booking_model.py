from __future__ import annotations

from app.analysis.booking_model import train_booking_propensity_model
from app.analysis.repositories import BookingTrainingRecord, HotelMarketingProfileRecord


def test_booking_propensity_model_learns_booking_signal() -> None:
    model = train_booking_propensity_model(
        [
            BookingTrainingRecord(
                is_mobile=0,
                is_package=0,
                stay_nights=1,
                days_until_checkin=50,
                event_count=100,
                booking_count=5,
            ),
            BookingTrainingRecord(
                is_mobile=1,
                is_package=1,
                stay_nights=4,
                days_until_checkin=3,
                event_count=100,
                booking_count=70,
            ),
        ]
    )

    assert model is not None
    low_prediction = model.predict_profile(
        HotelMarketingProfileRecord(
            project_id="hotel-client-a",
            profile_name="low_booking_profile",
            profile_json={
                "mobile_ratio": 0.0,
                "package_ratio": 0.0,
                "avg_stay_nights": 1.0,
                "avg_days_until_checkin": 50.0,
            },
        )
    )
    high_prediction = model.predict_profile(
        HotelMarketingProfileRecord(
            project_id="hotel-client-a",
            profile_name="high_booking_profile",
            profile_json={
                "mobile_ratio": 1.0,
                "package_ratio": 1.0,
                "avg_stay_nights": 4.0,
                "avg_days_until_checkin": 3.0,
            },
        )
    )

    assert low_prediction is not None
    assert high_prediction is not None
    assert high_prediction.probability > low_prediction.probability
    assert high_prediction.model_version == "booking_propensity_logistic_v1"
    assert high_prediction.training_sample_count == 2


def test_booking_propensity_model_returns_none_without_training_data() -> None:
    assert train_booking_propensity_model([]) is None
