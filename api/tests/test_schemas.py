"""Tests de validation Pydantic pour StationFeatures et BatchPredictionRequest."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from velib_api.schemas import BatchPredictionRequest, StationFeatures

VALID_PAYLOAD: dict = {
    "capacity": 30,
    "capacity_group": 1,
    "morning_evening_ratio": 1.05,
    "hour_sin": 0.866,
    "hour_cos": 0.5,
    "dow_sin": 0.0,
    "dow_cos": 1.0,
    "month": 5,
    "is_peak_hour": 0,
    "is_friday_evening": 0,
    "is_monday_morning": 1,
    "is_holiday": 0,
    "is_vacation": 0,
    "apparent_temperature": 16.5,
    "temp_anomalie": 1.2,
    "weather_severity": 0,
    "is_frozen": 0,
    "is_stormy": 0,
    "lag_60min": 45.0,
    "lag_240min": 38.0,
    "lag_res_240min": -3.5,
    "lat": 48.8566,
    "lon": 2.3522,
    "hour": 8,
    "station_trend_avg": 42.0,
}


# ─────────────────────────────────────────────────────────────────────────────
# StationFeatures
# ─────────────────────────────────────────────────────────────────────────────

def test_station_features_valid():
    sf = StationFeatures(**VALID_PAYLOAD)
    assert sf.capacity == 30
    assert sf.month == 5


def test_station_features_extra_field_rejected():
    with pytest.raises(ValidationError, match="extra_forbidden"):
        StationFeatures(**VALID_PAYLOAD, unknown_field=99)


def test_station_features_missing_required_field():
    payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "capacity"}
    with pytest.raises(ValidationError):
        StationFeatures(**payload)


def test_station_features_capacity_negative():
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "capacity": -1})


def test_station_features_capacity_group_out_of_range():
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "capacity_group": 4})


def test_station_features_month_boundaries():
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "month": 0})
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "month": 13})


def test_station_features_hour_out_of_range():
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "hour": 24})


def test_station_features_hour_sin_out_of_range():
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "hour_sin": 1.1})


def test_station_features_lag_out_of_range():
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "lag_60min": 101.0})


def test_station_features_station_trend_avg_out_of_range():
    with pytest.raises(ValidationError):
        StationFeatures(**{**VALID_PAYLOAD, "station_trend_avg": -0.1})


# ─────────────────────────────────────────────────────────────────────────────
# BatchPredictionRequest
# ─────────────────────────────────────────────────────────────────────────────

def test_batch_request_single_valid():
    req = BatchPredictionRequest(items=[VALID_PAYLOAD])
    assert req.items[0].capacity == 30


def test_batch_request_empty_list_rejected():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(items=[])


def test_batch_request_too_large_rejected():
    with pytest.raises(ValidationError):
        BatchPredictionRequest(items=[VALID_PAYLOAD] * 2001)


def test_batch_request_max_valid():
    req = BatchPredictionRequest(items=[VALID_PAYLOAD] * 2000)
    assert len(req.items) == 2000
