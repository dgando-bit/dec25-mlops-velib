"""Tests des fonctions de feature engineering (ml/src/features/_helpers.py)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.src.features._helpers import (
    CAPACITY_BINS,
    CAPACITY_LABELS,
    add_capacity_group,
    add_residual_target,
    add_temporal_features,
    encode_calendar_booleans,
    temporal_split,
)


# ─────────────────────────────────────────────────────────────────────────────
# add_temporal_features
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_features_hour_extracted():
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-03-11 08:30")]})
    result = add_temporal_features(df)
    assert result["hour"].iloc[0] == 8
    assert result["month"].iloc[0] == 3


def test_temporal_features_cyclic_encoding():
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-01 06:00")]})
    result = add_temporal_features(df)
    expected_sin = np.sin(2 * np.pi * 6 / 24)
    assert result["hour_sin"].iloc[0] == pytest.approx(expected_sin, abs=1e-3)


def test_temporal_features_is_peak_hour_true():
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-01 08:00")]})
    result = add_temporal_features(df)
    assert result["is_peak_hour"].iloc[0] == 1


def test_temporal_features_is_peak_hour_false():
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-01 14:00")]})
    result = add_temporal_features(df)
    assert result["is_peak_hour"].iloc[0] == 0


def test_temporal_features_friday_evening():
    # 2024-01-05 est un vendredi
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-05 19:00")]})
    result = add_temporal_features(df)
    assert result["is_friday_evening"].iloc[0] == 1


def test_temporal_features_monday_morning():
    # 2024-01-01 est un lundi
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-01 08:00")]})
    result = add_temporal_features(df)
    assert result["is_monday_morning"].iloc[0] == 1


def test_temporal_features_not_monday_morning():
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-02 08:00")]})
    result = add_temporal_features(df)
    assert result["is_monday_morning"].iloc[0] == 0


def test_temporal_features_no_mutation():
    df = pd.DataFrame({"datetime": [pd.Timestamp("2024-01-01 12:00")]})
    original_cols = set(df.columns)
    _ = add_temporal_features(df)
    assert set(df.columns) == original_cols


# ─────────────────────────────────────────────────────────────────────────────
# add_capacity_group
# ─────────────────────────────────────────────────────────────────────────────

def test_capacity_group_small():
    df = pd.DataFrame({"capacity": [15]})
    result = add_capacity_group(df)
    assert result["capacity_group"].iloc[0] == 0


def test_capacity_group_medium():
    df = pd.DataFrame({"capacity": [28]})
    result = add_capacity_group(df)
    assert result["capacity_group"].iloc[0] == 1


def test_capacity_group_large():
    df = pd.DataFrame({"capacity": [42]})
    result = add_capacity_group(df)
    assert result["capacity_group"].iloc[0] == 2


def test_capacity_group_very_large():
    df = pd.DataFrame({"capacity": [60]})
    result = add_capacity_group(df)
    assert result["capacity_group"].iloc[0] == 3


def test_capacity_group_all_labels_covered():
    capacities = [10, 25, 40, 70]
    df = pd.DataFrame({"capacity": capacities})
    result = add_capacity_group(df)
    assert sorted(result["capacity_group"].unique().tolist()) == CAPACITY_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# encode_calendar_booleans
# ─────────────────────────────────────────────────────────────────────────────

def test_encode_calendar_booleans_converts_to_int8():
    df = pd.DataFrame({"is_holiday": [True, False], "is_vacation": [False, True]})
    result = encode_calendar_booleans(df)
    assert result["is_holiday"].dtype == np.dtype("int8")
    assert result["is_vacation"].dtype == np.dtype("int8")
    assert result["is_holiday"].tolist() == [1, 0]
    assert result["is_vacation"].tolist() == [0, 1]


def test_encode_calendar_booleans_missing_column_ignored():
    df = pd.DataFrame({"is_holiday": [True, False]})
    result = encode_calendar_booleans(df)
    assert "is_vacation" not in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# add_residual_target
# ─────────────────────────────────────────────────────────────────────────────

def test_residual_target_value():
    train = pd.DataFrame({
        "taux":               [50.0, 60.0],
        "station_trend_avg":  [45.0, 55.0],
        "lag_240min":         [48.0, 58.0],
    })
    test = pd.DataFrame({
        "taux":               [40.0],
        "station_trend_avg":  [42.0],
        "lag_240min":         [39.0],
    })
    train_out, test_out = add_residual_target(train, test)
    assert train_out["residual_target"].iloc[0] == pytest.approx(5.0)
    assert train_out["residual_target"].iloc[1] == pytest.approx(5.0)
    assert test_out["residual_target"].iloc[0] == pytest.approx(-2.0)


def test_residual_lag_240():
    train = pd.DataFrame({
        "taux":               [50.0],
        "station_trend_avg":  [40.0],
        "lag_240min":         [55.0],
    })
    test = pd.DataFrame({
        "taux":               [45.0],
        "station_trend_avg":  [40.0],
        "lag_240min":         [35.0],
    })
    train_out, test_out = add_residual_target(train, test)
    assert train_out["lag_res_240min"].iloc[0] == pytest.approx(15.0)
    assert test_out["lag_res_240min"].iloc[0] == pytest.approx(-5.0)


def test_residual_target_no_mutation():
    train = pd.DataFrame({
        "taux": [50.0], "station_trend_avg": [45.0], "lag_240min": [48.0],
    })
    test = pd.DataFrame({
        "taux": [40.0], "station_trend_avg": [42.0], "lag_240min": [39.0],
    })
    cols_before = set(train.columns)
    add_residual_target(train, test)
    assert set(train.columns) == cols_before


# ─────────────────────────────────────────────────────────────────────────────
# temporal_split
# ─────────────────────────────────────────────────────────────────────────────

def test_temporal_split_respects_ratio():
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({"datetime": dates, "val": range(100)})
    train, test = temporal_split(df, test_size=0.2)
    assert len(train) + len(test) == 100
    assert abs(len(test) / 100 - 0.2) <= 0.05


def test_temporal_split_no_data_leakage():
    dates = pd.date_range("2024-01-01", periods=100, freq="h")
    df = pd.DataFrame({"datetime": dates, "val": range(100)})
    train, test = temporal_split(df, test_size=0.2)
    assert train["datetime"].max() <= test["datetime"].min()
