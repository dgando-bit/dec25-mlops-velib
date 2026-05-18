"""Tests des helpers de nettoyage de shared/utils/data_cleaning.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shared.utils.data_cleaning import (
    apply_weather_severity,
    coerce_booleans,
    compute_fill_rate,
    compute_total_capacity,
    filter_active_stations,
    filter_valid_capacities,
    filter_valid_fill_rate,
    filter_stations_with_variance,
)


# ─────────────────────────────────────────────────────────────────────────────
# coerce_booleans
# ─────────────────────────────────────────────────────────────────────────────

def test_coerce_booleans_string_true_false():
    df = pd.DataFrame({"col": ["True", "False", "true", "false"]})
    result = coerce_booleans(df, ["col"])
    assert result["col"].tolist() == [True, False, True, False]


def test_coerce_booleans_numeric_1_0():
    df = pd.DataFrame({"col": [1, 0, 1, 0]})
    result = coerce_booleans(df, ["col"])
    assert result["col"].tolist() == [True, False, True, False]


def test_coerce_booleans_native_bool_unchanged():
    df = pd.DataFrame({"col": [True, False]})
    result = coerce_booleans(df, ["col"])
    assert result["col"].dtype == bool


def test_coerce_booleans_missing_column_ignored():
    df = pd.DataFrame({"a": [1, 0]})
    result = coerce_booleans(df, ["a", "nonexistent"])
    assert "nonexistent" not in result.columns


# ─────────────────────────────────────────────────────────────────────────────
# compute_total_capacity
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_total_capacity_normal():
    df = pd.DataFrame({
        "bikes_mechanical":   [10, 5],
        "bikes_ebike":        [2, 3],
        "numdocksavailable":  [8, 12],
    })
    result = compute_total_capacity(df)
    assert result.tolist() == [20, 20]


def test_compute_total_capacity_with_nan():
    df = pd.DataFrame({
        "bikes_mechanical":   [np.nan, 5],
        "bikes_ebike":        [2, np.nan],
        "numdocksavailable":  [8, 12],
    })
    result = compute_total_capacity(df)
    assert result.iloc[0] == 10
    assert result.iloc[1] == 17


# ─────────────────────────────────────────────────────────────────────────────
# compute_fill_rate
# ─────────────────────────────────────────────────────────────────────────────

def test_compute_fill_rate_fifty_percent():
    df = pd.DataFrame({
        "bikes_mechanical":   [10],
        "bikes_ebike":        [0],
        "numdocksavailable":  [10],
    })
    result = compute_fill_rate(df)
    assert result.iloc[0] == pytest.approx(50.0)


def test_compute_fill_rate_full():
    df = pd.DataFrame({
        "bikes_mechanical":   [20],
        "bikes_ebike":        [0],
        "numdocksavailable":  [0],
    })
    result = compute_fill_rate(df)
    assert result.iloc[0] == pytest.approx(100.0)


def test_compute_fill_rate_zero_capacity_returns_nan():
    df = pd.DataFrame({
        "bikes_mechanical":   [0],
        "bikes_ebike":        [0],
        "numdocksavailable":  [0],
    })
    result = compute_fill_rate(df)
    assert result.isna().all()


# ─────────────────────────────────────────────────────────────────────────────
# filter_active_stations
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_active_stations_keeps_renting():
    df = pd.DataFrame({"is_renting": [True, False, True], "val": [1, 2, 3]})
    result = filter_active_stations(df)
    assert len(result) == 2
    assert result["val"].tolist() == [1, 3]


def test_filter_active_stations_no_column_returns_all():
    df = pd.DataFrame({"val": [1, 2, 3]})
    result = filter_active_stations(df)
    assert len(result) == 3


# ─────────────────────────────────────────────────────────────────────────────
# filter_valid_capacities
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_valid_capacities_removes_zero():
    df = pd.DataFrame({
        "capacity":           [0, 20, 30],
        "bikes_mechanical":   [0, 10, 15],
        "bikes_ebike":        [0, 0, 0],
        "numdocksavailable":  [0, 10, 15],
    })
    result = filter_valid_capacities(df)
    assert len(result) == 2
    assert result["capacity"].tolist() == [20, 30]


# ─────────────────────────────────────────────────────────────────────────────
# filter_valid_fill_rate
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_valid_fill_rate_removes_out_of_bounds():
    df = pd.DataFrame({"taux": [-1.0, 0.0, 50.0, 100.0, 101.0]})
    result = filter_valid_fill_rate(df)
    assert len(result) == 3
    assert result["taux"].tolist() == [0.0, 50.0, 100.0]


# ─────────────────────────────────────────────────────────────────────────────
# apply_weather_severity
# ─────────────────────────────────────────────────────────────────────────────

def test_weather_severity_clear_codes():
    codes = pd.Series([0, 1, 2, 3])
    result = apply_weather_severity(codes)
    assert (result["weather_severity"] == 0).all()
    assert (result["is_frozen"] == 0).all()
    assert (result["is_stormy"] == 0).all()


def test_weather_severity_heavy_rain():
    codes = pd.Series([63, 82])
    result = apply_weather_severity(codes)
    assert (result["weather_severity"] == 3).all()


def test_weather_severity_storm_codes():
    codes = pd.Series([95, 96, 99])
    result = apply_weather_severity(codes)
    assert (result["weather_severity"] == 4).all()
    assert (result["is_stormy"] == 1).all()


def test_weather_severity_frozen_codes():
    codes = pd.Series([71, 73, 75])
    result = apply_weather_severity(codes)
    assert (result["is_frozen"] == 1).all()
    assert (result["weather_severity"] == 4).all()


def test_weather_severity_returns_three_series():
    codes = pd.Series([0, 45, 95])
    result = apply_weather_severity(codes)
    assert set(result.keys()) == {"weather_severity", "is_frozen", "is_stormy"}
    for series in result.values():
        assert len(series) == 3


# ─────────────────────────────────────────────────────────────────────────────
# filter_stations_with_variance
# ─────────────────────────────────────────────────────────────────────────────

def test_filter_stations_with_variance_removes_flat():
    df = pd.DataFrame({
        "station_id": [1, 1, 1, 2, 2, 2],
        "taux":       [50.0, 50.0, 50.0, 20.0, 60.0, 80.0],
    })
    result, excluded = filter_stations_with_variance(df, min_variance=1.0)
    assert 1 in excluded
    assert 2 not in excluded
    assert len(result) == 3
