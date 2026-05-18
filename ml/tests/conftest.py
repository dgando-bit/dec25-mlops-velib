"""Fixtures partagées pour les tests du module ML."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from ml.src.models._helpers import FEATURES_FINAL


def make_features_df(n: int, rng: np.random.RandomState | None = None) -> pd.DataFrame:
    """Génère un DataFrame de n lignes avec les 24 features FEATURES_FINAL."""
    if rng is None:
        rng = np.random.RandomState(42)
    hours = rng.randint(0, 24, n)
    dows = rng.randint(0, 7, n)
    return pd.DataFrame({
        "capacity":              rng.randint(10, 60, n).astype(float),
        "capacity_group":        rng.randint(0, 4, n).astype(float),
        "morning_evening_ratio": rng.uniform(0.5, 2.0, n),
        "hour_sin":              np.sin(2 * np.pi * hours / 24),
        "hour_cos":              np.cos(2 * np.pi * hours / 24),
        "dow_sin":               np.sin(2 * np.pi * dows / 7),
        "dow_cos":               np.cos(2 * np.pi * dows / 7),
        "month":                 rng.randint(1, 13, n).astype(float),
        "is_peak_hour":          rng.randint(0, 2, n).astype(float),
        "is_friday_evening":     rng.randint(0, 2, n).astype(float),
        "is_monday_morning":     rng.randint(0, 2, n).astype(float),
        "is_holiday":            rng.randint(0, 2, n).astype(float),
        "is_vacation":           rng.randint(0, 2, n).astype(float),
        "apparent_temperature":  rng.uniform(-5, 35, n),
        "temp_anomalie":         rng.uniform(-10, 10, n),
        "weather_severity":      rng.randint(0, 5, n).astype(float),
        "is_frozen":             rng.randint(0, 2, n).astype(float),
        "is_stormy":             rng.randint(0, 2, n).astype(float),
        "lag_60min":             rng.uniform(0, 100, n),
        "lag_240min":            rng.uniform(0, 100, n),
        "lag_res_240min":        rng.uniform(-50, 50, n),
        "lat":                   rng.uniform(48.80, 48.90, n),
        "lon":                   rng.uniform(2.20, 2.40, n),
        "hour":                  hours.astype(float),
    })


@pytest.fixture(scope="session")
def fixture_model() -> Pipeline:
    """XGBRegressor minimal entraîné sur 500 lignes synthétiques (24 features)."""
    rng = np.random.RandomState(42)
    n = 500
    X = make_features_df(n, rng)
    y = rng.uniform(-20, 20, n)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(n_estimators=10, random_state=42, verbosity=0)),
    ])
    pipe.fit(X[FEATURES_FINAL], y)
    return pipe
