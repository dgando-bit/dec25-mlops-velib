"""
Fixtures de session pour les tests de l'API Vélib'.

Stratégie :
  - fixture_model : XGBRegressor minimal (n_estimators=10) entraîné sur 300 lignes
    synthétiques couvrant les 24 features exactes du modèle de production.
  - mock_metadata  : dict de métadonnées statiques simulant une réponse preload_model.
  - client         : TestClient FastAPI avec les 3 patches nécessaires pour éviter
    tout appel réseau vers MLflow pendant les tests :
      * velib_api.main.preload_model     → retourne mock_metadata (lifespan + /model/reload)
      * velib_api.dependencies.load_staging_model → retourne fixture_model (/predict)
      * velib_api.main.load_model_by_alias         → MagicMock avec cache_clear (/model/reload)
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

FEATURES_FINAL: list[str] = [
    "capacity", "capacity_group", "morning_evening_ratio",
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month",
    "is_peak_hour", "is_friday_evening", "is_monday_morning",
    "is_holiday", "is_vacation",
    "apparent_temperature", "temp_anomalie", "weather_severity",
    "is_frozen", "is_stormy",
    "lag_60min", "lag_240min", "lag_res_240min",
    "lat", "lon", "hour",
]


def _make_features_df(n: int, rng: np.random.RandomState) -> pd.DataFrame:
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
    """XGBRegressor minimal entraîné sur données synthétiques (24 features)."""
    rng = np.random.RandomState(42)
    n = 300
    X = _make_features_df(n, rng)
    y = rng.uniform(-20, 20, n)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(n_estimators=10, random_state=42, verbosity=0)),
    ])
    pipe.fit(X[FEATURES_FINAL], y)
    return pipe


@pytest.fixture(scope="session")
def mock_metadata() -> dict:
    return {
        "model_name": "velib_fill_rate_predictor",
        "alias":      "staging",
        "version":    "1",
        "run_id":     "test-run-id-000000",
        "framework":  "xgboost",
        "n_features": 24,
        "taux_r2":    0.83,
        "taux_mae":   8.3,
        "taux_mape":  36.7,
    }


@pytest.fixture(scope="session")
def client(fixture_model: Pipeline, mock_metadata: dict):
    """TestClient avec modèle fixture injecté — aucun appel MLflow."""
    from velib_api.main import app

    mock_lba = MagicMock(return_value=fixture_model)
    mock_lba.cache_clear = MagicMock()

    with (
        patch("velib_api.main.preload_model", return_value=mock_metadata),
        patch("velib_api.dependencies.load_staging_model", return_value=fixture_model),
        patch("velib_api.main.load_model_by_alias", mock_lba),
    ):
        with TestClient(app) as c:
            yield c
