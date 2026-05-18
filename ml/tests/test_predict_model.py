"""Tests des fonctions d'inférence (ml/src/models/predict_model.py et _helpers.py)."""
from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.src.models._helpers import FEATURES_FINAL, reconstruct_target
from ml.src.models.predict_model import predict_with_confidence
from ml.tests.conftest import make_features_df


# ─────────────────────────────────────────────────────────────────────────────
# reconstruct_target
# ─────────────────────────────────────────────────────────────────────────────

def test_reconstruct_target_normal():
    residual = np.array([5.0, -5.0])
    trend = pd.Series([50.0, 50.0])
    result = reconstruct_target(residual, trend)
    assert result.iloc[0] == pytest.approx(55.0)
    assert result.iloc[1] == pytest.approx(45.0)


def test_reconstruct_target_clipped_at_100():
    residual = np.array([60.0])
    trend = pd.Series([80.0])
    result = reconstruct_target(residual, trend)
    assert result.iloc[0] == 100.0


def test_reconstruct_target_clipped_at_0():
    residual = np.array([-60.0])
    trend = pd.Series([20.0])
    result = reconstruct_target(residual, trend)
    assert result.iloc[0] == 0.0


def test_reconstruct_target_preserves_index():
    idx = pd.RangeIndex(start=10, stop=13)
    residual = np.array([0.0, 0.0, 0.0])
    trend = pd.Series([50.0, 50.0, 50.0], index=idx)
    result = reconstruct_target(residual, trend)
    assert list(result.index) == list(idx)


# ─────────────────────────────────────────────────────────────────────────────
# predict_with_confidence — structure de sortie
# ─────────────────────────────────────────────────────────────────────────────

def test_predict_with_confidence_output_columns(fixture_model):
    rng = np.random.RandomState(0)
    X = make_features_df(5, rng)
    trend = pd.Series(rng.uniform(30, 70, 5))
    result = predict_with_confidence(X[FEATURES_FINAL], trend, model=fixture_model)
    assert set(result.columns) == {"taux_predicted", "residual_predicted", "alert_level"}
    assert len(result) == 5


def test_predict_with_confidence_taux_in_bounds(fixture_model):
    rng = np.random.RandomState(1)
    X = make_features_df(20, rng)
    trend = pd.Series(rng.uniform(0, 100, 20))
    result = predict_with_confidence(X[FEATURES_FINAL], trend, model=fixture_model)
    assert (result["taux_predicted"] >= 0).all()
    assert (result["taux_predicted"] <= 100).all()


def test_predict_with_confidence_alert_level_values(fixture_model):
    rng = np.random.RandomState(2)
    X = make_features_df(30, rng)
    trend = pd.Series(rng.uniform(0, 100, 30))
    result = predict_with_confidence(X[FEATURES_FINAL], trend, model=fixture_model)
    assert set(result["alert_level"].unique()).issubset({"green", "yellow", "red"})


def test_predict_with_confidence_missing_features_raises(fixture_model):
    X = pd.DataFrame({"capacity": [30.0], "lag_60min": [50.0]})
    trend = pd.Series([42.0])
    with pytest.raises(ValueError, match="manquantes"):
        predict_with_confidence(X, trend, model=fixture_model)


# ─────────────────────────────────────────────────────────────────────────────
# alert_level thresholds — test déterministe via mock du résidu
# ─────────────────────────────────────────────────────────────────────────────

def test_alert_level_thresholds_deterministic(fixture_model):
    """Vérifie les seuils green/yellow/red avec résidu=0 (taux = station_trend_avg)."""
    rng = np.random.RandomState(3)
    X = make_features_df(5, rng)
    # résidu=0 → taux_predicted == station_trend_avg
    trend = pd.Series([50.0, 25.0, 75.0, 5.0, 95.0])

    with patch(
        "ml.src.models.predict_model.predict_residual",
        return_value=np.zeros(5),
    ):
        result = predict_with_confidence(X[FEATURES_FINAL], trend, model=fixture_model)

    assert result["alert_level"].iloc[0] == "green"   # 50 → [30, 70]
    assert result["alert_level"].iloc[1] == "yellow"  # 25 → < 30
    assert result["alert_level"].iloc[2] == "yellow"  # 75 → > 70
    assert result["alert_level"].iloc[3] == "red"     # 5  → < 10
    assert result["alert_level"].iloc[4] == "red"     # 95 → > 90


def test_alert_level_boundary_30(fixture_model):
    """Exactement 30% → yellow (seuil strict <30 est yellow)."""
    rng = np.random.RandomState(4)
    X = make_features_df(1, rng)
    trend = pd.Series([30.0])

    with patch(
        "ml.src.models.predict_model.predict_residual",
        return_value=np.zeros(1),
    ):
        result = predict_with_confidence(X[FEATURES_FINAL], trend, model=fixture_model)

    assert result["alert_level"].iloc[0] == "green"


def test_alert_level_boundary_below_30(fixture_model):
    rng = np.random.RandomState(5)
    X = make_features_df(1, rng)
    trend = pd.Series([29.9])

    with patch(
        "ml.src.models.predict_model.predict_residual",
        return_value=np.zeros(1),
    ):
        result = predict_with_confidence(X[FEATURES_FINAL], trend, model=fixture_model)

    assert result["alert_level"].iloc[0] == "yellow"
