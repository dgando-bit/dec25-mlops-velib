"""
velib_api.inference — Logique d'inférence auto-contenue.

Remplace la dépendance vers ml.src.models.* : l'API est un service
indépendant qui ne doit pas embarquer le module de training.

Le contrat de features (FEATURES_FINAL) est intentionnellement dupliqué
depuis ml/src/models/_helpers.py — train et serve sont des services
séparés; la liste figée ici est le contrat d'interface de l'API.
"""
from __future__ import annotations

from functools import lru_cache

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

MODEL_NAME = "velib_fill_rate_predictor"
DEFAULT_ALIAS = "staging"

FEATURES_FINAL: list[str] = [
    "capacity",
    "capacity_group",
    "morning_evening_ratio",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month",
    "is_peak_hour",
    "is_friday_evening",
    "is_monday_morning",
    "is_holiday",
    "is_vacation",
    "apparent_temperature",
    "temp_anomalie",
    "weather_severity",
    "is_frozen",
    "is_stormy",
    "lag_60min",
    "lag_240min",
    "lag_res_240min",
    "lat",
    "lon",
    "hour",
]


def reconstruct_target(
    residual_pred: np.ndarray | pd.Series,
    station_trend_avg: pd.Series,
) -> pd.Series:
    """Reconstruit taux_prédit = station_trend_avg + résidu_prédit, clippé [0, 100]."""
    residual_pred = np.asarray(residual_pred)
    station_trend = np.asarray(station_trend_avg)
    return pd.Series(
        np.clip(station_trend + residual_pred, 0, 100),
        index=station_trend_avg.index if hasattr(station_trend_avg, "index") else None,
        name="taux_predicted",
    )


@lru_cache(maxsize=4)
def load_model_by_alias(alias: str = DEFAULT_ALIAS) -> Pipeline:
    """Charge le modèle depuis le MLflow Registry par alias (résultat mis en cache)."""
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = f"models:/{MODEL_NAME}@{alias}"
    logger.info("Chargement modèle MLflow", extra={"uri": model_uri})
    model = mlflow.sklearn.load_model(model_uri)
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(MODEL_NAME, alias)
    logger.info(
        "Modèle chargé",
        extra={
            "model_name": MODEL_NAME,
            "alias": alias,
            "version": version_info.version,
            "run_id": version_info.run_id,
        },
    )
    return model


def load_staging_model() -> Pipeline:
    return load_model_by_alias("staging")


def predict_with_confidence(
    features: pd.DataFrame,
    station_trend_avg: pd.Series,
    model: Pipeline | None = None,
) -> pd.DataFrame:
    """Prédit taux + résidu + alert_level pour N stations."""
    if model is None:
        model = load_staging_model()

    missing = set(FEATURES_FINAL) - set(features.columns)
    if missing:
        raise ValueError(f"Features manquantes : {sorted(missing)}")

    residual_pred = model.predict(features[FEATURES_FINAL])
    taux_pred = reconstruct_target(residual_pred, station_trend_avg)

    alert_level = pd.Series("green", index=taux_pred.index, name="alert_level")
    alert_level.loc[(taux_pred < 30) | (taux_pred > 70)] = "yellow"
    alert_level.loc[(taux_pred < 10) | (taux_pred > 90)] = "red"

    return pd.DataFrame({
        "taux_predicted":     taux_pred,
        "residual_predicted": pd.Series(residual_pred, index=taux_pred.index),
        "alert_level":        alert_level,
    })
