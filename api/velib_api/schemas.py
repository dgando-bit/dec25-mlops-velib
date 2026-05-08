"""
api.schemas — Modèles Pydantic pour les requêtes et réponses de l'API.

Format des inputs :
    Le client envoie les 24 features pré-calculées par le pipeline
    de feature engineering (build_features.py + station_trend_avg).
    L'API ne fait QUE l'inférence — pas de calcul de lags, de météo, etc.

Validation : strict.
    Une feature manquante ou un type invalide → 422 Unprocessable Entity.
    C'est volontaire pour détecter les erreurs côté client tôt.

Cohérence des features :
    La liste FEATURES_FINAL doit correspondre à celle de
    ml/src/models/_helpers.py. Si tu changes l'une, change l'autre.
    On préfère dupliquer le contrat ici plutôt que d'importer ml/ dans
    schemas.py (séparation des préoccupations).
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# REQUÊTES — INPUTS
# ─────────────────────────────────────────────────────────────────────────────
class StationFeatures(BaseModel):
    """Features pré-calculées pour une prédiction unitaire.

    Les 24 features attendues par le modèle XGBoost + station_trend_avg
    nécessaire à la reconstruction du taux à partir du résidu prédit.
    """

    model_config = ConfigDict(
        extra="forbid",  # rejet des champs inconnus → 422
        json_schema_extra={
            "example": {
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
        },
    )

    # Capacité et profil station
    capacity: int = Field(..., ge=0, description="Nombre total de places de la station")
    capacity_group: int = Field(..., ge=0, le=3, description="Groupe taille (0=petite, 3=très grande)")
    morning_evening_ratio: float = Field(..., ge=0.0, description="Ratio taux moyen matin/soir")

    # Encodage cyclique du temps
    hour_sin: float = Field(..., ge=-1.0, le=1.0)
    hour_cos: float = Field(..., ge=-1.0, le=1.0)
    dow_sin: float = Field(..., ge=-1.0, le=1.0)
    dow_cos: float = Field(..., ge=-1.0, le=1.0)
    month: int = Field(..., ge=1, le=12)

    # Flags temporels
    is_peak_hour: int = Field(..., ge=0, le=1)
    is_friday_evening: int = Field(..., ge=0, le=1)
    is_monday_morning: int = Field(..., ge=0, le=1)

    # Calendaires
    is_holiday: int = Field(..., ge=0, le=1)
    is_vacation: int = Field(..., ge=0, le=1)

    # Météo
    apparent_temperature: float = Field(..., description="Température ressentie en °C")
    temp_anomalie: float = Field(..., description="Écart à la normale mensuelle (°C)")
    weather_severity: int = Field(..., ge=0, le=4, description="Sévérité météo 0-4 (codes WMO)")
    is_frozen: int = Field(..., ge=0, le=1)
    is_stormy: int = Field(..., ge=0, le=1)

    # Lags
    lag_60min: float = Field(..., ge=0.0, le=100.0, description="Taux il y a 60min")
    lag_240min: float = Field(..., ge=0.0, le=100.0, description="Taux il y a 240min")
    lag_res_240min: float = Field(..., description="lag_240min - station_trend_avg")

    # Géographie
    lat: float = Field(..., ge=-90.0, le=90.0)
    lon: float = Field(..., ge=-180.0, le=180.0)

    # Coordonnée temporelle additionnelle
    hour: int = Field(..., ge=0, le=23)

    # Tendance historique (pour reconstruction du taux à partir du résidu)
    station_trend_avg: float = Field(..., ge=0.0, le=100.0,
                                     description="Tendance historique de la station × dow × hour")


class BatchPredictionRequest(BaseModel):
    """Requête de prédiction batch — N stations simultanément."""

    model_config = ConfigDict(extra="forbid")

    items: list[StationFeatures] = Field(
        ...,
        min_length=1,
        max_length=2000,  # protège contre des requêtes énormes (1492 stations + marge)
        description="Liste des features par station à prédire.",
    )


# ─────────────────────────────────────────────────────────────────────────────
# RÉPONSES — OUTPUTS
# ─────────────────────────────────────────────────────────────────────────────
class PredictionResponse(BaseModel):
    """Réponse pour une prédiction unitaire.

    Contient les 3 niveaux d'information demandés :
        - residual_predicted : sortie brute du modèle (utile pour debug)
        - taux_predicted     : taux reconstruit (cible métier)
        - alert_level        : niveau d'alerte basé sur seuils opérateur
    """

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "residual_predicted": -2.34,
                "taux_predicted": 39.66,
                "alert_level": "green",
            }
        }
    )

    residual_predicted: float = Field(..., description="Résidu prédit par le modèle")
    taux_predicted: float = Field(..., ge=0.0, le=100.0,
                                  description="Taux de remplissage prédit (%)")
    alert_level: Literal["green", "yellow", "red"] = Field(
        ..., description=(
            "Niveau d'alerte opérationnel : "
            "green (30-70%), yellow (10-30% ou 70-90%), red (<10% ou >90%)"
        ),
    )


class BatchPredictionResponse(BaseModel):
    """Réponse pour une prédiction batch."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "n_predictions": 3,
                "predictions": [
                    {"residual_predicted": -2.34, "taux_predicted": 39.66, "alert_level": "green"},
                    {"residual_predicted": 5.10, "taux_predicted": 78.50, "alert_level": "yellow"},
                    {"residual_predicted": 12.20, "taux_predicted": 92.00, "alert_level": "red"},
                ],
            }
        }
    )

    n_predictions: int = Field(..., ge=0)
    predictions: list[PredictionResponse]


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS DIVERS
# ─────────────────────────────────────────────────────────────────────────────
class HealthResponse(BaseModel):
    """Réponse de l'endpoint /health."""

    status: Literal["ok", "degraded"] = "ok"
    api_version: str = Field(..., description="Version de l'API")
    model_loaded: bool = Field(..., description="True si le modèle MLflow est chargé en cache")


class ModelInfoResponse(BaseModel):
    """Réponse de l'endpoint /model/info — métadonnées du modèle chargé."""

    model_config = ConfigDict(
        protected_namespaces=(),  # désactive l'avertissement Pydantic sur 'model_'
        json_schema_extra={
            "example": {
                "model_name": "velib_fill_rate_predictor",
                "alias": "staging",
                "version": "3",
                "run_id": "6a9e0bf50616487abb11db1a5dc31469",
                "framework": "xgboost",
                "n_features": 24,
            }
        },
    )

    model_name: str
    alias: str
    version: str
    run_id: str
    framework: str = "xgboost"
    n_features: int
