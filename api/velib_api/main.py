"""
velib_api.main — Application FastAPI d'inférence Vélib'.

5 endpoints :
    GET  /                    — racine, redirige vers /docs
    GET  /health              — santé du service (Docker healthcheck, K8s liveness)
    GET  /model/info          — métadonnées du modèle chargé
    POST /predict             — prédiction unitaire (1 station)
    POST /predict/batch       — prédiction batch (N stations)

Le modèle MLflow est chargé en mémoire au démarrage du service via
l'event handler ``startup``. Pas de I/O par requête (sauf le calcul XGBoost).

Documentation auto-générée par FastAPI :
    http://localhost:8000/docs    — Swagger UI (testable)
    http://localhost:8000/redoc   — ReDoc (lisible)

Lancement local :
    uvicorn velib_api.main:app --reload --port 8000

Lancement production :
    uvicorn velib_api.main:app --host 0.0.0.0 --port 8000 --workers 4
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Annotated

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import RedirectResponse
from sklearn.pipeline import Pipeline

from shared.logger import get_logger
from velib_api.inference import FEATURES_FINAL, predict_with_confidence
from velib_api import __version__
from velib_api.dependencies import get_model, preload_model
from velib_api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    StationFeatures,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — pré-chargement du modèle au démarrage
# ─────────────────────────────────────────────────────────────────────────────
# State partagé entre les routes pour exposer les métadonnées du modèle
# (utile pour /model/info sans devoir re-questionner MLflow à chaque appel)
_MODEL_METADATA: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie de l'application — exécuté au startup puis au shutdown.

    Au startup : pré-charge le modèle (LRU cache rempli, prochaines requêtes rapides).
    Au shutdown : rien de spécial pour l'instant.
    """
    logger.info("API démarre...", extra={"version": __version__})
    try:
        metadata = preload_model()
        _MODEL_METADATA.update(metadata)
        logger.info("API prête à recevoir des requêtes.")
    except Exception as e:  # noqa: BLE001
        # Si le modèle ne charge pas, on continue quand même : /health renverra
        # 'degraded' et l'utilisateur verra l'erreur sur /predict.
        logger.exception(
            "Échec du pré-chargement du modèle — l'API démarre en mode dégradé",
            extra={"error_type": type(e).__name__},
        )

    yield  # le service tourne ici

    logger.info("API s'arrête.")


# ─────────────────────────────────────────────────────────────────────────────
# APP FASTAPI
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vélib' Fill Rate Predictor API",
    description=(
        "API d'inférence pour le modèle XGBoost de prédiction du taux de "
        "remplissage des stations Vélib' Paris. Charge le modèle depuis "
        "le MLflow Registry par alias."
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — RACINE & MÉTA
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/", include_in_schema=False)
def root():
    """Racine — redirige vers la doc Swagger."""
    return RedirectResponse(url="/docs", status_code=status.HTTP_302_FOUND)


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Santé du service",
    description=(
        "Endpoint léger pour Docker healthcheck / Kubernetes liveness. "
        "Renvoie 'ok' si le modèle est chargé, 'degraded' sinon."
    ),
)
def health() -> HealthResponse:
    is_loaded = bool(_MODEL_METADATA)
    return HealthResponse(
        status="ok" if is_loaded else "degraded",
        api_version=__version__,
        model_loaded=is_loaded,
    )


@app.get(
    "/model/info",
    response_model=ModelInfoResponse,
    summary="Métadonnées du modèle chargé",
    description=(
        "Renvoie le nom, l'alias, la version, le run_id MLflow du modèle "
        "actuellement servi par l'API."
    ),
    responses={
        503: {"description": "Modèle non chargé (service en mode dégradé)"},
    },
)
def model_info() -> ModelInfoResponse:
    if not _MODEL_METADATA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Modèle non chargé. Vérifier que mlflow.db existe et qu'un "
                "modèle 'velib_fill_rate_predictor' avec alias 'staging' est "
                "enregistré dans le Registry."
            ),
        )
    return ModelInfoResponse(**_MODEL_METADATA)


# ─────────────────────────────────────────────────────────────────────────────
# ENDPOINTS — PRÉDICTION
# ─────────────────────────────────────────────────────────────────────────────
def _predict_single(
    features: StationFeatures, model: Pipeline,
) -> PredictionResponse:
    """Prédit pour une seule station. Mutualisé entre /predict et /predict/batch."""
    # Convertit le Pydantic en DataFrame (1 ligne, 25 colonnes)
    record = features.model_dump()
    station_trend = record["station_trend_avg"]

    # Construit le DataFrame d'entrée du modèle (24 features, sans station_trend_avg)
    df = pd.DataFrame([record])[FEATURES_FINAL]

    # Inférence + reconstruction + alert level
    result = predict_with_confidence(
        features=df,
        station_trend_avg=pd.Series([station_trend]),
        model=model,
    )

    row = result.iloc[0]
    return PredictionResponse(
        residual_predicted=round(float(row["residual_predicted"]), 2),
        taux_predicted=round(float(row["taux_predicted"]), 2),
        alert_level=str(row["alert_level"]),
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Prédiction unitaire (1 station)",
    description=(
        "Prédit le taux de remplissage et le niveau d'alerte pour une station "
        "à partir des 24 features pré-calculées + station_trend_avg.\n\n"
        "Le client est responsable de calculer les features (lags, météo, "
        "encodages cycliques, etc.) — voir `ml/src/features/build_features.py`."
    ),
    responses={
        503: {"description": "Modèle non chargé"},
        500: {"description": "Erreur d'inférence interne"},
    },
)
def predict(
    features: StationFeatures,
    model: Annotated[Pipeline, Depends(get_model)],
) -> PredictionResponse:
    if not _MODEL_METADATA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé.",
        )
    try:
        return _predict_single(features, model)
    except Exception as e:  # noqa: BLE001
        logger.exception("Échec prediction unitaire",
                         extra={"error_type": type(e).__name__})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'inférence : {type(e).__name__}",
        )


@app.post(
    "/predict/batch",
    response_model=BatchPredictionResponse,
    summary="Prédiction batch (N stations)",
    description=(
        "Prédit pour N stations en une requête. Plus efficace que N appels "
        "à /predict (1 seule désérialisation, 1 seul .predict() XGBoost). "
        "Limite : 2000 stations par requête."
    ),
    responses={
        503: {"description": "Modèle non chargé"},
        500: {"description": "Erreur d'inférence interne"},
    },
)
def predict_batch(
    request: BatchPredictionRequest,
    model: Annotated[Pipeline, Depends(get_model)],
) -> BatchPredictionResponse:
    if not _MODEL_METADATA:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non chargé.",
        )

    try:
        # Construit un DataFrame de N lignes en une passe (plus efficace que une-par-une)
        records = [item.model_dump() for item in request.items]
        df_full = pd.DataFrame(records)
        station_trend = df_full["station_trend_avg"]
        df_features = df_full[FEATURES_FINAL]

        result = predict_with_confidence(
            features=df_features,
            station_trend_avg=station_trend,
            model=model,
        )

        predictions = [
            PredictionResponse(
                residual_predicted=round(float(row["residual_predicted"]), 2),
                taux_predicted=round(float(row["taux_predicted"]), 2),
                alert_level=str(row["alert_level"]),
            )
            for _, row in result.iterrows()
        ]

        return BatchPredictionResponse(
            n_predictions=len(predictions),
            predictions=predictions,
        )
    except Exception as e:  # noqa: BLE001
        logger.exception("Échec prediction batch",
                         extra={"n_items": len(request.items),
                                "error_type": type(e).__name__})
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur d'inférence batch : {type(e).__name__}",
        )
