"""
velib_api.dependencies — Injection de dépendances FastAPI.

Centralise les dépendances injectées via Depends() dans les routers.
"""
from __future__ import annotations

import mlflow.pyfunc
from fastapi import HTTPException, status

from shared.config import Settings, settings
from api.src.services.model import get_model


def dep_settings() -> Settings:
    """Injecte les settings du projet."""
    return settings


def dep_model() -> mlflow.pyfunc.PyFuncModel:
    """Injecte le modèle MLflow chargé.

    Raises:
        HTTPException 503 si le modèle n'est pas disponible.
    """
    try:
        return get_model()
    except RuntimeError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e),
        ) from e