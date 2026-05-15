"""
velib_api.routers.health — Endpoints de santé et de disponibilité.

GET /health  → vérifie que le service tourne
GET /ready   → vérifie que le modèle est chargé et prêt
"""
from __future__ import annotations

from fastapi import APIRouter
from api.src.schemas.response import HealthResponse, ReadyResponse
from api.src.services.model import is_model_loaded

router = APIRouter(tags=["Health"])

API_VERSION = "0.1.0"


@router.get("/health", response_model=HealthResponse, summary="Santé du service")
def health() -> HealthResponse:
    """Vérifie que l'API est opérationnelle."""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        model_loaded=is_model_loaded(),
    )


@router.get("/ready", response_model=ReadyResponse, summary="Disponibilité du service")
def ready() -> ReadyResponse:
    """Vérifie que l'API est prête à recevoir des prédictions."""
    if not is_model_loaded():
        return ReadyResponse(
            ready=False,
            reason="Modèle MLflow non chargé.",
        )
    return ReadyResponse(ready=True)