"""
Schémas Pydantic pour les réponses sortantes.
"""
from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

class PredictResponse(BaseModel):
    station_id: int = Field(..., description="Identifiant de la station.")
    predicted_at: datetime = Field(..., description="Date et heure de la prédiction.")  # ← renommé
    predicted_bikes: int = Field(..., ge=0)
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_version: str = Field(...)

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool

class ReadyResponse(BaseModel):
    ready: bool
    reason: str | None = Field(default=None)

# class PredictResponse(BaseModel):
#     """Réponse du endpoint POST /predict."""
#
#     station_id: int = Field(..., description="Identifiant de la station.")
#     datetime: datetime = Field(..., description="Date et heure de la prédiction.")
#     predicted_bikes: int = Field(..., description="Nombre de vélos disponibles prédit.", ge=0)
#     confidence: float = Field(..., description="Score de confiance (0.0 → 1.0).", ge=0.0, le=1.0)
#     model_version: str = Field(..., description="Version du modèle utilisé.")

# class HealthResponse(BaseModel):
#     """Réponse du endpoint GET /health."""
#
#     status: str = Field(..., description="'ok' si le service est opérationnel.")
#     version: str = Field(..., description="Version de l'API.")
#     model_loaded: bool = Field(..., description="True si le modèle MLflow est chargé.")
#
#
# class ReadyResponse(BaseModel):
#     """Réponse du endpoint GET /ready."""
#
#     ready: bool = Field(..., description="True si l'API est prête à recevoir des requêtes.")
#     reason: str | None = Field(default=None, description="Raison si not ready.")