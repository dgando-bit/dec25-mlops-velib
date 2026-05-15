"""
Schémas Pydantic pour les requêtes entrantes.
"""
from __future__ import annotations
from datetime import datetime as DateTime
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Corps de la requête POST /predict."""

    station_id: int = Field(
        ...,
        description="Identifiant de la station Vélib'.",
    )
    predicted_at: DateTime = Field(
        ...,
        description="Date et heure de la prédiction (ISO 8601).",
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "station_id": 10042,
                "predicted_at": "2026-01-15T08:30:00",
            }
        }
    }
