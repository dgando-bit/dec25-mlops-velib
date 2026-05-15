"""
velib_api.routers.predict — Endpoint de prédiction.

POST /predict → retourne le nombre de vélos disponibles prédit
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from api.src.schemas.request import PredictRequest
from api.src.schemas.response import PredictResponse
from api.src.services import model as model_service

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    summary="Prédiction de disponibilité Vélib'",
    status_code=status.HTTP_200_OK,
)
def predict(request: PredictRequest) -> PredictResponse:
    """Prédit le nombre de vélos disponibles pour une station et une datetime.

    - **station_id** : identifiant de la station Vélib'
    - **datetime** : date et heure souhaitées (ISO 8601)
    """
    if not model_service.is_model_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Modèle non disponible. Réessayer dans quelques secondes.",
        )

    try:
        result = model_service.predict(
            station_id=request.station_id,
            dt=request.datetime.isoformat(),
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur de prédiction : {e}",
        ) from e

    return PredictResponse(
	    station_id=request.station_id,
	    predicted_at=request.datetime,  # ← renommé
	    **result,
    )