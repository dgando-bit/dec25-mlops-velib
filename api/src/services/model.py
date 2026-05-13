"""
velib_api.services.model — Chargement MLflow et logique de prédiction.

Toute interaction avec MLflow est isolée ici.
Les routers ne connaissent pas MLflow — ils appellent uniquement ce service.
"""
from __future__ import annotations

import pandas as pd
import mlflow
import mlflow.pyfunc

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)
# settings = get_settings()

# Modèle chargé une seule fois au démarrage (état global du service)
_model: mlflow.pyfunc.PyFuncModel | None = None
_model_version: str = "unknown"


def load_model() -> None:
    """Charge le modèle depuis le Model Registry MLflow.

    Appelé une seule fois dans le lifespan de l'app FastAPI.
    Modifie l'état global _model et _model_version.

    Raises:
        RuntimeError: si le modèle ne peut pas être chargé.
    """
    global _model, _model_version

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    model_uri = f"models:/{settings.registered_model_name}/Production"

    logger.info(
        "Chargement du modèle MLflow — %s @ %s",
        settings.registered_model_name,
        settings.mlflow_tracking_uri,
    )

    try:
        _model = mlflow.pyfunc.load_model(model_uri)
        # Récupère la version depuis les métadonnées MLflow
        client = mlflow.MlflowClient()
        versions = client.get_latest_versions(
            settings.registered_model_name, stages=["Production"]
        )
        _model_version = versions[0].version if versions else "unknown"
        logger.info("Modèle chargé — version %s", _model_version)

    except Exception as e:
        logger.error("Échec du chargement du modèle — %s", str(e))
        raise RuntimeError(f"Impossible de charger le modèle MLflow : {e}") from e


def get_model() -> mlflow.pyfunc.PyFuncModel:
    """Retourne le modèle chargé.

    Raises:
        RuntimeError: si le modèle n'est pas encore chargé.
    """
    if _model is None:
        raise RuntimeError("Modèle non chargé. Vérifier le démarrage de l'API.")
    return _model


def get_model_version() -> str:
    """Retourne la version du modèle actuellement chargé."""
    return _model_version


def is_model_loaded() -> bool:
    """Indique si le modèle est prêt à recevoir des prédictions."""
    return _model is not None


def predict(station_id: int, dt: str) -> dict:
    """Effectue une prédiction pour une station et une datetime donnée.

    Args:
        station_id: identifiant de la station Vélib'.
        dt: datetime de la prédiction (ISO 8601).

    Returns:
        Dictionnaire avec predicted_bikes, confidence, model_version.

    Raises:
        RuntimeError: si le modèle n'est pas chargé.
        ValueError: si les données d'entrée sont invalides.
    """
    model = get_model()

    # Construction du DataFrame d'entrée attendu par le modèle
    input_df = pd.DataFrame([{
        "station_id": station_id,
        "datetime": dt,
    }])

    logger.debug("Prédiction — station %d @ %s", station_id, dt)

    raw = model.predict(input_df)

    # Adaptation selon le format de sortie de ton modèle
    predicted_bikes = int(raw[0]) if hasattr(raw, "__len__") else int(raw)
    confidence = float(raw[1]) if (hasattr(raw, "__len__") and len(raw) > 1) else 1.0

    return {
        "predicted_bikes": max(0, predicted_bikes),
        "confidence": min(1.0, max(0.0, confidence)),
        "model_version": get_model_version(),
    }