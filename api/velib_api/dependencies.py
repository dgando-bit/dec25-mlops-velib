"""
velib_api.dependencies — Dépendances injectables FastAPI.

Centralise la logique de chargement / accès au modèle MLflow.
FastAPI résout ces dépendances via le système ``Depends`` qui injecte
automatiquement les objets dans les routes.

Le modèle est chargé une seule fois au démarrage (event handler ``startup``
défini dans main.py) et conservé en cache via ``functools.lru_cache``
dans ml.src.models.predict_model. Toutes les requêtes suivantes accèdent
au modèle déjà en mémoire — pas de I/O répétée.

Pour invalider le cache (après promotion d'une nouvelle version) :
    POST /model/reload  — TODO Phase 3
    Ou : redémarrer le service (Docker/Kubernetes pod restart).
"""
from __future__ import annotations

from sklearn.pipeline import Pipeline

from shared.config import settings
from shared.logger import get_logger
from ml.src.models.predict_model import load_staging_model, load_model_by_alias
from ml.src.models._helpers import FEATURES_FINAL

logger = get_logger(__name__)


# Réexport d'éléments partagés pour ne pas avoir à les ré-importer dans main.py
__all__ = ["get_settings", "get_model", "preload_model", "FEATURES_FINAL"]


def get_settings():
    """Dépendance pour injecter les settings dans une route.

    Utilisation :
        @app.get("/foo")
        def foo(s = Depends(get_settings)):
            return {"hf_repo": s.hf_repo}
    """
    return settings


def get_model() -> Pipeline:
    """Dépendance pour injecter le modèle chargé.

    Profite du ``lru_cache`` posé sur ``load_model_by_alias`` :
    - Premier appel : chargement (3-5 secondes)
    - Appels suivants : retour instantané du cache

    Returns:
        Pipeline sklearn (imputer + XGBoost) déjà entraîné, prêt à .predict().
    """
    return load_staging_model()


def preload_model() -> dict:
    """Pré-charge le modèle au démarrage du service.

    Appelé depuis l'event handler ``startup`` de FastAPI dans main.py.
    Le but est d'éviter que la première requête HTTP paie le coût du
    chargement (3-5s) — au lieu, c'est le démarrage du service qui le paie.

    Returns:
        dict avec les métadonnées du modèle chargé (pour log).
    """
    from mlflow.tracking import MlflowClient
    import mlflow

    logger.info("Pré-chargement du modèle 'staging' au démarrage de l'API...")
    model = load_staging_model()

    # Récupère les métadonnées pour log + endpoint /model/info
    mlflow.set_tracking_uri(f"sqlite:///{settings.repo_root / 'mlflow.db'}")
    client = MlflowClient()
    version_info = client.get_model_version_by_alias(
        "velib_fill_rate_predictor", "staging"
    )

    metadata = {
        "model_name": "velib_fill_rate_predictor",
        "alias": "staging",
        "version": str(version_info.version),
        "run_id": version_info.run_id,
        "framework": "xgboost",
        "n_features": len(FEATURES_FINAL),
    }

    logger.info("Modèle pré-chargé avec succès", extra=metadata)
    return metadata
