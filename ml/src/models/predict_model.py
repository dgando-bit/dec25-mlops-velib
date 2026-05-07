"""
ml.src.models.predict_model — Inférence à partir du modèle MLflow Registry.

Fournit l'API d'inférence réutilisable par :
    - L'API FastAPI (à venir, Phase 3)
    - Le Streamlit MVP (à venir, Phase 4)
    - Tout script de batch prediction

Charge le modèle via MLflow Registry par alias (par défaut 'staging').
Le chargement se fait une seule fois (cache module-level) pour éviter le
coût de re-chargement à chaque appel d'inférence.

Usage typique (FastAPI / Streamlit) :
    from ml.src.models.predict_model import (
        load_production_model, predict_taux,
    )

    model = load_production_model()
    taux_predit = predict_taux(features_df, station_trend_avg_series, model)

CLI (test rapide en ligne de commande) :
    python -m ml.src.models.predict_model
        # → charge le modèle staging et fait une prédiction sur le test set
"""
from __future__ import annotations

import sys
from functools import lru_cache

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from sklearn.pipeline import Pipeline

from shared.config import settings
from shared.logger import get_logger
from ml.src.models._helpers import (
    FEATURES_FINAL,
    reconstruct_target,
)

logger = get_logger(__name__)


# Constantes alignées avec train_model.py
MODEL_NAME = "velib_fill_rate_predictor"
DEFAULT_ALIAS = "staging"
TRACKING_URI = f"sqlite:///{settings.repo_root / 'mlflow.db'}"


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DU MODÈLE (avec cache pour éviter re-chargement coûteux)
# ─────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def load_model_by_alias(alias: str = DEFAULT_ALIAS) -> Pipeline:
    """Charge le modèle MLflow Registry par alias.

    Le résultat est mis en cache (1 chargement par alias). Si tu déploies
    une nouvelle version, tu dois soit redémarrer le service, soit appeler
    ``load_model_by_alias.cache_clear()``.

    Args:
        alias: alias du modèle dans le Registry (ex: "staging", "production").

    Returns:
        Pipeline sklearn (imputer + XGBoost) prêt à l'inférence.

    Raises:
        mlflow.exceptions.MlflowException: si l'alias n'existe pas.
    """
    mlflow.set_tracking_uri(TRACKING_URI)
    model_uri = f"models:/{MODEL_NAME}@{alias}"
    logger.info("Chargement modèle MLflow", extra={"uri": model_uri})

    model = mlflow.sklearn.load_model(model_uri)

    # Récupère et logue la version chargée pour traçabilité
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
    """Raccourci pour charger le modèle taggué 'staging'."""
    return load_model_by_alias("staging")


def load_production_model() -> Pipeline:
    """Raccourci pour charger le modèle taggué 'production'."""
    return load_model_by_alias("production")


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION DES INPUTS (pour API FastAPI/BentoML futurs)
# ─────────────────────────────────────────────────────────────────────────────
def _validate_input(features: pd.DataFrame) -> None:
    """Vérifie que le DataFrame d'entrée contient toutes les features attendues.

    Lève une ValueError explicite si features manquantes — utile pour
    diagnostiquer les requêtes mal formées côté API.
    """
    missing = set(FEATURES_FINAL) - set(features.columns)
    if missing:
        raise ValueError(
            f"Features manquantes pour l'inférence : {sorted(missing)}.\n"
            f"Attendues : {FEATURES_FINAL}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# API D'INFÉRENCE
# ─────────────────────────────────────────────────────────────────────────────
def predict_residual(
    features: pd.DataFrame,
    model: Pipeline | None = None,
) -> np.ndarray:
    """Prédit le résidu (= taux - station_trend_avg).

    Sortie brute du modèle. Pour obtenir le taux reconstruit, utiliser
    ``predict_taux`` qui ajoute station_trend_avg.

    Args:
        features: DataFrame avec les colonnes de FEATURES_FINAL.
        model: pipeline pré-chargé (sinon charge depuis 'staging').

    Returns:
        Array numpy des résidus prédits (1 valeur par ligne).
    """
    _validate_input(features)
    if model is None:
        model = load_staging_model()
    return model.predict(features[FEATURES_FINAL])


def predict_taux(
    features: pd.DataFrame,
    station_trend_avg: pd.Series,
    model: Pipeline | None = None,
) -> pd.Series:
    """Prédit le taux de remplissage (en %).

    Reconstitue : taux_prédit = station_trend_avg + résidu_prédit, clippé [0, 100].

    Args:
        features: DataFrame avec les colonnes de FEATURES_FINAL.
        station_trend_avg: Series alignée avec features (même index/longueur),
            représentant la tendance historique de chaque station × heure × dow × month.
        model: pipeline pré-chargé (sinon charge depuis 'staging').

    Returns:
        Series de taux prédits (en %), clippés entre 0 et 100.
    """
    if len(features) != len(station_trend_avg):
        raise ValueError(
            f"features ({len(features)} lignes) et station_trend_avg "
            f"({len(station_trend_avg)} lignes) ne sont pas alignés."
        )
    residual_pred = predict_residual(features, model)
    return reconstruct_target(residual_pred, station_trend_avg)


def predict_with_confidence(
    features: pd.DataFrame,
    station_trend_avg: pd.Series,
    model: Pipeline | None = None,
) -> pd.DataFrame:
    """Prédit le taux avec un indicateur de confiance simplifié.

    XGBoost ne fournit pas nativement d'intervalle de confiance. À la place,
    on retourne :
        - taux_predicted : prédiction
        - residual_predicted : résidu brut (utile pour debug)
        - alert_level : flag basé sur seuils métier (vert/jaune/rouge)
            - rouge si taux < 10% ou > 90%
            - jaune si taux < 30% ou > 70%
            - vert sinon

    Args:
        features: DataFrame avec FEATURES_FINAL.
        station_trend_avg: Series alignée.
        model: pipeline pré-chargé.

    Returns:
        DataFrame avec colonnes ['taux_predicted', 'residual_predicted', 'alert_level'].
    """
    residual_pred = predict_residual(features, model)
    taux_pred = reconstruct_target(residual_pred, station_trend_avg)

    alert_level = pd.Series("green", index=taux_pred.index, name="alert_level")
    alert_level.loc[(taux_pred < 30) | (taux_pred > 70)] = "yellow"
    alert_level.loc[(taux_pred < 10) | (taux_pred > 90)] = "red"

    return pd.DataFrame({
        "taux_predicted":     taux_pred,
        "residual_predicted": pd.Series(residual_pred, index=taux_pred.index),
        "alert_level":        alert_level,
    })


# ─────────────────────────────────────────────────────────────────────────────
# UTILITAIRE — Invalider le cache pour forcer un rechargement
# ─────────────────────────────────────────────────────────────────────────────
def reload_model_cache() -> None:
    """Vide le cache module pour forcer un rechargement au prochain appel.

    À appeler après une nouvelle promotion staging→production, ou après un
    re-entraînement, pour que les services consommateurs prennent en compte
    la nouvelle version.
    """
    load_model_by_alias.cache_clear()
    logger.info("Cache modèle vidé")


# ─────────────────────────────────────────────────────────────────────────────
# CLI — TEST RAPIDE
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """Test rapide : charge le modèle staging, prédit sur le test set, affiche."""
    try:
        # Charger le test set
        test_path = settings.test_path
        if not test_path.exists():
            logger.error(
                "Test set absent",
                extra={"path": str(test_path)},
            )
            return 2

        test = pd.read_parquet(test_path)
        sample = test.head(5)

        logger.info("Chargement du modèle 'staging'...")
        model = load_staging_model()

        logger.info("Prédiction sur 5 lignes du test set...")
        result = predict_with_confidence(
            features=sample[FEATURES_FINAL],
            station_trend_avg=sample["station_trend_avg"],
            model=model,
        )

        # Affichage formaté
        result["taux_real"] = sample["taux"].values
        result["error"] = (result["taux_predicted"] - result["taux_real"]).abs().round(2)
        result = result.round(2)

        print("\n=== Test prédiction sur 5 lignes du test set ===\n")
        print(result.to_string())
        print(f"\nMAE sur ces 5 points : {result['error'].mean():.2f}%")
        return 0

    except Exception as e:  # noqa: BLE001
        logger.exception(
            "Échec predict_model",
            extra={"error_type": type(e).__name__},
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
