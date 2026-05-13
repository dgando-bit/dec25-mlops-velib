"""
ml.src.models.train_model — Entraînement XGBoost + tracking MLflow.

Lit les datasets train/test produits par ``build_features``, entraîne XGBoost
sur la cible résiduelle, évalue, logue tout dans MLflow, et promeut
automatiquement le modèle en alias 'staging' du Model Registry.

Pipeline :
    1. Chargement train + test parquets (data/processed/)
    2. Configuration MLflow (URI depuis settings + experiment + autolog)
    3. Entraînement du Pipeline (imputer + XGBoost)
    4. Évaluation (résidu + taux reconstruit)
    5. Génération des 3 plots (importance, résidus, predictions_vs_actual)
    6. Logging MLflow (params + metrics + artifacts + model)
    7. Enregistrement dans le Model Registry sous le nom 'velib_fill_rate_predictor'
    8. Promotion en alias 'staging' automatique
    9. Export metrics.json pour DVC

Pipeline DVC :
    Stage  : train_model
    Entrée : data/processed/{train,test}_preprocessed.parquet
    Sorties :
        - mlruns/                              (métadonnées MLflow)
        - mlartifacts/                         (artefacts MLflow)

        - data/outputs/metrics.json            (métriques pour DVC)
        - data/outputs/feature_importance.png  (plot DVC)
        - data/outputs/residuals_distribution.png
        - data/outputs/predictions_vs_actual.png

Usage :
    # En CLI
    python -m ml.src.models.train_model

    # En import
    from ml.src.models.train_model import train_model
    metrics = train_model()
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.exceptions import RestException
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

from shared.config import settings
from shared.logger import get_logger
from ml.src.models._helpers import (
    FEATURES_FINAL,
    HYPERPARAMS_XGB,
    build_xgboost,
    compute_metrics,
    plot_feature_importance,
    plot_predictions_vs_actual,
    plot_residuals_distribution,
    reconstruct_target,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES MLflow
# ─────────────────────────────────────────────────────────────────────────────
EXPERIMENT_NAME = "velib_fill_rate"
MODEL_NAME = "velib_fill_rate_predictor"
STAGING_ALIAS = "staging"

TRACKING_URI = settings.mlflow_tracking_uri
ARTIFACT_ROOT = settings.mlflow_artifact_uri


# ─────────────────────────────────────────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────────────────────────────────────────
def _load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge train et test depuis data/processed/.

    Vérifie que toutes les FEATURES_FINAL + colonnes auxiliaires sont présentes.
    """
    train_path = settings.train_path
    test_path = settings.test_path

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(
            f"Datasets train/test absents. Lancer d'abord :\n"
            f"  python -m ml.src.features.build_features"
        )

    train = pd.read_parquet(train_path)
    test = pd.read_parquet(test_path)

    # Vérifier que les features attendues sont présentes
    missing_train = set(FEATURES_FINAL) - set(train.columns)
    missing_test = set(FEATURES_FINAL) - set(test.columns)
    if missing_train or missing_test:
        raise ValueError(
            f"Features manquantes — train: {missing_train}, test: {missing_test}"
        )

    # Vérifier les colonnes auxiliaires nécessaires
    required_aux = {"residual_target", "taux", "station_trend_avg"}
    missing_aux = required_aux - set(train.columns)
    if missing_aux:
        raise ValueError(f"Colonnes auxiliaires manquantes : {missing_aux}")

    logger.info(
        "Train/test chargés",
        extra={
            "train_rows": len(train),
            "test_rows":  len(test),
            "n_features": len(FEATURES_FINAL),
        },
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# SETUP MLflow
# ─────────────────────────────────────────────────────────────────────────────
def _setup_mlflow() -> None:
    """Configure MLflow : URI depuis settings + experiment + crée si inexistant."""
    mlflow.set_tracking_uri(TRACKING_URI)

    # Créer l'expérience si elle n'existe pas (avec artifact_root explicite)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        client.create_experiment(
            name=EXPERIMENT_NAME, artifact_location=ARTIFACT_ROOT
        )
        logger.info(
            "Experiment MLflow créé",
            extra={"name": EXPERIMENT_NAME, "artifact_root": ARTIFACT_ROOT},
        )
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(
        "MLflow configuré",
        extra={"tracking_uri": TRACKING_URI, "experiment": EXPERIMENT_NAME},
    )


# ─────────────────────────────────────────────────────────────────────────────
# PROMOTION EN ALIAS 'staging'
# ─────────────────────────────────────────────────────────────────────────────
def _promote_to_staging(client: MlflowClient, model_uri: str, run_id: str) -> int:
    """Enregistre le modèle dans le Registry et lui pose l'alias 'staging'.

    Utilise le système d'alias (recommandé par MLflow ≥ 2.9), pas les stages
    dépréciés (None/Staging/Production).

    Args:
        client: MlflowClient instancié.
        model_uri: URI du modèle (ex: "runs:/abc123/model").
        run_id: ID du run MLflow.

    Returns:
        Numéro de version créée dans le Registry.
    """
    # Étape 1 : créer le RegisteredModel s'il n'existe pas
    try:
        client.create_registered_model(MODEL_NAME)
        logger.info(
            "Registered model créé",
            extra={"model_name": MODEL_NAME},
        )
    except (RestException, mlflow.exceptions.MlflowException) as e:
        if "already exists" not in str(e).lower():
            raise
        # Existe déjà : on continue
        logger.info(
            "Registered model existe déjà — ajout d'une nouvelle version",
            extra={"model_name": MODEL_NAME},
        )

    # Étape 2 : créer une nouvelle version
    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id,
    )
    version_int = int(model_version.version)
    logger.info(
        "Model version créée",
        extra={"model_name": MODEL_NAME, "version": version_int},
    )

    # Étape 3 : poser l'alias 'staging' sur cette version
    client.set_registered_model_alias(
        name=MODEL_NAME,
        alias=STAGING_ALIAS,
        version=version_int,
    )
    logger.info(
        "Alias 'staging' posé",
        extra={"model_name": MODEL_NAME, "version": version_int},
    )

    return version_int


# ─────────────────────────────────────────────────────────────────────────────
# ENREGISTREMENT DES MÉTRIQUES POUR DVC
# ─────────────────────────────────────────────────────────────────────────────
def _save_metrics_for_dvc(metrics: dict[str, float], run_id: str) -> None:
    """Sauve les métriques en JSON pour que DVC puisse les versionner.

    DVC utilise ces fichiers pour `dvc metrics show` et `dvc metrics diff`.
    """
    out_path = settings.repo_root / "data" / "outputs" / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        **metrics,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    logger.info("Metrics JSON écrit", extra={"path": str(out_path)})


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────────────────────
def train_model() -> dict[str, float]:
    """Entraîne XGBoost, logue MLflow, promeut en staging.

    Returns:
        dict des métriques calculées sur le test set.

    Raises:
        FileNotFoundError: si les parquets train/test n'existent pas.
        ValueError: si les features attendues sont absentes.
    """
    settings.ensure_directories()

    # ── 1. Chargement ──────────────────────────────────────────────────────
    train, test = _load_train_test()

    X_train = train[FEATURES_FINAL]
    y_train_residual = train["residual_target"]
    X_test = test[FEATURES_FINAL]
    y_test_residual = test["residual_target"]
    y_test_taux = test["taux"]
    test_station_trend = test["station_trend_avg"]

    # ── 2. Setup MLflow ────────────────────────────────────────────────────
    _setup_mlflow()
    client = MlflowClient()

    with mlflow.start_run(run_name="xgboost_residual") as run:
        run_id = run.info.run_id
        logger.info(
            "Run MLflow démarré",
            extra={"run_id": run_id, "experiment": EXPERIMENT_NAME},
        )

        # ── Tags pour traçabilité ──────────────────────────────────────────
        # Lire le mode station_trend_avg depuis le metadata produit par build_features
        metadata_path = settings.processed_data_dir / "build_metadata.json"
        station_trend_mode = "unknown"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                station_trend_mode = metadata.get("station_trend_mode", "unknown")
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(
                    "Échec lecture build_metadata.json",
                    extra={"error": str(e)},
                )

        mlflow.set_tags({
            "framework": "xgboost",
            "target": "residual_target",
            "n_features": len(FEATURES_FINAL),
            "git_repo": "dec25-mlops-velib",
            "station_trend_mode": station_trend_mode,
        })

        # ── 3. Entraînement ────────────────────────────────────────────────
        logger.info("Entraînement XGBoost démarré",
                    extra={"n_train": len(X_train)})
        pipeline = build_xgboost(HYPERPARAMS_XGB)
        pipeline.fit(X_train, y_train_residual)
        logger.info("Entraînement terminé")

        # Logger les hyperparamètres explicitement (au cas où autolog ne les capte pas)
        mlflow.log_params({f"xgb_{k}": v for k, v in HYPERPARAMS_XGB.items()})
        mlflow.log_param("features_count", len(FEATURES_FINAL))
        mlflow.log_param("train_rows", len(X_train))
        mlflow.log_param("test_rows", len(X_test))

        # ── 4. Évaluation ──────────────────────────────────────────────────
        y_pred_residual = pipeline.predict(X_test)
        y_pred_taux = reconstruct_target(y_pred_residual, test_station_trend)

        metrics = compute_metrics(
            y_true_residual=y_test_residual,
            y_pred_residual=y_pred_residual,
            y_true_taux=y_test_taux,
            y_pred_taux=y_pred_taux,
        )
        mlflow.log_metrics(metrics)
        logger.info(
            "Évaluation terminée",
            extra={k: round(v, 3) for k, v in metrics.items()},
        )

        # ── 5. Plots ───────────────────────────────────────────────────────
        plots_dir = settings.repo_root / "data" / "outputs"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Récupérer le modèle XGBoost natif (pas le pipeline) pour feature_importances_
        xgb_model = pipeline.named_steps["model"]

        fig_importance = plot_feature_importance(xgb_model, top_n=20)
        path_importance = plots_dir / "feature_importance.png"
        fig_importance.savefig(path_importance, dpi=120)
        mlflow.log_artifact(str(path_importance), artifact_path="plots")

        fig_residuals = plot_residuals_distribution(y_test_residual, y_pred_residual)
        path_residuals = plots_dir / "residuals_distribution.png"
        fig_residuals.savefig(path_residuals, dpi=120)
        mlflow.log_artifact(str(path_residuals), artifact_path="plots")

        fig_pva = plot_predictions_vs_actual(y_test_taux, y_pred_taux)
        path_pva = plots_dir / "predictions_vs_actual.png"
        fig_pva.savefig(path_pva, dpi=120)
        mlflow.log_artifact(str(path_pva), artifact_path="plots")

        logger.info("Plots générés et loggés", extra={"output_dir": str(plots_dir)})

        # Fermer les figures pour libérer la mémoire
        import matplotlib.pyplot as plt
        plt.close("all")

        # ── 6. Logging du modèle (avec signature et input_example) ────────
        signature = infer_signature(X_train, pipeline.predict(X_train.head(5)))
        input_example = X_train.head(3).copy()

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            signature=signature,
            input_example=input_example,
        )
        model_uri = f"runs:/{run_id}/model"
        logger.info("Modèle loggé MLflow", extra={"uri": model_uri})

        # ── 7. Enregistrement dans Registry + alias 'staging' ─────────────
        version = _promote_to_staging(client, model_uri, run_id)
        mlflow.set_tag("registered_model_version", str(version))

        # ── 8. Export metrics pour DVC ─────────────────────────────────────
        _save_metrics_for_dvc(metrics, run_id)

        # ── Bilan ──────────────────────────────────────────────────────────
        logger.info(
            "✅ Run terminé avec succès",
            extra={
                "run_id": run_id,
                "model_version": version,
                "alias": STAGING_ALIAS,
                "taux_mae": round(metrics["taux_mae"], 2),
                "taux_r2": round(metrics["taux_r2"], 3),
            },
        )

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        train_model()
        return 0
    except FileNotFoundError as e:
        logger.error("Données manquantes", extra={"error": str(e)})
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception("Échec train_model", extra={"error_type": type(e).__name__})
        return 1


if __name__ == "__main__":
    sys.exit(main())
