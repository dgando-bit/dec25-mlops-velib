"""
ml.src.models._helpers — Briques pour l'entraînement, l'évaluation et la génération de plots.

Module privé au sous-package ``models``. Regroupe :

    - FEATURES_FINAL          : liste figée des features utilisées par le modèle
    - HYPERPARAMS_XGB         : hyperparamètres XGBoost (depuis Optuna ancien code)
    - build_xgboost           : factory du pipeline sklearn(imputer + xgboost)
    - compute_metrics         : MAE, RMSE, R², MAPE (sur résidu et taux reconstruit)
    - reconstruct_target      : taux_prédit = station_trend_avg + résidu_prédit
    - plot_feature_importance : barre des importances XGBoost
    - plot_residuals_distribution : histogramme des résidus
    - plot_predictions_vs_actual  : scatter prédiction vs réalité

Toutes les fonctions sont **pures** (pas de side-effect, pas d'I/O sauf
les plot_* qui retournent un objet matplotlib Figure que l'orchestrateur
sauve via fig.savefig()).
"""
from __future__ import annotations

from typing import Any

import matplotlib
matplotlib.use("Agg")  # backend non-interactif (pas de fenêtre, juste écriture fichier)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURES UTILISÉES PAR LE MODÈLE
# ─────────────────────────────────────────────────────────────────────────────
# Liste figée pour reproductibilité. Doit matcher build_features.py.
# IMPORTANT : station_trend_avg N'EST PAS dans les features du modèle
# (la cible étant le résidu = taux - station_trend_avg, inclure
# station_trend_avg comme feature serait une fuite triviale).
FEATURES_FINAL: list[str] = [
    # Capacité et profil
    "capacity",
    "capacity_group",
    "morning_evening_ratio",
    # Temporel cyclique
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month",
    # Flags temporels
    "is_peak_hour",
    "is_friday_evening",
    "is_monday_morning",
    # Calendaires
    "is_holiday",
    "is_vacation",
    # Météo
    "apparent_temperature",
    "temp_anomalie",
    "weather_severity",
    "is_frozen",
    "is_stormy",
    # Lags
    "lag_60min",
    "lag_240min",
    "lag_res_240min",
    # Géographie
    "lat",
    "lon",
    # Coordonnée temporelle additionnelle
    "hour",
]


# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMÈTRES XGBOOST (figés depuis l'ancien tuning Optuna)
# ─────────────────────────────────────────────────────────────────────────────
# Source : meilleurs hyperparams trouvés par 03_hyperparameter.py de l'ancien
# code, après ~100 trials Optuna sur le résidu. Conservés tels quels pour
# garantir la reproductibilité.
HYPERPARAMS_XGB: dict[str, Any] = {
    "n_estimators": 232,
    "max_depth": 6,
    "learning_rate": 0.122,
    "subsample": 0.937,
    "colsample_bytree": 0.524,
    "colsample_bylevel": 0.562,
    "min_child_weight": 21,
    "reg_alpha": 1.40,
    "reg_lambda": 1.47,
    "gamma": 0.776,
    "random_state": settings.random_state,
    "n_jobs": -1,
    "tree_method": "hist",
    "verbosity": 0,
}


# ─────────────────────────────────────────────────────────────────────────────
# FACTORY DU PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def build_xgboost(hyperparams: dict[str, Any] | None = None) -> Pipeline:
    """Construit un pipeline sklearn (imputer + XGBoost).

    L'imputer en amont protège contre les NaN résiduels (par exemple sur les
    nouvelles stations sans historique pour temp_anomalie).
    XGBoost lui-même gère nativement les NaN, mais on chaîne par cohérence
    avec sklearn et pour pouvoir ajouter d'autres preprocessors plus tard.

    Args:
        hyperparams: dict d'hyperparamètres. Si None, utilise HYPERPARAMS_XGB.

    Returns:
        Pipeline sklearn entraînable via .fit(X, y).
    """
    params = hyperparams if hyperparams is not None else HYPERPARAMS_XGB
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", XGBRegressor(**params)),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# RECONSTRUCTION DE LA CIBLE
# ─────────────────────────────────────────────────────────────────────────────
def reconstruct_target(
    residual_pred: np.ndarray | pd.Series,
    station_trend_avg: pd.Series,
) -> pd.Series:
    """Reconstruit le taux prédit à partir du résidu prédit.

    Le modèle prédit ``residual = taux - station_trend_avg``.
    Pour reconstruire le taux : ``taux = station_trend_avg + résidu_prédit``.
    On clippe entre [0, 100] parce que physiquement un taux ne peut pas
    sortir de cet intervalle.

    Args:
        residual_pred: prédictions du modèle (résidus prédits).
        station_trend_avg: tendance historique par station (alignée).

    Returns:
        Series de taux reconstruits, clippés entre 0 et 100.
    """
    residual_pred = np.asarray(residual_pred)
    station_trend = np.asarray(station_trend_avg)
    return pd.Series(
        np.clip(station_trend + residual_pred, 0, 100),
        index=station_trend_avg.index if hasattr(station_trend_avg, "index") else None,
        name="taux_predicted",
    )


# ─────────────────────────────────────────────────────────────────────────────
# CALCUL DES MÉTRIQUES (sur résidu ET taux reconstruit)
# ─────────────────────────────────────────────────────────────────────────────
def compute_metrics(
    y_true_residual: pd.Series,
    y_pred_residual: np.ndarray | pd.Series,
    y_true_taux: pd.Series,
    y_pred_taux: pd.Series,
) -> dict[str, float]:
    """Calcule les métriques d'évaluation.

    Les métriques sur le résidu reflètent la qualité brute du modèle.
    Les métriques sur le taux reconstruit reflètent la qualité métier
    (c'est ce que verra l'opérateur Vélib').

    Args:
        y_true_residual: cible résiduelle réelle (test set).
        y_pred_residual: prédictions résiduelles du modèle.
        y_true_taux: taux réel (test set).
        y_pred_taux: taux reconstruit (station_trend + résidu_prédit).

    Returns:
        dict avec préfixes "residual_" et "taux_" pour chaque métrique.
    """
    y_pred_residual = np.asarray(y_pred_residual)
    y_pred_taux = np.asarray(y_pred_taux)

    # MAPE : on ignore les lignes où taux_réel ≈ 0 pour éviter la division par zéro
    nonzero_mask = y_true_taux > 1.0
    mape = (
        np.mean(np.abs((y_true_taux[nonzero_mask] - y_pred_taux[nonzero_mask])
                       / y_true_taux[nonzero_mask])) * 100
        if nonzero_mask.any() else float("nan")
    )

    return {
        # Résidu (fidélité brute du modèle)
        "residual_mae":  float(mean_absolute_error(y_true_residual, y_pred_residual)),
        "residual_rmse": float(np.sqrt(mean_squared_error(y_true_residual, y_pred_residual))),
        "residual_r2":   float(r2_score(y_true_residual, y_pred_residual)),
        # Taux reconstruit (vue métier)
        "taux_mae":      float(mean_absolute_error(y_true_taux, y_pred_taux)),
        "taux_rmse":     float(np.sqrt(mean_squared_error(y_true_taux, y_pred_taux))),
        "taux_r2":       float(r2_score(y_true_taux, y_pred_taux)),
        "taux_mape_pct": float(mape),
    }


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────
def plot_feature_importance(model: XGBRegressor, top_n: int = 20) -> plt.Figure:
    """Bar chart des importances XGBoost (top N)."""
    importances = pd.Series(
        model.feature_importances_, index=FEATURES_FINAL,
    ).sort_values(ascending=True).tail(top_n)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(importances.index, importances.values, color="#2E7D32")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Feature importance — top {top_n}")
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_residuals_distribution(
    y_true_residual: pd.Series,
    y_pred_residual: np.ndarray,
) -> plt.Figure:
    """Histogramme des erreurs de prédiction sur le résidu."""
    errors = np.asarray(y_pred_residual) - np.asarray(y_true_residual)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(errors, bins=80, color="#1976D2", edgecolor="white", alpha=0.85)
    ax.axvline(0, color="#C62828", linestyle="--", linewidth=1.5,
               label="Erreur nulle")
    ax.axvline(errors.mean(), color="#F57C00", linestyle=":", linewidth=2,
               label=f"Moyenne = {errors.mean():.2f}")
    ax.set_xlabel("Erreur de prédiction (résidu prédit - résidu réel)")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des erreurs de prédiction (résidu)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig


def plot_predictions_vs_actual(
    y_true_taux: pd.Series,
    y_pred_taux: pd.Series,
    sample_size: int = 10_000,
) -> plt.Figure:
    """Scatter (échantillonné) des prédictions vs valeurs réelles.

    Échantillonné pour rester lisible (un scatter de 1M points n'apporte rien).

    Args:
        y_true_taux: taux réel.
        y_pred_taux: taux prédit reconstruit.
        sample_size: taille de l'échantillon affiché.

    Returns:
        matplotlib Figure.
    """
    n = len(y_true_taux)
    if n > sample_size:
        idx = np.random.RandomState(settings.random_state).choice(
            n, size=sample_size, replace=False
        )
        y_true_sample = np.asarray(y_true_taux)[idx]
        y_pred_sample = np.asarray(y_pred_taux)[idx]
    else:
        y_true_sample = np.asarray(y_true_taux)
        y_pred_sample = np.asarray(y_pred_taux)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_true_sample, y_pred_sample, alpha=0.15, s=8, color="#1976D2")
    ax.plot([0, 100], [0, 100], color="#C62828", linestyle="--", linewidth=1.5,
            label="y = x (prédiction parfaite)")
    ax.set_xlabel("Taux réel (%)")
    ax.set_ylabel("Taux prédit (%)")
    ax.set_title(
        f"Prédictions vs réalité — échantillon {min(n, sample_size):,} points"
    )
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left")
    ax.grid(linestyle="--", alpha=0.4)
    fig.tight_layout()
    return fig
