# `ml/src/models` — Entraînement, évaluation et inférence des modèles ML

Ce dossier contient l'entraînement et l'inférence du modèle de prédiction du taux de remplissage Vélib'.

## Fichiers

### `train_model.py`
Orchestrateur d'entraînement : charge le dataset, entraîne XGBoost sur la cible résiduelle, évalue, logue tout dans MLflow, enregistre le modèle dans le Registry et lui pose l'alias `staging`.

### `predict_model.py`
Module d'inférence réutilisable. Charge le modèle depuis le Registry par alias et expose 3 fonctions :
- `predict_residual()` — prédit le résidu brut (sortie modèle)
- `predict_taux()` — prédit le taux reconstruit (= station_trend_avg + résidu)
- `predict_with_confidence()` — prédit + niveau d'alerte (vert/jaune/rouge)

Réutilisable par l'API FastAPI (Phase 3) et le Streamlit MVP (Phase 4).

### `_helpers.py` (privé)
Briques unitaires : factory de pipeline, hyperparamètres XGBoost figés, calcul de métriques, génération de plots. Le préfixe `_` indique qu'elles sont privées au sous-package.

## Lancement

```bash
# Depuis la racine du repo, avec .venv activé

# Entraîner et enregistrer dans le Registry
python -m ml.src.models.train_model

# Tester l'inférence sur 5 lignes du test set
python -m ml.src.models.predict_model
```

**Prérequis** : avoir préalablement lancé `build_features` pour produire `data/processed/{train,test}_preprocessed.parquet`.

## Sorties de `train_model.py`

| Fichier | Rôle |
|---|---|
| `mlflow.db` | Backend SQLite MLflow (métadonnées, runs, registry) |
| `mlartifacts/<run_id>/` | Artefacts (modèle, plots) du run |
| `mlruns/` | Métadonnées additionnelles |
| `data/outputs/metrics.json` | Métriques pour DVC (R², MAE, etc.) |
| `data/outputs/feature_importance.png` | Top 20 features par importance |
| `data/outputs/residuals_distribution.png` | Distribution des erreurs de résidu |
| `data/outputs/predictions_vs_actual.png` | Scatter prédiction vs réalité |

Plus l'enregistrement dans le Model Registry MLflow :
- Modèle : `velib_fill_rate_predictor`
- Alias : `staging` (pose automatique sur la dernière version)

## Configuration MLflow

Le tracking utilise un backend **SQLite** (cohérent avec le module MLflow production de la formation), nécessaire pour le Model Registry. Le backend filesystem-only ne supporte pas le Registry.

```python
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("velib_fill_rate")
```

Pour ouvrir l'UI MLflow et visualiser les runs :
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# Puis ouvrir http://localhost:5000 dans le navigateur
```

## Hyperparamètres XGBoost

Les hyperparamètres sont **figés** dans `_helpers.py` (`HYPERPARAMS_XGB`), issus du tuning Optuna de l'ancien code (~100 trials sur la cible résiduelle).

```python
HYPERPARAMS_XGB = {
    "n_estimators": 232,
    "max_depth": 6,
    "learning_rate": 0.122,
    "subsample": 0.937,
    "colsample_bytree": 0.524,
    # ... etc
}
```

Pas de tuning à chaque entraînement (économie de temps, reproductibilité). Quand le dataset s'allongera (3+ mois), il pourra être pertinent de réintroduire un script `tune_hyperparameters.py` séparé pour ajuster.

## Cible : résiduelle, pas brute

Le modèle est entraîné sur `residual_target = taux - station_trend_avg`, pas sur `taux` directement.

**Avantage** : variance réduite ×2.7 à 4× selon la taille du dataset → modèle plus précis.

**Inférence** : `taux_prédit = station_trend_avg + résidu_prédit` (clippé [0, 100]).

## Métriques retournées

Trois métriques sur le résidu (qualité brute) + quatre sur le taux reconstruit (vue métier) :

| Métrique | Sur résidu | Sur taux | Interprétation |
|---|---|---|---|
| MAE | `residual_mae` | `taux_mae` | Erreur moyenne en points de % |
| RMSE | `residual_rmse` | `taux_rmse` | Erreur quadratique |
| R² | `residual_r2` | `taux_r2` | Variance expliquée |
| MAPE | — | `taux_mape_pct` | Erreur relative moyenne (%) |

L'évaluation sur le **taux reconstruit** est ce qui compte pour la démo et la soutenance — c'est ce que verra l'opérateur Vélib'.

## Pipeline DVC

Voir `dvc.yaml` à la racine. Les métriques et plots sont versionnés DVC :

```bash
dvc metrics show              # Affiche les métriques du dernier run
dvc metrics diff              # Compare 2 runs (utile après ré-entraînement)
dvc plots show                # Ouvre les 3 plots dans un viewer
```

## Architecture future (Phase 3+)

`predict_model.py` est conçu pour être consommé par :

- **FastAPI** : `from ml.src.models.predict_model import load_staging_model, predict_taux`
- **Streamlit** : idem, dans une page de prédiction live
- **Batch prediction** (cron toutes les 15 min) : dans `predict_batch.py` (à créer)

Le cache `@lru_cache` évite le re-chargement coûteux du modèle entre appels (~3-5 secondes par chargement).
