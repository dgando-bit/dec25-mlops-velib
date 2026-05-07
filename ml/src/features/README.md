# `ml/src/features` — Feature engineering du pipeline ML

Ce dossier contient le pipeline de feature engineering qui transforme le dataset nettoyé en features prêtes à l'entraînement XGBoost.

## Fichiers

### `build_features.py`
Orchestrateur principal du feature engineering. Lit le parquet nettoyé produit par `make_dataset`, applique 9 étapes de transformation, et écrit les datasets train/test ainsi que la table `stations_geo`.

### `_helpers.py` (privé)
Briques unitaires de feature engineering, factorisées pour testabilité. Le préfixe `_` indique qu'elles sont privées au sous-package — elles n'ont pas vocation à être utilisées hors de `features/`.

## Lancement

```bash
# Depuis la racine du repo, avec .venv activé
python -m ml.src.features.build_features
```

**Prérequis** : avoir préalablement lancé `load_from_hf` puis `make_dataset` pour produire `data/interim/velib_cleaned_latest.parquet`.

## Sorties

Trois parquets dans `data/processed/` :

| Fichier | Rôle |
|---|---|
| `train_preprocessed.parquet` | Dataset d'entraînement (~80% temporel) |
| `test_preprocessed.parquet` | Dataset de test (~20% le plus récent) |
| `stations_geo.parquet` | Table de correspondance station_id ↔ name + GPS (pour Streamlit) |

Plus un log humain : `data/processed/.features.log`.

## Les 9 étapes du pipeline

| # | Étape | Rôle |
|---|---|---|
| 1 | Chargement | Lecture du parquet nettoyé |
| 2 | Features temporelles | hour, dow, month, sin/cos cycliques, flags binaires |
| 3 | capacity_group | Catégorisation des stations par taille (0/1/2/3) |
| 4 | Features météo | weather_severity, is_frozen, is_stormy |
| 5 | Encodage calendaire | is_holiday, is_vacation → int8 |
| 6 | Lags temporels | lag_60min, lag_240min via merge_asof |
| 7 | Split temporel | 80/20 par quantile sur datetime |
| 8 | Post-split features | morning_evening_ratio, temp_anomalie (sur train uniquement) |
| 9 | station_trend_avg | Moyenne historique avec fallback 2 niveaux (sur train uniquement) |
| 10 | Cible résiduelle | residual_target = taux - station_trend_avg ; lag_res_240min |

## Liste des features finales (38 colonnes)

**Identifiants et contexte (7)** :
`station_id`, `name`, `lat`, `lon`, `capacity`, `datetime`, `capacity_status`

**Vélos / docks (4)** :
`bikes_mechanical`, `bikes_ebike`, `numdocksavailable`, `total_capacity`

**Booléens originaux (3)** :
`is_renting`, `is_holiday`, `is_vacation`

**Météo (3)** :
`apparent_temperature`, `weather_code`, `weather_severity`, `is_frozen`, `is_stormy`

**Cible (2)** :
`taux` (cible brute), `residual_target` (cible utilisée par le modèle)

**Features temporelles (10)** :
`hour`, `day_of_week`, `month`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `is_peak_hour`, `is_friday_evening`, `is_monday_morning`

**Features de profil (2)** :
`capacity_group`, `morning_evening_ratio`

**Features dérivées (2)** :
`temp_anomalie`, `station_trend_avg`

**Lags (3)** :
`lag_60min`, `lag_240min`, `lag_res_240min`

## Anti-fuite (data leakage)

Trois features sont calculées sur le train UNIQUEMENT, puis appliquées au test via mapping :

- `morning_evening_ratio` : ratio par station, calculé sur train, mappé sur test (fallback 1.0 pour stations absentes du train)
- `temp_anomalie` : normale mensuelle calculée sur train, appliquée au test
- `station_trend_avg` : moyenne historique par (station × dow × hour × month), avec fallback 2 niveaux

Si on les calculait sur le dataset complet avant le split, l'information du test set "fuiterait" dans le train, gonflant artificiellement les métriques.

## Cible résiduelle — pourquoi ?

Le modèle prédit `residual_target = taux_réel - station_trend_avg`, pas le taux brut directement.

**Avantage** : réduit la variance de la cible d'un facteur ~2.7 (29% → 11%).
Le problème devient plus ciblé, plus facile pour XGBoost.

**En production** : `taux_prédit = station_trend_avg + résidu_prédit`.

## Pipeline DVC

Voir `dvc.yaml` à la racine pour le DAG complet.

```bash
dvc repro build_features      # ré-exécute uniquement ce stage
dvc repro                     # ré-exécute tout ce qui doit l'être
```
