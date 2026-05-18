# Projet Vélib' MLOps — Présentation technique
### Formation Machine Learning Engineer — DataScientest (promo décembre 2025)

---

## 1. Problématique métier

Paris compte **1 511 stations Vélib'** réparties sur l'ensemble de l'agglomération, avec une fréquentation qui varie fortement selon l'heure, le jour, la météo et les événements. Deux frustrations majeures reviennent dans les usages :

- **L'usager** ne sait pas si la station la plus proche aura un vélo disponible dans 30 minutes.
- **Smovengo** (l'opérateur) doit planifier ses tournées de rééquilibrage logistique sans visibilité prédictive sur l'occupation future.

**Objectif du projet** : construire un système MLOps de bout en bout capable de prédire le taux de remplissage de chaque station Vélib' à horizon court terme, en s'appuyant exclusivement sur des services open source ou free tier.

---

## 2. Architecture globale

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         COLLECTE CONTINUE                                │
│  cron-job.org (toutes les 5 min) ──► HF Space (Flask) ──► HF Datasets   │
│  Sources : Vélib' GBFS · open-meteo.com · vacances · jours fériés        │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION (Airflow 2.11)                        │
│  DAG quotidien 04h00 UTC : vérif HF → vérif MLflow → DVC repro          │
│                            → rechargement modèle → smoke test            │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                          ┌─────────┴─────────┐
                          ▼                   ▼
┌────────────────────┐  ┌────────────────────────────────┐
│   PIPELINE ML      │  │   SERVING & MONITORING         │
│   (DVC, 5 stages)  │  │                                │
│                    │  │  Nginx :8080 → FastAPI          │
│  load_from_hf      │  │  Nginx :5000 → MLflow UI       │
│       ↓            │  │  Nginx :9090 → Prometheus      │
│  make_dataset      │  │  Nginx :3000 → Grafana         │
│       ↓            │  │  Nginx :8090 → Airflow UI      │
│  build_features    │  │  Nginx :8888 → JupyterLab      │
│       ↓            │  │                                │
│  train_model       │  │  PostgreSQL (MLflow backend)   │
│  → MLflow Registry │  │  PostgreSQL (Airflow metadata) │
└────────────────────┘  └────────────────────────────────┘
```

**Contrainte de conception** : tous les ports exposés vers l'extérieur transitent par Nginx. Aucun service n'est accessible directement depuis Internet.

---

## 3. Stack technique

| Couche | Technologie | Justification |
|---|---|---|
| **Langage** | Python 3.12 | Écosystème ML mature, timezone-aware |
| **API** | FastAPI + Uvicorn | Performance async, OpenAPI auto-généré |
| **Validation** | Pydantic v2 | Contrats de données stricts à l'entrée |
| **ML** | XGBoost 3.0.2 + scikit-learn 1.5 | Gradient boosting performant sur données tabulaires |
| **Tuning** | Optuna 3.6.1 | Optimisation bayésienne des hyperparamètres |
| **Tracking** | MLflow 2.20.3 + PostgreSQL 15 | Reproductibilité, registry, alias `staging` |
| **Versioning données** | DVC + DagsHub (S3-compatible) | Reproductibilité du pipeline, remote gratuit |
| **Source données** | HuggingFace Hub (~0.36) | Stockage ~700 Mo/semaine de données Vélib' |
| **Orchestration** | Airflow 2.11.2 (LocalExecutor) | DAG quotidien, retry, observabilité |
| **Monitoring** | Prometheus + Grafana | Métriques temps réel de l'API |
| **Proxy** | Nginx | Point d'entrée unique, rate limiting |
| **Conteneurs** | Docker Compose v2 | Portabilité, isolation par service |
| **Tests** | pytest + pytest-cov | 94 tests unitaires, fixture XGBoost réelle |

---

## 4. Pipeline de données (DVC)

Le pipeline est composé de 5 stages enchaînés, versionnés et reproductibles via DVC :

```
load_from_hf ──► make_dataset ──► dataviz (HTML Plotly, 7 graphes)
                     │
                     └──► build_features ──► train_model
```

| Stage | Rôle | Sortie |
|---|---|---|
| `load_from_hf` | Télécharge tous les CSV HuggingFace, concatène, déduplique | `data/raw/velib_snapshot_latest.parquet` (8,7 Mo zstd) |
| `make_dataset` | Nettoyage : filtres qualité, calcul du taux de remplissage | `data/interim/velib_cleaned_latest.parquet` |
| `dataviz` | Rapport HTML Plotly (7 graphes interactifs) | `data/outputs/plots/dataviz_report.html` (~110 Mo) |
| `build_features` | Feature engineering, split temporel train/test | `data/processed/{train,test}_preprocessed.parquet` |
| `train_model` | XGBoost, Optuna, log MLflow, promotion alias `staging` | Modèle dans MLflow Registry |

**Volumétrie** : ~4,9 millions de lignes après nettoyage, 1 492 stations couvertes, taux de filtrage ~1,3 % (qualité saine).

---

## 5. Modèle ML

### Approche : prédiction du résidu

Plutôt que de prédire directement le taux de remplissage, le modèle apprend l'**écart par rapport à la tendance historique de la station** :

```
résidu = taux_observé − station_trend_avg
taux_prédit = résidu_prédit + station_trend_avg
```

Cette formulation réduit la variance cible et permet au modèle de se concentrer sur les déviations significatives (météo, événements, heure de pointe).

### Features (24)

| Catégorie | Features |
|---|---|
| Station | `capacity`, `capacity_group`, `morning_evening_ratio`, `lat`, `lon` |
| Temporel | `hour`, `hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`, `month` |
| Flags | `is_peak_hour`, `is_friday_evening`, `is_monday_morning`, `is_holiday`, `is_vacation` |
| Météo | `apparent_temperature`, `temp_anomalie`, `weather_severity`, `is_frozen`, `is_stormy` |
| Lags | `lag_60min`, `lag_240min`, `lag_res_240min` |
| Tendance | `station_trend_avg` |

### Métriques (run `ea250421`, 15 mai 2026)

| Métrique | Valeur |
|---|---|
| R² (taux reconstruit) | **0,833** |
| MAE (taux reconstruit) | **8,32 points de %** |
| RMSE | 12,08 points de % |
| MAPE | 36,7 % |

> Un R² de 0,833 signifie que le modèle explique 83,3 % de la variance du taux de remplissage. Une MAE de 8,3 points correspond à une erreur moyenne de moins d'une demi-case sur un dock de 20 places.

### Niveaux d'alerte

L'API enrichit chaque prédiction d'un niveau d'alerte exploitable par un client ou une interface :

| Niveau | Condition | Interprétation |
|---|---|---|
| 🟢 `green` | 30 % ≤ taux ≤ 70 % | Station bien équilibrée |
| 🟡 `yellow` | 10–30 % ou 70–90 % | Tension modérée |
| 🔴 `red` | < 10 % ou > 90 % | Station quasi-vide ou saturée |

---

## 6. API d'inférence

L'API FastAPI est exposée via Nginx sur `http://localhost:8080`. Elle est conçue pour la tolérance aux pannes : si MLflow est indisponible au démarrage, l'API répond `degraded` plutôt que de crasher.

| Endpoint | Description |
|---|---|
| `GET /health` | Liveness — retourne `ok` ou `degraded` |
| `GET /model/info` | Métadonnées du modèle en mémoire (version, alias, features) |
| `POST /model/reload` | Recharge le modèle `staging` depuis MLflow sans redémarrage |
| `POST /predict` | Prédiction unitaire pour une station |
| `POST /predict/batch` | Prédiction batch (1 à 2 000 stations) |
| `GET /docs` | Swagger UI auto-générée (Pydantic → OpenAPI) |

---

## 7. Monitoring

### Prometheus + Grafana

Prometheus scrape l'endpoint `/metrics` de l'API toutes les 15 secondes. Grafana est provisionné automatiquement (datasource + dashboard via fichiers de configuration versionnés).

Métriques collectées : nombre de requêtes par endpoint, latence (p50/p95/p99), taux d'erreur, statut du modèle chargé.

### Qualité des données (logs structurés)

Chaque stage du pipeline produit des logs structurés (text ou JSON) avec des métriques de qualité :

- `load_from_hf` : nombre de lignes, stations couvertes, hash SHA256 du parquet
- `make_dataset` : taux de filtrage, nombre de stations exclues pour variance nulle
- `train_model` : métriques MLflow loguées à chaque run, comparaison avec le run précédent

---

## 8. Tests unitaires (Phase 2)

94 tests répartis sur 3 packages, exécutables sans la stack complète :

| Périmètre | Fichier | Tests |
|---|---|---|
| Schémas Pydantic | `api/tests/test_schemas.py` | 13 |
| Endpoints FastAPI | `api/tests/test_endpoints.py` | 14 |
| Nettoyage données | `ml/tests/test_data_cleaning.py` | 18 |
| Feature engineering | `ml/tests/test_build_features.py` | 16 |
| Inférence & alertes | `ml/tests/test_predict_model.py` | 10 |
| Configuration | `shared/tests/test_config.py` | 12 |

**Stratégie** : les tests API injectent un vrai modèle XGBoost entraîné sur données synthétiques (fixture session-scoped). Aucun appel MLflow ou PostgreSQL pendant la CI — les tests sont hermétiques et rapides.

```bash
make test-unit   # lance les 94 tests sans démarrer la stack
```

---

## 9. Orchestration Airflow (Phase 3)

Airflow 2.11.2 orchestre l'exécution quotidienne du pipeline et le rechargement du modèle.

**DAG `velib_pipeline`** (planifié à 04h00 UTC chaque jour) :

```
check_hf_connectivity
        │
        ▼
check_mlflow_health
        │
        ▼
dvc_repro  ◄─── docker compose run ml_training dvc repro
        │       (Docker-out-of-Docker via socket monté)
        ▼
reload_model  ◄─── POST /model/reload
        │
        ▼
smoke_test  ◄─── GET /health → vérifie status "ok"
```

L'UI Airflow est accessible via Nginx sur `http://localhost:8090`.

---

## 10. Collecte continue (infrastructure HF)

La collecte tourne 24/7 sur des services free tier, sans coût d'infrastructure :

| Service | Rôle | Coût |
|---|---|---|
| cron-job.org | Déclenchement toutes les 5 min | 0 € |
| HuggingFace Space (Flask) | Réception et ingestion des données Vélib' | 0 € |
| HuggingFace Datasets | Stockage historique (~700 Mo/semaine) | 0 € |
| DagsHub | Remote DVC (artefacts ML) | 0 € |

**Sources ingérées** à chaque relevé : API Vélib' GBFS (statuts stations), open-meteo.com (météo apparente, code météo), data.education.gouv.fr (vacances scolaires), bibliothèque `holidays` (jours fériés France).

---

## 11. État d'avancement

| Composant | État |
|---|---|
| Collecte continue HuggingFace | ✅ Opérationnel |
| Pipeline DVC (5 stages) | ✅ Reproductible |
| Rapport Plotly (7 graphes) | ✅ Généré à chaque run |
| Modèle XGBoost + MLflow Registry | ✅ R² = 0,833 |
| API FastAPI (6 endpoints) | ✅ Mode dégradé, rechargement à chaud |
| Reverse proxy Nginx | ✅ Point d'entrée unique |
| Monitoring Prometheus + Grafana | ✅ Dashboard provisionné |
| Tests unitaires (94 tests) | ✅ CI hermétique |
| Orchestration Airflow | ✅ DAG quotidien opérationnel |
| Streamlit de démonstration | ⏳ En cours |
| Drift monitoring (Evidently) | ⏳ Planifié |

---

## 12. Prochaines étapes

**Streamlit (en cours)** : interface de démonstration avec deux volets —
- *Volet validation* : test interactif de chaque service de la stack (healthchecks, prédiction, MLflow, Nginx)
- *Volet MVP* : sélection de station, prédiction d'occupation, visualisation des alertes

**Evidently** : rapport de drift automatique intégré au DAG Airflow et au Streamlit — data drift sur les features et model drift sur les résidus prédits vs observés.

---

## Ressources

| Ressource | Lien |
|---|---|
| Repo GitHub | https://github.com/dgando-bit/dec25-mlops-velib |
| Dataset HuggingFace | https://huggingface.co/datasets/voroman/velib-ml-data |
| API Vélib' Open Data | https://www.velib-metropole.fr/donnees-open-data-gbfs-du-service-velib-metropole |

---

*Développé dans le cadre de la formation Machine Learning Engineer — DataScientest (promo décembre 2025).*
