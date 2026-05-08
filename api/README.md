# `api/` — API FastAPI d'inférence Vélib'

Service HTTP qui expose le modèle XGBoost de prédiction du taux de remplissage des stations Vélib'. Charge le modèle depuis le MLflow Registry par alias (`staging` par défaut), garde en cache pour des prédictions rapides.

## Architecture

```
api/
├── pyproject.toml             ← package + dépendances (FastAPI, uvicorn)
├── README.md                  ← ce fichier
├── Dockerfile                 ← (à venir, Phase 3.1)
└── velib_api/                 ← package Python (importable comme 'velib_api')
    ├── __init__.py
    ├── main.py                ← app FastAPI + 5 routes
    ├── schemas.py             ← modèles Pydantic I/O (validation stricte)
    └── dependencies.py        ← injection modèle, settings, lifespan
```

Le **dossier extérieur** `api/` est le **service** (avec ses configs, son Dockerfile, etc.).
Le **sous-dossier** `velib_api/` est le **package Python** importable.
Cette dissociation évite les conflits de namespace Python (le dossier `api/` et le module `api` qui se télescopaient).

L'API **réutilise** :
- `shared.config.settings` — Pydantic Settings centralisée
- `ml.src.models.predict_model` — fonctions d'inférence (`predict_with_confidence`)
- `ml.src.models._helpers.FEATURES_FINAL` — liste figée des 24 features

## Endpoints

| Endpoint | Méthode | Rôle |
|---|---|---|
| `/` | GET | Redirige vers `/docs` |
| `/health` | GET | Santé du service (ok / degraded) |
| `/model/info` | GET | Métadonnées du modèle chargé |
| `/predict` | POST | Prédiction unitaire (1 station) |
| `/predict/batch` | POST | Prédiction batch (jusqu'à 2000 stations) |
| `/docs` | GET | Swagger UI (généré par FastAPI) |
| `/redoc` | GET | ReDoc (généré par FastAPI) |

## Format des inputs

L'API attend les **24 features pré-calculées** par le pipeline de feature engineering, plus `station_trend_avg` pour la reconstruction du taux à partir du résidu prédit (25 champs au total).

**Exemple de requête `POST /predict`** :

```json
{
  "capacity": 30,
  "capacity_group": 1,
  "morning_evening_ratio": 1.05,
  "hour_sin": 0.866,
  "hour_cos": 0.5,
  "dow_sin": 0.0,
  "dow_cos": 1.0,
  "month": 5,
  "is_peak_hour": 0,
  "is_friday_evening": 0,
  "is_monday_morning": 1,
  "is_holiday": 0,
  "is_vacation": 0,
  "apparent_temperature": 16.5,
  "temp_anomalie": 1.2,
  "weather_severity": 0,
  "is_frozen": 0,
  "is_stormy": 0,
  "lag_60min": 45.0,
  "lag_240min": 38.0,
  "lag_res_240min": -3.5,
  "lat": 48.8566,
  "lon": 2.3522,
  "hour": 8,
  "station_trend_avg": 42.0
}
```

**Réponse** :

```json
{
  "residual_predicted": -2.34,
  "taux_predicted": 39.66,
  "alert_level": "green"
}
```

### Niveaux d'alerte

| Niveau | Condition (taux prédit) | Signification opérateur |
|---|---|---|
| `green` | 30% — 70% | Normal |
| `yellow` | 10-30% ou 70-90% | À surveiller |
| `red` | <10% ou >90% | Saturation imminente, intervention possible |

## Installation locale

```bash
# Depuis la racine du repo, avec .venv activé
cd ~/dec25-mlops-velib
source .venv/bin/activate

# Installer le service api/ en mode editable
pip install -e api/

# Vérifier que les imports marchent
python -c "from velib_api.main import app; print('OK')"
```

## Lancement

### Mode développement (autoreload)

```bash
uvicorn velib_api.main:app --reload --port 8000
```

Ouvre http://localhost:8000/docs dans ton navigateur. Tu vois le Swagger UI, tu peux tester `POST /predict` directement depuis l'interface.

### Mode production

```bash
uvicorn velib_api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

Avec 4 workers Uvicorn pour paralléliser les requêtes (chaque worker a sa propre copie du modèle en mémoire — ~50-100 Mo par worker).

## Prérequis

L'API a besoin de :
1. **`mlflow.db`** présent à la racine du repo (créé par `train_model.py`)
2. **Un modèle `velib_fill_rate_predictor` enregistré** dans le Registry avec alias `staging`
3. **Les artefacts MLflow** dans `mlartifacts/` (joblib pickle XGBoost)

Si l'un manque, l'API démarre quand même mais en **mode dégradé** : `/health` renverra `status=degraded`, `/predict` renverra `503 Service Unavailable`.

## Tests rapides

### Health check

```bash
curl http://localhost:8000/health
# → {"status":"ok","api_version":"0.1.0","model_loaded":true}
```

### Modèle info

```bash
curl http://localhost:8000/model/info
# → {"model_name":"velib_fill_rate_predictor","alias":"staging","version":"3",...}
```

### Prédiction unitaire (avec un fichier JSON)

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @example_request.json
```

## Cycle de vie du modèle

```
Démarrage uvicorn
       │
       ▼
[lifespan startup] → preload_model() → lru_cache rempli
       │
       ▼
Service prêt à recevoir des requêtes (/predict instantané)
       │
       ▼
[Requêtes HTTP] → get_model() → cache hit (rapide)
       │
       ▼
[lifespan shutdown] → log et fin
```

**Coût du démarrage** : ~3-5 secondes (chargement MLflow + désérialisation joblib XGBoost).
**Coût d'une requête `/predict`** : ~5-20 ms (XGBoost predict + sérialisation Pydantic).

## À venir (Phase 3+)

- **Dockerfile** + intégration `docker-compose.yml`
- Endpoint `POST /model/reload` pour recharger sans redémarrer le service
- Endpoint `GET /metrics` pour exposition Prometheus
- Tests unitaires (`api/tests/`) avec FastAPI TestClient

## Références

- FastAPI : https://fastapi.tiangolo.com/
- Documentation MLflow Registry : https://mlflow.org/docs/latest/model-registry.html
- Module formation associé : S14.3 (FastAPI), S15.1 (MLflow)
