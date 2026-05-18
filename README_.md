# Projet Vélib' MLOps — Prédiction du taux de remplissage des stations

> Pipeline ML complet pour prédire le taux de remplissage des stations Vélib' à Paris, conçu comme projet fil rouge de la formation Machine Learning Engineer (branches Data Product Manager, Data Scientist, MLOps).

---

## Vue d'ensemble

Le projet répond à un besoin concret : **anticiper l'état de remplissage des 1 511 stations Vélib' de Paris** pour aider les usagers (trouver un vélo / une place) et l'opérateur Smovengo (optimiser le rééquilibrage logistique).

L'architecture est pensée comme un mini-système MLOps end-to-end avec une contrainte forte : **rester compatible avec les offres free tier** des services tiers (HuggingFace Spaces, cron-job.org, DagsHub).

### Flux de données global

```
┌──────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│  COLLECTE CONTINUE (toutes les 5 min, hébergé HuggingFace Spaces)        │
│  ─────────────────────────────────────────────────────────────           │
│  cron-job.org ──► HF Space (Flask) ──► HF Datasets (CSV rotatif 4Go max) │
│                                                                          │
│                                            │                             │
│                                            ▼                             │
│                                                                          │
│  PIPELINE ML LOCAL (lancé manuellement ou via Airflow plus tard)         │
│  ─────────────────────────────────────────────────                       │
│                                                                          │
│  ① load_from_hf.py    → data/raw/velib_snapshot_latest.parquet           │
│  ② make_dataset.py    → data/interim/velib_cleaned_latest.parquet        │
│  ③ dataviz.py         → data/outputs/plots/dataviz_report.html           │
│  ④ build_features.py  → data/processed/{train,test}_preprocessed.parquet │
│  ⑤ train_model.py     → MLflow Registry (velib_fill_rate_predictor)      │
│                                                                          │
│                                            │                             │
│                                            ▼                             │
│                                                                          │
│  SERVING                                                                 │
│  ────────                                                                │
│  api/velib_api/main.py (FastAPI)  ──►  /predict                         │
│  deployments/nginx/ (reverse proxy :8080)                                │
│  Streamlit dashboard (Phase 3 — démonstration jury)                      │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### État d'avancement (mai 2026)

| Composant                              | État             | Module formation    |
|----------------------------------------|------------------|---------------------|
| Collecte continue (HF Space)           | ✅ Opérationnel   | —                   |
| `load_from_hf.py` (téléchargement)     | ✅ Implémenté     | DVC                 |
| `make_dataset.py` (nettoyage)          | ✅ Implémenté     | DVC + cookiecutter  |
| `dataviz.py` (visualisation Plotly)    | ✅ Implémenté     | Seaborn / Plotly    |
| `build_features.py` (FE + split)       | ✅ Implémenté     | Feature engineering |
| `train_model.py` (XGBoost + MLflow)    | ✅ Implémenté     | MLflow              |
| Pipeline DVC (5 stages)                | ✅ Opérationnel   | DVC + DagsHub       |
| API d'inférence (FastAPI)              | ✅ Opérationnelle — 6 endpoints dont `POST /model/reload` | FastAPI |
| Reverse proxy Nginx                    | ✅ Point d'entrée unique + pages d'erreur personnalisées (404/429/50x) | Nginx |
| Monitoring Prometheus                  | ✅ Opérationnel (scrape API /metrics toutes les 15s) | Prometheus/Grafana  |
| Dashboard Grafana                      | ✅ Validé visuellement (datasource uid fixe, panels alimentés) | Prometheus/Grafana  |
| Tests unitaires pytest                 | ✅ Phase 2 — 94 tests (API, ML, shared) avec modèle XGBoost fixture | pytest |
| Orchestration Airflow                  | ✅ Phase 3 — DAG quotidien (HF check → DVC repro → reload modèle) | Airflow |
| Streamlit (démo jury)                  | ✅ Phase 3 — Accueil + Validation + Prédiction MVP | Streamlit |
| Drift detection (Evidently)            | ⏳ Phase 4        | Evidently           |

### Volumétrie observée (05 mai 2026)

| Métrique | Valeur |
|---|---|
| Stations couvertes | 1 511 |
| Fréquence de collecte | 1 relevé / 5 min / station |
| Volume hebdomadaire | ~700 Mo en CSV bruts (HF) |
| Snapshot téléchargé | 8.7 Mo en parquet zstd |
| Snapshot nettoyé | 13.2 Mo en parquet zstd |
| Lignes après nettoyage | ~4.9 millions |
| Taux de filtrage | ~1.3% (qualité saine) |

---

## Arborescence du projet

```
dec25-mlops-velib/
├── README_.md                      ← (ce fichier)
├── .env.example                    ← template variables d'environnement
├── .gitignore                      ← exclusions Git
├── dvc.yaml                        ← pipeline DVC reproductible (5 stages)
├── dvc.lock                        ← snapshot des hash DVC (versionné Git)
├── docker-compose.yml              ← orchestration 6 services
├── Makefile                        ← CLI projet : build/up/down/train/dvc-*
│
├── shared/                         ← package Python partagé (installé en editable)
│   ├── pyproject.toml
│   └── shared/
│       ├── __init__.py
│       ├── config.py               ← Pydantic Settings singleton (lit .env)
│       ├── logger.py               ← logger structuré text/JSON
│       └── utils/
│           ├── __init__.py
│           ├── data_cleaning.py    ← utilitaires nettoyage partagés
│           └── helpers.py          ← helpers génériques
│
├── ml/                             ← service ML
│   ├── requirements.txt            ← dépendances ML
│   ├── Dockerfile                  ← multi-stage : base / training / jupyter
│   └── src/
│       ├── main.py                 ← point d'entrée ml_training — appelle train_model.main()
│       ├── data/
│       │   ├── load_from_hf.py     ← stage 1 DVC ✅
│       │   └── make_dataset.py     ← stage 2 DVC ✅
│       ├── features/
│       │   ├── build_features.py   ← stage 4 DVC ✅
│       │   └── _helpers.py         ← FEATURES_FINAL (24 features) + constantes
│       ├── models/
│       │   ├── train_model.py      ← stage 5 DVC ✅ (XGBoost + MLflow)
│       │   ├── predict_model.py    ← load_staging_model(), predict_with_confidence()
│       │   └── _helpers.py         ← helpers partagés training/inference
│       └── visualization/
│           └── dataviz.py          ← stage 3 DVC ✅ (rapport HTML Plotly, 7 graphes)
│
├── api/                            ← API FastAPI d'inférence ✅
│   ├── Dockerfile
│   ├── requirements.txt
│   └── velib_api/
│       ├── main.py                 ← app FastAPI, 5 endpoints, lifespan startup/shutdown
│       ├── dependencies.py         ← get_model() / preload_model() — injection FastAPI
│       ├── schemas.py              ← Pydantic : StationFeatures (25 champs), réponses
│       └── inference.py
│
├── mlflow/                         ← serveur MLflow (tracking + registry)
│   ├── Dockerfile
│   ├── requirements.txt
│   └── artifacts/                  ← (ignoré Git, persisté en volume Docker)
│
├── data/                           ← données versionnées DVC (ignorées Git)
│   ├── raw/
│   │   ├── velib_snapshot_latest.parquet  ← généré par load_from_hf
│   │   └── .snapshots.log
│   ├── interim/
│   │   ├── velib_cleaned_latest.parquet   ← généré par make_dataset
│   │   └── .cleaning.log
│   ├── processed/
│   │   ├── train_preprocessed.parquet
│   │   ├── test_preprocessed.parquet
│   │   └── stations_geo.parquet
│   └── outputs/
│       ├── metrics.json            ← métriques du dernier run (non caché DVC)
│       └── plots/
│           ├── dataviz_report.html ← généré par dataviz (~110 Mo)
│           ├── feature_importance.png
│           ├── residuals_distribution.png
│           └── predictions_vs_actual.png
│
└── deployments/
    ├── nginx/
    │   ├── Dockerfile
    │   └── nginx.conf              ← point d'entrée unique : :8080→api, :5000→mlflow, :8888→jupyter, :9090→prometheus, :3000→grafana, :8090→airflow, :8501→streamlit
    └── prometheus/
        └── prometheus.yml          ← config Prometheus (scrape API /metrics toutes les 15s)
```

> **Ports d'accès (tous via Nginx) :**
> - API FastAPI  → `http://localhost:8080`
> - MLflow UI   → `http://localhost:${MLFLOW_PORT}` (défaut `.env.example` : `5000`)
> - JupyterLab  → `http://localhost:8888`
> - Prometheus  → `http://localhost:9090`
> - Grafana     → `http://localhost:3000`
> - Airflow UI  → `http://localhost:${AIRFLOW_PORT}` (défaut : `8090`)
> - Streamlit   → `http://localhost:${STREAMLIT_PORT}` (défaut : `8501`)

---

## Prérequis

- **WSL2 + Ubuntu** (sur Windows) ou Linux/macOS natif
- **Docker Desktop** (Windows) ou **Docker Engine + Docker Compose v2** (Linux/macOS)
  - Vérifier : `docker compose version` doit renvoyer `v2.x.x` (avec espace, sans tiret)
- **Git**
- **Python 3.11 ou 3.12** (recommandé : 3.12) — uniquement pour le mode debug local (étapes §3–5)
- **VSCode** avec extension WSL recommandé pour le développement
- Un compte **HuggingFace** (pour générer un token Read accédant au dataset `voroman/velib-ml-data`)

> **Important** : si tu travailles sous Windows, **tout doit se faire depuis un terminal WSL**, jamais depuis PowerShell ou CMD. Les paquets Python (numpy, scikit-learn, pyarrow) n'ont pas de wheel précompilée pour les versions Python récentes sous Windows et tenteraient de compiler depuis les sources, ce qui plante.

> **Instances multiples sur la même machine** : la variable `COMPOSE_PROJECT_NAME` dans `.env` préfixe tous les conteneurs et volumes générés par Compose. Pour faire coexister deux instances (ex. dev et test de reproductibilité), changer cette valeur dans chaque `.env` (`velib-dev`, `velib-test`, etc.). Par défaut elle vaut `velib`.

---

## Installation pas à pas

### 1. Ouvrir un terminal WSL

Trois options pour entrer dans WSL :

**Option A — VSCode avec extension WSL (recommandé)** :
- Lance VSCode côté Windows
- `Ctrl+Shift+P` → tape `WSL: Open Folder in WSL`
- Sélectionne ton dossier de travail (par ex. `/home/voroman/`)
- L'indicateur en bas à gauche doit afficher `WSL: Ubuntu` sur fond coloré
- Le terminal intégré (`Ctrl+ù`) ouvre alors un bash Linux

**Option B — Ubuntu directement** :
- Menu Démarrer Windows → cherche `Ubuntu` → lance

**Option C — Depuis PowerShell** :
```powershell
wsl
```

### 2. Cloner le repo

```bash
cd ~
git clone https://github.com/dgando-bit/dec25-mlops-velib.git
cd dec25-mlops-velib
git checkout feat-monitoring-prometheus-grafana
```

> Cette branche contient la stack complète : pipeline DVC, API FastAPI, Nginx, MLflow, Prometheus et Grafana.

### 3. Créer un environnement virtuel Python

> **Mode Docker (nominal)** : les étapes §3, §4 et §5 ne sont pas nécessaires pour lancer le pipeline via `make pipeline`. Passe directement à l'étape §6 si tu utilises uniquement Docker. Ces étapes sont réservées au mode debug local (exécution des scripts Python hors Docker).

C'est **non négociable** — sans `venv`, tu vas mélanger les dépendances de ce projet avec celles de tes autres projets et créer des conflits illisibles.

```bash
# Créer le venv (à faire UNE seule fois par projet)
python3 -m venv .venv

# Activer le venv (à faire à CHAQUE nouveau terminal)
source .venv/bin/activate
```

> Tu sauras que le venv est activé quand ton prompt commence par `(.venv)`. Pour le désactiver plus tard : `deactivate`.

### 4. Vérifier que tu utilises bien le bon Python

```bash
which python
# Doit retourner : ~/dec25-mlops-velib/.venv/bin/python

python --version
# Doit retourner : Python 3.12.x  (ou 3.11.x)
```

### 5. Installer les dépendances

```bash
# Mettre à jour pip
pip install --upgrade pip

# Installer le package shared/ en mode editable
# (toute modification de shared/ est prise en compte sans réinstallation)
pip install -e shared/

# Installer les dépendances du service ML
pip install -r ml/requirements.txt
```

L'installation prend 2-4 minutes selon ta connexion. Tu verras des téléchargements de wheels (`numpy`, `pandas`, `mlflow`, `plotly`, etc.) — c'est normal.

### 6. Configurer le fichier .env

> ⚠️ **Requis avant tout `make`** : sans ce fichier, `docker compose` échoue au démarrage avec une erreur `env file not found`.

```bash
cp .env.example .env
```

Édite le fichier `.env` avec ton éditeur préféré (`nano .env`, `code .env` si VSCode WSL, etc.) et renseigne au minimum :

```env
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

> Pour générer un token HF : https://huggingface.co/settings/tokens → "New token" → type "Read" → copier la valeur.

> **Sécurité** : le fichier `.env` est dans `.gitignore`, il ne sera jamais commité. Ne le partage jamais publiquement.

### 7. Vérifier que tout marche

```bash
python -c "
from shared.config import settings
from shared.logger import get_logger
print('Config OK')
print('  HF repo:', settings.hf_repo)
print('  Raw data dir:', settings.raw_data_dir)
print('  Token chargé:', settings.hf_token is not None)
"
```

Si la sortie affiche `Token chargé: True`, c'est gagné.

---

## Utilisation quotidienne

### Activer le venv (mode debug local uniquement)

> En mode Docker nominal, cette étape n'est pas nécessaire. Le venv n'est utile que pour exécuter les scripts Python directement hors conteneur.

```bash
cd ~/dec25-mlops-velib
source .venv/bin/activate
```

### Lancer le pipeline complet

**En production et en intégration, le pipeline se lance exclusivement via Docker + DVC.**
L'exécution manuelle des scripts est réservée au mode debug local (voir ci-dessous).

#### Mode nominal — Docker (recommandé)

```bash
# Lance dvc repro dans le conteneur ml_training, puis recharge automatiquement
# le modèle dans l'API via POST /model/reload
make pipeline
```

Les 5 stages s'enchaînent dans l'ordre : `load_from_hf → make_dataset → dataviz → build_features → train_model`.
MLflow est joignable via le réseau Docker interne (`mlflow-server:5000`), PostgreSQL est utilisé comme backend.
En fin de pipeline, `make pipeline` appelle automatiquement `POST /model/reload` pour que l'API serve immédiatement le nouveau modèle sans redémarrage.

| Stage | Durée estimée | Sortie principale |
|---|---|---|
| load_from_hf | ~30 s | `data/raw/velib_snapshot_latest.parquet` |
| make_dataset | ~5 s | `data/interim/velib_cleaned_latest.parquet` |
| dataviz | ~10 s | `data/outputs/plots/dataviz_report.html` |
| build_features | ~20 s | `data/processed/{train,test}_preprocessed.parquet` |
| train_model | ~2–5 min | modèle promu alias `staging` dans MLflow Registry |
| reload API | ~2 s | `POST /model/reload` → API passe en `status: ok` |

#### Mode debug local — hors Docker

> ⚠️ **Contrainte connue** : `train_model` se connecte par défaut à `http://mlflow-server:5000`,
> nom de service Docker non résolvable hors du réseau `mlops-net`.
> Les étapes 1 à 4 fonctionnent sans Docker. L'étape 5 requiert de surcharger le tracking URI.

```bash
# Étapes 1 à 4 — exécutables directement depuis le venv local
python -m ml.src.data.load_from_hf
python -m ml.src.data.make_dataset
python -m ml.src.visualization.dataviz
python -m ml.src.features.build_features

# Étape 5 — surcharger le tracking URI pour pointer sur SQLite local
# (pas de serveur MLflow requis, le registre est écrit dans mlflow.db)
MLFLOW_TRACKING_URI=sqlite:///mlflow.db python -m ml.src.models.train_model
```

> En production réelle, le tracking URI serait injecté par l'orchestrateur (Airflow, Prefect…)
> via un Secret Manager centralisé — jamais surclassé manuellement.

### Ouvrir le rapport visuel

Le rapport HTML est autonome (aucun serveur nécessaire). Triple choix pour l'ouvrir :

```bash
# Option A — depuis WSL, avec l'app par défaut Windows (le plus simple)
explorer.exe data/outputs/plots/dataviz_report.html

# Option B — copier le chemin et ouvrir manuellement dans Chrome/Firefox
# Chemin Windows depuis WSL : \\wsl.localhost\Ubuntu\home\voroman\dec25-mlops-velib\data\outputs\plots\dataviz_report.html

# Option C — depuis VSCode (extension Live Preview ou clic droit "Open with Live Server")
```

> **À savoir** : le rapport pèse ~110 Mo sur le dataset complet (5M lignes). Le navigateur prend 5-15 secondes pour le parser à la première ouverture. Si tu rencontres des lenteurs, voir la section "Optimisation du rapport" plus bas.

### Inspecter les sorties intermédiaires

```bash
# Voir les snapshots téléchargés
cat data/raw/.snapshots.log

# Voir les nettoyages effectués
cat data/interim/.cleaning.log

# Lire la taille des fichiers générés
ls -lh data/raw/*.parquet data/interim/*.parquet data/outputs/plots/*.html 2>/dev/null

# Inspecter les parquet en Python
python -c "
import pandas as pd
df = pd.read_parquet('data/interim/velib_cleaned_latest.parquet')
print('Lignes:', len(df))
print('Colonnes:', df.columns.tolist())
print(df.head())
print()
print('Stats taux:')
print(df['taux'].describe())
"
```

### Travailler avec le pipeline DVC

> **Mode debug local uniquement** : les commandes ci-dessous nécessitent un venv actif avec DVC installé (`pip install -r ml/requirements.txt`). En mode Docker nominal, utiliser `make pipeline` à la place.

```bash
dvc dag                # affiche le graphe des 5 stages
dvc status             # liste les stages dont les inputs ont changé
dvc repro              # ré-exécute uniquement ce qui doit l'être
dvc repro load_from_hf # force la ré-exécution d'un stage particulier
dvc metrics show       # affiche les métriques du dernier run (r2, mae, mape)
dvc plots show         # ouvre les graphes de performance (feature importance, résidus)
dvc push               # pousse les artefacts vers le remote DagsHub
dvc pull               # récupère les artefacts depuis DagsHub
```

---

## Le rapport visuel — les 7 graphes

Le `dataviz_report.html` produit par `dataviz.py` regroupe 7 graphiques interactifs en navigation par onglets, conçus pour valider la qualité du dataset et nourrir les choix de modélisation à venir.

| # | Graphe | Donnée source | Question répondue |
|---|---|---|---|
| 1 | Couverture temporelle | parquet **brut** | "Y a-t-il des trous dans la collecte ?" |
| 2 | Distribution du taux | parquet nettoyé | "À quoi ressemble la target ? Hétérogène entre stations ?" |
| 3 | Profils temporels | parquet nettoyé | "Quels sont les patterns horaires et hebdomadaires ?" |
| 4 | Impact météo | parquet nettoyé | "La pluie/orage affecte-t-elle vraiment l'usage ?" |
| 5 | Anomalie thermique | parquet nettoyé | "Un jour anormalement chaud/froid impacte-t-il l'usage ?" |
| 6 | Profils fonctionnels | parquet nettoyé | "Quelles stations sont résidentielles vs bureaux ?" |
| 7 | Impact vacances | parquet nettoyé | "Les vacances scolaires changent-elles le taux moyen ?" |

> **Note** : le graphe 1 utilise volontairement le parquet **brut** (avant nettoyage) parce que les filtres du nettoyage masqueraient les trous de collecte qu'on cherche justement à détecter.

### Réutilisation depuis Streamlit

Les 7 fonctions `render_*()` sont conçues pour être appelées indépendamment et peuvent être intégrées dans le dashboard Streamlit (`streamlit/`) :

```python
import streamlit as st
import pandas as pd
from ml.src.visualization.dataviz import (
    render_temporal_coverage,
    render_fill_rate_distribution,
    render_temporal_patterns,
    render_weather_impact,
    render_temp_anomaly,
    render_station_profiles,
    render_vacation_impact,
)

df_raw = pd.read_parquet("data/raw/velib_snapshot_latest.parquet")
df_clean = pd.read_parquet("data/interim/velib_cleaned_latest.parquet")

st.plotly_chart(render_temporal_coverage(df_raw), use_container_width=True)
st.plotly_chart(render_fill_rate_distribution(df_clean), use_container_width=True)
# ...
```

### Optimisation du rapport (si nécessaire)

Le rapport pèse ~110 Mo sur le dataset actuel (5M lignes). Trois pistes d'optimisation à activer si la taille devient bloquante :

- **Échantillonnage du graphe 1** : downsampling à 1 point par heure au lieu d'1 point par relevé → gain ~40%.
- **Réduction du top-N du graphe 2** : top 10 stations au lieu du top 20 → gain ~10%.
- **Format JSON sérialisé** : exporter chaque figure en JSON et les charger à la demande via JavaScript → gain ~60% à l'ouverture (mais complexifie le code).

Aucune n'est appliquée par défaut pour rester fidèle au code original.

---

## Tests unitaires (Phase 2)

Les tests utilisent **pytest** avec un modèle XGBoost fixture réel (entraîné sur données synthétiques) pour éviter tout appel MLflow pendant la CI.

### Lancer les tests

```bash
# Tests unitaires uniquement (API + ML + shared) — ne nécessite pas la stack complète
make test-unit

# Tests API seuls
make test-unit-api

# Tests ML + shared seuls
make test-unit-ml

# Suite complète (unit + intégration curl — nécessite make up)
make test
```

> **Prérequis** : les images Docker doivent être construites (`make build`) avant de lancer les tests.
> Les cibles `test-unit-*` utilisent `docker compose run --rm --no-deps` — la stack n'a pas besoin d'être démarrée.

### Couverture

| Répertoire | Fichiers | Ce qui est testé |
|---|---|---|
| `api/tests/test_schemas.py` | 13 tests | Validation Pydantic — StationFeatures, BatchPredictionRequest |
| `api/tests/test_endpoints.py` | 14 tests | Tous les endpoints FastAPI via TestClient |
| `ml/tests/test_data_cleaning.py` | 18 tests | Fonctions pures `shared/utils/data_cleaning.py` |
| `ml/tests/test_build_features.py` | 16 tests | Feature engineering — temporal, capacité, résidu, split |
| `ml/tests/test_predict_model.py` | 10 tests | Inférence, clipping, seuils alert_level |
| `shared/tests/test_config.py` | 12 tests | Validateurs Pydantic Settings |

### Stratégie de mock (option B — modèle fixture)

Les tests API injectent un vrai `Pipeline(SimpleImputer + XGBRegressor)` entraîné sur 300 lignes synthétiques.
Trois patches évitent tout appel MLflow/PostgreSQL :
- `velib_api.main.preload_model` → retourne des métadonnées statiques
- `velib_api.dependencies.load_staging_model` → retourne le modèle fixture
- `velib_api.main.load_model_by_alias` → MagicMock avec `cache_clear`

---

## Points de contrôle qualité

Le pipeline produit des **logs structurés** à chaque étape qui permettent de valider la qualité des données.

**Étape 1 — `load_from_hf`** :

```
INFO  Concaténation terminée  rows=4965146  stations=1511  date_min=...  date_max=...
INFO  Parquet écrit          path=...velib_snapshot_latest.parquet  size_mb=8.7
INFO  Snapshot log mis à jour  sha=8516253f
```

**Étape 2 — `make_dataset`** :

```
INFO  Cohérence capacity_status validée  rows_compared=4958081
INFO  Filtre is_renting==True            removed=61965  remaining=4903181
INFO  Filtre capacity > 0                removed=1162   remaining=4902019
INFO  Filtre stations sans variance      excluded_count=1  rows_removed=3286
INFO  Nettoyage terminé                  rows_in=4965146  rows_out=4898733  filter_pct=1.34
```

**Étape 3 — `dataviz`** :

```
INFO  Parquets chargés       raw_rows=4965146  cleaned_rows=4898733  stations=1492
INFO  Génération 1. Couverture
... (7 graphes)
INFO  Rapport HTML écrit     size_mb=112.49  n_figures=7
```

> **Signaux d'alerte** :
> - Si `filter_pct > 5%` → la qualité de la collecte se dégrade, à investiguer côté HF Space.
> - Si `excluded_count > 50 stations` → seuil `MIN_VARIANCE` à reconsidérer.
> - Si le warning "Divergence avec capacity_status du collector" apparaît avec `rows_diverging` non nul → vérifier la formule du collector dans `ingestion_hf.py`.

---

## Variables d'environnement (`.env`)

Toutes les variables sont lues par `shared/config.py` au démarrage. Voir `.env.example` pour la liste complète. Les plus importantes :

| Variable | Défaut | Rôle |
|---|---|---|
| `HF_REPO` | `voroman/velib-ml-data` | Identifiant du repo HF Datasets |
| `HF_TOKEN` | (à renseigner) | Token HF pour l'authentification |
| `HF_FORCE_DOWNLOAD` | `true` | Re-télécharger même si cache local existe |
| `LOG_LEVEL` | `INFO` | Niveau de verbosité (DEBUG, INFO, WARNING, ERROR) |
| `LOG_FORMAT` | `text` | Format des logs (`text` lisible / `json` pour Loki) |
| `RANDOM_STATE` | `42` | Seed pour reproductibilité |
| `TEST_SIZE` | `0.20` | Proportion du test set (split temporel) |
| `MIN_VARIANCE` | `0.5` | Seuil minimum de variance pour garder une station |

---

## Architecture de la collecte (livrable jury)

La collecte continue est l'un des livrables explicites du projet. Elle tourne **24/7** sur des services free tier :

```
┌─────────────────┐  GET /collect    ┌──────────────────────────┐
│  cron-job.org   │ ───toutes 5min──►│  HF Space (collector_app)│
│  (free, 5 min   │                  │  Flask + gunicorn        │
│   minimum)      │                  │  port 7860               │
└─────────────────┘                  └────────────┬─────────────┘
                                                  │ ingest()
                                                  ▼
       ┌──────────────────────────────────────────────────────────┐
       │  4 sources externes + 1 librairie locale                 │
       │  ─────────────────────────────────────────────────────── │
       │  · Vélib station_status.json    (Smovengo)               │
       │  · Vélib station_information.json                        │
       │  · open-meteo.com (apparent_temperature, weather_code)   │
       │  · data.education.gouv.fr (vacances zone Paris)          │
       │  · holidays.France() (jours fériés, lib python locale)   │
       └──────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
                                  ┌────────────────────────────┐
                                  │ HfApi.upload_file()         │
                                  │ → voroman/velib-ml-data     │
                                  │   dataset_velib_raw_NN.csv  │
                                  │   (rotation à 4 Go)         │
                                  └────────────────────────────┘
```

**Coût d'opération** : 0 €/mois (cron-job.org gratuit, HF Space gratuit avec sleep autorisé, HF Datasets gratuit jusqu'à 1 To en public).

**Volumétrie observée** : ~700 Mo/semaine en CSV bruts, soit ~9 Mo/semaine en parquet zstd après concaténation. Le seuil 4 Go par fichier sera atteint vers fin 2026.

---

## Dépannage courant

### `command not found: python`

Tu n'es pas dans WSL, ou ton venv n'est pas activé.

```bash
# Vérifier que tu es bien dans WSL :
uname -a    # doit contenir "Linux"

# Réactiver le venv :
source ~/dec25-mlops-velib/.venv/bin/activate
```

### `ModuleNotFoundError: No module named 'shared'`

Le package `shared/` n'est pas installé.

```bash
cd ~/dec25-mlops-velib
source .venv/bin/activate
pip install -e shared/
```

### `ModuleNotFoundError: No module named 'pandas'` (ou autre lib ML)

Les requirements ne sont pas installés.

```bash
pip install -r ml/requirements.txt
```

### `ModuleNotFoundError: No module named 'plotly'`

Les requirements n'ont pas été ré-installés depuis l'ajout de Plotly.

```bash
pip install -r ml/requirements.txt
```

### `Snapshot brut absent : data/raw/velib_snapshot_latest.parquet`

`make_dataset` ou `dataviz` a été lancé sans avoir préalablement téléchargé le snapshot.

```bash
python -m ml.src.data.load_from_hf
python -m ml.src.data.make_dataset
python -m ml.src.visualization.dataviz
```

### `Parquet nettoyé absent : data/interim/velib_cleaned_latest.parquet`

`dataviz` a été lancé sans avoir nettoyé le snapshot.

```bash
python -m ml.src.data.make_dataset
python -m ml.src.visualization.dataviz
```

### Le téléchargement HF échoue avec `401 Unauthorized`

Le token HF est invalide ou absent.

```bash
# Vérifier que .env contient bien le token
grep HF_TOKEN .env

# Le token doit commencer par hf_
# Régénérer un token Read sur https://huggingface.co/settings/tokens si besoin
```

### Le navigateur freeze à l'ouverture du `dataviz_report.html`

Le rapport est lourd (~110 Mo). Trois solutions :

```bash
# Solution A — utiliser un navigateur récent (Chrome/Firefox/Edge à jour)
# Les versions <2024 peuvent peiner sur les gros HTML.

# Solution B — fermer les autres onglets gourmands en mémoire
# Plotly est gourmand, il a besoin de RAM disponible.

# Solution C — appliquer une des optimisations décrites dans la section "Optimisation du rapport"
```

### `Token chargé: False` et `Raw data dir: /app/data/raw`

Le fichier `.env` n'est pas trouvé. Deux causes possibles :

1. **`.env` absent** — as-tu bien fait `cp .env.example .env` et renseigné `HF_TOKEN` ?
2. **Mauvaise branche** — la branche `main` a un `config.py` avec `APP_DIR=/app` par défaut. Assure-toi d'être sur `add-rectification-global` :
   ```bash
   git branch          # doit afficher * add-rectification-global
   git checkout add-rectification-global
   ```

### Lenteur de l'installation pip sur Windows

Tu n'es **pas** dans WSL. Vérifie l'invite : si elle commence par `PS C:\` ou `\\wsl.localhost\`, tu es côté Windows. Ouvre un vrai terminal Ubuntu.

---

## Conventions du projet

### Nommage des fichiers

- **Snapshots** : `velib_snapshot_latest.parquet` — DVC versionne le contenu, pas le nom
- **Datasets nettoyés** : `velib_cleaned_latest.parquet`
- **Train/test** : `train_preprocessed.parquet`, `test_preprocessed.parquet`
- **Rapports** : `dataviz_report.html`
- **Logs humains** : `.snapshots.log`, `.cleaning.log` (CSV-like, commentés)

### Format des données

- **Parquet** avec compression `zstd` partout
- **Pas de CSV** dans le pipeline (sauf en entrée brute depuis HF)
- **Datetime** typé `datetime64[ns, UTC]`

### Format des graphiques

- **Plotly Graph Objects** (`go`) avec template `plotly_white`
- **Charte couleur Vélib'** centralisée dans `dataviz.py` (variable `COLORS`)
- **Sortie** : HTML autonome avec navigation par onglets, ou objets `Figure` réutilisables depuis Streamlit

### Commits Git

- `feat:` nouvelle fonctionnalité
- `fix:` correction de bug
- `data:` nouveau snapshot ou dataset versionné via DVC
- `docs:` documentation
- `refactor:` refactoring sans changement fonctionnel
- `chore:` tâches diverses (déps, config)

### Branches Git

- `main` : protégée, déploiable
- `add-rectification-global` : branche active (pipeline complet)
- `feat/<nom>` : nouvelles fonctionnalités
- `fix/<nom>` : corrections

---

## Pages d'erreur Nginx personnalisées

Trois pages HTML brandées Vélib' MLOps sont servies par Nginx en cas d'erreur :

| Code | Fichier | Déclencheur |
|------|---------|-------------|
| 404 | `deployments/nginx/errors/404.html` | Route inexistante sur l'API |
| 429 | `deployments/nginx/errors/429.html` | Rate limiting dépassé (> 10 req/s, burst 20) |
| 50x | `deployments/nginx/errors/50x.html` | Erreur interne API ou upstream indisponible |

### Configuration

- `limit_req_status 429` — nginx retourne 429 (au lieu de 503) en cas de rate limiting
- `proxy_intercept_errors on` — nginx intercepte les erreurs upstream sur le bloc API (port 80)
- Les fichiers sont montés en lecture seule : `./deployments/nginx/errors:/etc/nginx/errors:ro`
- `location ^~ /errors/ { root /etc/nginx; internal; }` — bloc interne, non accessible directement depuis le client

---

## Streamlit — Application de démonstration (Phase 3)

Interface de présentation du projet pour le jury DataScientest, accessible via Nginx sur `http://localhost:8501`.

### Structure

```
streamlit/
├── Dockerfile               # Python 3.12-slim + Docker CLI (pour pytest)
├── requirements.txt         # streamlit, requests, pandas, plotly, pyarrow
└── app/
    ├── main.py              # Page Accueil : état des services + architecture + liens
    ├── pages/
    │   ├── 01_Validation.py # Tests fonctionnels par service + pytest runner
    │   └── 02_Prediction.py # MVP métier : prédiction interactive
    └── utils/
        ├── api_client.py    # Appels HTTP vers API, MLflow, Prometheus, Grafana, Airflow
        └── stations.py      # 15 stations de référence + build_features()
```

### Pages

| Page | Contenu |
|------|---------|
| **Accueil** | Status live de chaque service · Graphique continuité HuggingFace (Plotly, cible 288/j) · Architecture · Métriques modèle · Liens natifs |
| **Validation** | Tests fonctionnels par onglet (API, MLflow, Prometheus, Nginx, pytest runner) · Grafana et Airflow : descriptions statiques avec lien direct (pas de duplication des statuts de l'Accueil) |
| **Prédiction** | Sélection station · Contexte temporel et météo · Prédiction POST /predict · Gauge Plotly · Alerte green/yellow/red |

### Commandes

```bash
make streamlit-up      # Démarrer Streamlit (après make up)
make streamlit-down    # Arrêter
make logs-streamlit    # Logs en temps réel
make shell-streamlit   # Shell dans le conteneur
```

### Prérequis

- `HOST_PROJECT_ROOT` et `DOCKER_GID` dans `.env` (mêmes que pour Airflow) — nécessaires pour le pytest runner

---

## Orchestration Airflow (Phase 3)

Airflow orchestre l'exécution quotidienne du pipeline DVC et le rechargement du modèle API.

### Architecture

```
airflow-db (Postgres 15)  ←─── airflow-init (migration + user)
        │
        ▼
airflow-webserver :8080 ──► nginx :8090  ← UI accessible
airflow-scheduler          (LocalExecutor — lance les tâches en sous-process)
        │ Docker socket
        ▼
docker compose run --rm --no-deps ml_training dvc repro
```

### DAG `velib_pipeline` (cron : `0 4 * * *`)

| Tâche | Rôle |
|---|---|
| `check_hf_connectivity` | Vérifie que HuggingFace est joignable |
| `check_mlflow_health` | Vérifie que MLflow répond sur le réseau interne |
| `dvc_repro` | Lance `dvc repro` dans le conteneur `ml_training` (5 stages DVC) |
| `reload_model` | `POST /model/reload` → API charge le nouveau modèle |
| `smoke_test` | `GET /health` → vérifie que le status est `ok` |

### Prérequis avant de démarrer Airflow

Deux variables **obligatoires** dans `.env` :

```bash
# Chemin absolu du projet sur l'hôte (pour DooD)
HOST_PROJECT_ROOT=$(pwd)

# GID du groupe docker (pour accès socket)
DOCKER_GID=$(stat -c '%g' /var/run/docker.sock)
```

### Démarrage

```bash
# 1. Construire l'image Airflow
make build

# 2. Initialiser la base de données Airflow (une seule fois)
make airflow-init

# 3. Démarrer les services Airflow
make airflow-up

# 4. UI disponible sur http://localhost:8090
#    (identifiants définis par AIRFLOW_ADMIN_USER / AIRFLOW_ADMIN_PASSWORD)
```

### Commandes courantes

```bash
make airflow-trigger     # Déclencher le DAG manuellement
make airflow-logs        # Logs du scheduler
make logs-airflow        # Logs du webserver
make airflow-down        # Arrêter Airflow sans toucher au reste de la stack
make shell-airflow       # Shell interactif dans le scheduler
```

> **Ports** : l'UI Airflow est exposée via Nginx sur `http://localhost:${AIRFLOW_PORT}` (défaut : `8090`).
> Aucun service Airflow n'est accessible directement depuis l'extérieur.

---

## Ressources

- **Repo GitHub** : https://github.com/dgando-bit/dec25-mlops-velib
- **HuggingFace dataset** : https://huggingface.co/datasets/voroman/velib-ml-data
- **HuggingFace Space (collector)** : (privé)
- **API Vélib' Open Data** : https://www.velib-metropole.fr/donnees-open-data-gbfs-du-service-velib-metropole

---

## Équipe

Projet fil rouge de la formation Machine Learning Engineer (promo Décembre 2025).

Développé par l'équipe MLOps Vélib'.
