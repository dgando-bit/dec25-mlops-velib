# 🚲 Vélib MLOp

Projet MLOps de prédiction de disponibilité des stations Vélib.


## Stack

| Service | Rôle | Port |
|---|---|---|
| PostgreSQL | Base de données MLflow | — |
| MLflow | Tracking des expériences ML | `5001` |
| API FastAPI | Endpoints d'inférence | `8000` |
| Jupyter | Notebooks d'exploration | `8888` |

---

## 🚀 Démarrage rapide

### 1. Prérequis

- [Docker](https://www.docker.com/) >= 24
- [Docker Compose](https://docs.docker.com/compose/) >= 2
- `make`

### 2. Configuration

```bash
cp .env.example .env
```

Édite `.env` et renseigne les valeurs :

```bash
MLFLOW_DB=mlflow
MLFLOW_USER=mlflow
MLFLOW_PASSWORD=ton_mot_de_passe
MLFLOW_PORT=5001
JUPYTER_TOKEN=ton_token_jupyter
```

### 3. Build & démarrage

```bash
# Vérifier la config et créer les dossiers
make setup

# Construire les images Docker
make build

# Démarrer tous les services
make up
```

### 4. Vérifier que tout fonctionne

```bash
make health
```

Tu dois obtenir ✓ sur les 4 services.

---

## 🌐 Accès aux interfaces

| Interface | URL |
|---|---|
| MLflow UI | http://localhost:5001 |
| API docs (Swagger) | http://localhost:8000/docs |
| Jupyter Lab | http://localhost:8888/lab?token=`<JUPYTER_TOKEN>` |

---

## 📋 Commandes utiles

```bash
# Démarrer uniquement l'infrastructure (DB + MLflow)
make up-infra

# Démarrer uniquement Jupyter
make up-jupyter

# Suivre les logs d'un service
make logs-api
make logs-mlflow
make logs-jupyter

# Ouvrir un shell dans un conteneur
make shell-api
make shell-ml

# Lancer un entraînement
make train

# Arrêter tous les services
make down

# Voir toutes les commandes disponibles
make help
```

---

## 🗂️ Structure du projet

```
.
├── Makefile
├── README.md
├── docker-compose.yml
├── api/                          # API FastAPI (inférence)
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
├── ml/                           # Module ML
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── notebooks/                # Notebooks Jupyter d'exploration
│   └── src/
│       ├── data/
│       │   └── make_dataset.py   # Ingestion et nettoyage
│       ├── features/
│       │   └── build_features.py # Feature engineering
│       ├── models/
│       │   ├── train_model.py    # Entraînement
│       │   └── predict_model.py  # Inférence
│       └── main.py
├── mlflow/                       # Serveur de tracking MLflow
│   ├── Dockerfile
│   ├── requirements.txt
│   └── artifacts/                # Artefacts modèles (ignoré par Git)
├── shared/                       # Package Python partagé
│   ├── pyproject.toml
│   └── shared/
│       ├── config.py
│       ├── logger.py
│       └── utils/
│           └── helpers.py
├── data/                         # Données versionnées par DVC
│   ├── raw/                      # Données brutes (ignoré par Git)
│   │   └── dataset_velib_300326.csv
│   ├── raw.dvc                   # Métadonnées DVC (suivi par Git)
│   ├── processed/                # Données transformées (ignoré par Git)
│   │   ├── train_preprocessed.csv
│   │   └── test_preprocessed.csv
│   └── processed.dvc             # Métadonnées DVC (suivi par Git)
└── deployments/                  # Configuration infrastructure
    ├── nginx/
    └── prometheus/
        └── prometheus.yml
```

> Les dossiers `data/raw/` et `data/processed/` sont ignorés par Git et versionnés via **DVC** sur DagsHub.

---

## 📦 Données (DVC)

Les données sont versionnées avec [DVC](https://dvc.org) et stockées sur [DagsHub](https://dagshub.com).

### Premier setup (nouveaux membres)

```bash
# Configurer les credentials DagsHub
dvc remote modify origin --local auth basic
dvc remote modify origin --local user TON_USERNAME
dvc remote modify origin --local password TON_TOKEN_DAGSHUB

# Récupérer les données
make dvc-pull
```

### Ajouter de nouvelles données

```bash
# Copier les fichiers dans data/raw/ ou data/processed/
make dvc-add
git add data/raw.dvc data/processed.dvc
git commit -m "data: description du changement"
make dvc-push
git push
```

---

## 🧹 Nettoyage

```bash
# Arrêter et supprimer les conteneurs
make down

# Supprimer aussi les images locales
make clean

# Supprimer également les volumes (⚠️ perte des données MLflow)
make clean-volumes
```
