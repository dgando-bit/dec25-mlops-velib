# 🚲 Vélib MLOps

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
├── api/              # API FastAPI (inférence)
├── ml/               # Module ML (entraînement, features)
│   ├── notebooks/    # Notebooks Jupyter
│   └── src/
│       ├── features/ # Feature engineering
│       └── models/   # Modèles entraînés
├── mlflow/           # Serveur MLflow
├── shared/           # Package Python partagé (config, logger, utils)
├── data/
│   ├── raw/          # Données brutes
│   └── processed/    # Données transformées
├── deployments/      # Nginx, Prometheus
├── docker-compose.yml
├── Makefile
└── .env.example
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