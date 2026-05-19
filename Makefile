# =============================================================================
# MAKEFILE — Projet MLOps Vélib
# =============================================================================
# Usage : make <cible>
# Toutes les cibles sont documentées via `make help`

# --- Configuration -----------------------------------------------------------

COMPOSE         := docker compose
COMPOSE_FILE    := docker-compose.yml
ENV_FILE        := .env

# Couleurs pour les logs
RESET  := \033[0m
BOLD   := \033[1m
GREEN  := \033[32m
YELLOW := \033[33m
RED    := \033[31m
CYAN   := \033[36m

# Services définis dans le docker-compose
SERVICES := mlflow-db mlflow-server ml_training jupyter-service api

.DEFAULT_GOAL := help
.PHONY: help \
        build build-nocache \
        up down restart \
        up-infra up-ml up-api up-jupyter \
        logs logs-api logs-ml logs-mlflow logs-jupyter \
        ps status \
        health \
        test test-api test-mlflow \
        train \
        clean clean-volumes clean-all \
        shell-api shell-ml shell-jupyter \
        check-env setup \
        dvc-pull dvc-push dvc-status dvc-add pipeline \
        logs-airflow shell-airflow airflow-init


# =============================================================================
# AIDE
# =============================================================================

help:
	@echo ""
	@echo "$(BOLD)$(CYAN)╔══════════════════════════════════════════════════╗$(RESET)"
	@echo "$(BOLD)$(CYAN)║         Projet MLOps Vélib — Makefile           ║$(RESET)"
	@echo "$(BOLD)$(CYAN)╚══════════════════════════════════════════════════╝$(RESET)"
	@echo ""
	@echo "$(BOLD)📦 Build$(RESET)"
	@echo "  $(GREEN)make build$(RESET)            Construire toutes les images"
	@echo "  $(GREEN)make build-nocache$(RESET)    Construire sans cache (rebuild complet)"
	@echo ""
	@echo "$(BOLD)🚀 Démarrage$(RESET)"
	@echo "  $(GREEN)make up$(RESET)               Démarrer tous les services"
	@echo "  $(GREEN)make up-infra$(RESET)         Démarrer uniquement mlflow-db + mlflow-server"
	@echo "  $(GREEN)make up-ml$(RESET)            Démarrer infra + ml_training"
	@echo "  $(GREEN)make up-api$(RESET)           Démarrer infra + api"
	@echo "  $(GREEN)make up-jupyter$(RESET)       Démarrer infra + jupyter"
	@echo ""
	@echo "$(BOLD)🛑 Arrêt$(RESET)"
	@echo "  $(GREEN)make down$(RESET)             Arrêter et supprimer les conteneurs"
	@echo "  $(GREEN)make restart$(RESET)          Redémarrer tous les services"
	@echo ""
	@echo "$(BOLD)📋 Logs$(RESET)"
	@echo "  $(GREEN)make logs$(RESET)             Logs de tous les services (follow)"
	@echo "  $(GREEN)make logs-api$(RESET)         Logs de l'API"
	@echo "  $(GREEN)make logs-ml$(RESET)          Logs du module ML"
	@echo "  $(GREEN)make logs-mlflow$(RESET)      Logs du serveur MLflow"
	@echo "  $(GREEN)make logs-jupyter$(RESET)     Logs de Jupyter"
	@echo ""
	@echo "$(BOLD)🏥 Santé & Tests$(RESET)"
	@echo "  $(GREEN)make status$(RESET)           État de tous les conteneurs"
	@echo "  $(GREEN)make health$(RESET)           Vérifier la santé de tous les services"
	@echo "  $(GREEN)make test$(RESET)             Lancer tous les tests"
	@echo "  $(GREEN)make test-api$(RESET)         Tester l'API (endpoints)"
	@echo "  $(GREEN)make test-mlflow$(RESET)      Tester la connexion MLflow"
	@echo ""
	@echo "$(BOLD)🤖 ML$(RESET)"
	@echo "  $(GREEN)make train$(RESET)            Lancer un job d'entraînement"
	@echo ""
	@echo "$(BOLD)📦 DVC$(RESET)"
	@echo "  $(GREEN)make dvc-pull$(RESET)         Récupérer les données depuis DagsHub"
	@echo "  $(GREEN)make dvc-push$(RESET)         Envoyer les données vers DagsHub"
	@echo "  $(GREEN)make dvc-status$(RESET)       Vérifier l'état des données DVC"
	@echo "  $(GREEN)make dvc-add$(RESET)          Tracker les changements dans data/raw/ et data/processed/"
	@echo "  $(GREEN)make pipeline$(RESET)         Exécuter le pipeline DVC complet"
	@echo ""
	@echo "$(BOLD)🐚 Shells$(RESET)"
	@echo "  $(GREEN)make shell-api$(RESET)        Shell interactif dans le conteneur API"
	@echo "  $(GREEN)make shell-ml$(RESET)         Shell interactif dans le conteneur ML"
	@echo "  $(GREEN)make shell-jupyter$(RESET)    Shell interactif dans le conteneur Jupyter"
	@echo ""
	@echo "$(BOLD)🌀 Airflow$(RESET)"
	@echo "  $(GREEN)make logs-airflow$(RESET)     Logs du scheduler Airflow"
	@echo "  $(GREEN)make shell-airflow$(RESET)    Shell interactif dans le scheduler"
	@echo "  $(GREEN)make airflow-init$(RESET)     Initialiser la DB Airflow (premier démarrage)"
	@echo ""
	@echo "$(BOLD)🧹 Nettoyage$(RESET)"
	@echo "  $(GREEN)make clean$(RESET)            Supprimer les conteneurs et images du projet"
	@echo "  $(GREEN)make clean-volumes$(RESET)    Supprimer aussi les volumes (⚠️  perte de données)"
	@echo "  $(GREEN)make clean-all$(RESET)        Nettoyage complet Docker système"
	@echo ""
	@echo "$(BOLD)⚙️  Setup$(RESET)"
	@echo "  $(GREEN)make check-env$(RESET)        Vérifier que le fichier .env est présent"
	@echo "  $(GREEN)make setup$(RESET)            Initialiser le projet (env + dossiers)"
	@echo ""


# =============================================================================
# SETUP & VÉRIFICATIONS
# =============================================================================

check-env:
	@echo "$(CYAN)→ Vérification du fichier .env...$(RESET)"
	@if [ ! -f "$(ENV_FILE)" ]; then \
		echo "$(RED)✗ Fichier .env introuvable !$(RESET)"; \
		echo "  Copie le template : cp .env.example .env"; \
		exit 1; \
	fi
	@echo "$(GREEN)✓ .env présent$(RESET)"
	@echo "$(CYAN)→ Vérification des variables requises...$(RESET)"
	@for var in MLFLOW_DB MLFLOW_USER MLFLOW_PASSWORD JUPYTER_TOKEN; do \
		if ! grep -q "^$$var=" $(ENV_FILE); then \
			echo "$(RED)✗ Variable manquante dans .env : $$var$(RESET)"; \
			exit 1; \
		fi; \
	done
	@echo "$(GREEN)✓ Toutes les variables requises sont présentes$(RESET)"

setup: check-env
	@echo "$(CYAN)→ Création des dossiers nécessaires...$(RESET)"
	@mkdir -p data/raw data/processed data/interim mlflow/artifacts ml/notebooks
	@mkdir -p airflow/dags airflow/logs airflow/plugins
	@echo "$(GREEN)✓ Dossiers créés$(RESET)"
	@echo "$(CYAN)→ Vérification de Docker...$(RESET)"
	@docker info > /dev/null 2>&1 || (echo "$(RED)✗ Docker n'est pas lancé$(RESET)" && exit 1)
	@echo "$(GREEN)✓ Docker opérationnel$(RESET)"
	@echo "$(GREEN)✓ Setup terminé — lance : make build && make up$(RESET)"


# =============================================================================
# BUILD
# =============================================================================

build: check-env
	@echo "$(CYAN)→ Build de toutes les images...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) build
	@echo "$(GREEN)✓ Build terminé$(RESET)"

build-nocache: check-env
	@echo "$(YELLOW)→ Build sans cache (peut prendre plusieurs minutes)...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) build --no-cache
	@echo "$(GREEN)✓ Build terminé$(RESET)"


# =============================================================================
# DÉMARRAGE
# =============================================================================

up: check-env
	@echo "$(CYAN)→ Démarrage de tous les services...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) up -d
	@echo "$(GREEN)✓ Services démarrés$(RESET)"
	@$(MAKE) --no-print-directory status

up-infra: check-env
	@echo "$(CYAN)→ Démarrage de l'infrastructure (DB + MLflow)...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) up -d mlflow-db mlflow-server
	@echo "$(GREEN)✓ Infrastructure démarrée$(RESET)"

up-ml: up-infra
	@echo "$(CYAN)→ Démarrage du module ML...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) up -d ml_training
	@echo "$(GREEN)✓ ML démarré$(RESET)"

up-api: up-infra
	@echo "$(CYAN)→ Démarrage de l'API...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) up -d api
	@echo "$(GREEN)✓ API démarrée$(RESET)"

up-jupyter: up-infra
	@echo "$(CYAN)→ Démarrage de Jupyter...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) up -d jupyter-service
	@sleep 2
	@echo "$(GREEN)✓ Jupyter démarré$(RESET)"
	@echo "$(CYAN)  URL : http://localhost:8888$(RESET)"
	@echo "$(CYAN)  Token : $$(grep JUPYTER_TOKEN $(ENV_FILE) | cut -d= -f2)$(RESET)"


# =============================================================================
# ARRÊT
# =============================================================================

down:
	@echo "$(YELLOW)→ Arrêt des services...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) down
	@echo "$(GREEN)✓ Services arrêtés$(RESET)"

restart: down up


# =============================================================================
# LOGS
# =============================================================================

logs:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f

logs-api:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f api

logs-ml:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f ml_training

logs-mlflow:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f mlflow-server

logs-jupyter:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f jupyter-service


# =============================================================================
# STATUT & SANTÉ
# =============================================================================

ps: status

status:
	@echo ""
	@echo "$(BOLD)État des conteneurs :$(RESET)"
	@$(COMPOSE) -f $(COMPOSE_FILE) ps
	@echo ""

health:
	@echo ""
	@echo "$(BOLD)$(CYAN)══ Vérification de santé des services ══$(RESET)"
	@echo ""

	@echo "$(BOLD)[1/4] PostgreSQL (mlflow-db)$(RESET)"
	@if docker exec dec25-mlops-mlflow-db pg_isready -U $$(grep MLFLOW_USER $(ENV_FILE) | cut -d= -f2) > /dev/null 2>&1; then \
		echo "  $(GREEN)✓ PostgreSQL opérationnel$(RESET)"; \
	else \
		echo "  $(RED)✗ PostgreSQL ne répond pas$(RESET)"; \
	fi

	@echo "$(BOLD)[2/4] MLflow Server$(RESET)"
	@if curl -sf http://localhost:5001/health > /dev/null 2>&1; then \
		echo "  $(GREEN)✓ MLflow opérationnel$(RESET)"; \
	elif curl -sf http://localhost:5001 > /dev/null 2>&1; then \
		echo "  $(GREEN)✓ MLflow répond (pas d'endpoint /health)$(RESET)"; \
	else \
		echo "  $(RED)✗ MLflow ne répond pas sur :5001$(RESET)"; \
	fi

	@echo "$(BOLD)[3/4] API$(RESET)"
	@if curl -sf http://localhost:8000/health > /dev/null 2>&1; then \
		echo "  $(GREEN)✓ API opérationnelle$(RESET)"; \
		curl -s http://localhost:8000/health | python3 -m json.tool 2>/dev/null | sed 's/^/     /'; \
	else \
		echo "  $(RED)✗ API ne répond pas sur :8000$(RESET)"; \
		echo "  $(YELLOW)  → Lance : make logs-api$(RESET)"; \
	fi

	@echo "$(BOLD)[4/4] Jupyter$(RESET)"
	@if curl -sf http://localhost:8888 > /dev/null 2>&1; then \
		echo "  $(GREEN)✓ Jupyter opérationnel$(RESET)"; \
	else \
		echo "  $(YELLOW)⚠ Jupyter ne répond pas (normal s'il n'est pas démarré)$(RESET)"; \
	fi
	@echo ""


# =============================================================================
# TESTS
# =============================================================================

test: test-mlflow test-api
	@echo "$(GREEN)✓ Tous les tests passés$(RESET)"

test-mlflow:
	@echo "$(CYAN)→ Test de connexion MLflow...$(RESET)"
	@curl -sf http://localhost:5001/api/2.0/mlflow/experiments/list > /dev/null 2>&1 \
		&& echo "  $(GREEN)✓ MLflow API répond$(RESET)" \
		|| echo "  $(RED)✗ MLflow API inaccessible$(RESET)"
	@echo "$(CYAN)→ Test de connexion à la DB depuis mlflow-server...$(RESET)"
	@docker exec dec25-mlops-mlflow-server python -c \
		"import mlflow; mlflow.set_tracking_uri('http://localhost:5000'); print('  $(GREEN)✓ MLflow client OK$(RESET)')" \
		2>/dev/null || echo "  $(RED)✗ Erreur client MLflow$(RESET)"

test-api:
	@echo "$(CYAN)→ Test des endpoints API...$(RESET)"
	@echo "  GET /health"
	@curl -sf -w "\n  Status: %{http_code}\n" http://localhost:8000/health 2>&1 | sed 's/^/  /' \
		|| echo "  $(RED)✗ /health inaccessible$(RESET)"
	@echo "  GET /docs"
	@curl -sf -o /dev/null -w "  Status: %{http_code}\n" http://localhost:8000/docs \
		|| echo "  $(RED)✗ /docs inaccessible$(RESET)"


# =============================================================================
# ML
# =============================================================================

train:
	@echo "$(CYAN)→ Lancement d'un job d'entraînement...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) run --rm ml_training python ml/src/main.py
	@echo "$(GREEN)✓ Entraînement terminé$(RESET)"


# =============================================================================
# DVC
# =============================================================================

dvc-pull:
	@echo "$(CYAN)→ Récupération des données depuis DagsHub...$(RESET)"
	dvc pull
	@echo "$(GREEN)✓ Données récupérées$(RESET)"

dvc-push:
	@echo "$(CYAN)→ Push des données vers DagsHub...$(RESET)"
	dvc push
	@echo "$(GREEN)✓ Données envoyées$(RESET)"

dvc-status:
	@echo "$(CYAN)→ État des données DVC...$(RESET)"
	dvc status

dvc-add:
	@echo "$(CYAN)→ Tracking des changements dans data/...$(RESET)"
	dvc add data/raw/ data/processed/
	@echo "$(GREEN)✓ Fichiers trackés$(RESET)"
	@echo "$(YELLOW)  → N'oublie pas : git add data/raw.dvc data/processed.dvc && git commit$(RESET)"

pipeline:
	@echo "$(CYAN)→ Exécution du pipeline DVC...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) run --rm ml_training dvc repro
	@echo "$(GREEN)✓ Pipeline terminé$(RESET)"


# =============================================================================
# SHELLS INTERACTIFS
# =============================================================================

shell-api:
	docker exec -it velib_api /bin/bash

shell-ml:
	docker exec -it velib_ml /bin/bash

shell-jupyter:
	docker exec -it dec25-mlops-jupyter /bin/bash


# =============================================================================
# AIRFLOW
# =============================================================================

logs-airflow:
	$(COMPOSE) -f $(COMPOSE_FILE) logs -f airflow-scheduler airflow-webserver

shell-airflow:
	docker exec -it dec25-mlops-airflow-scheduler /bin/bash

airflow-init:
	@echo "$(CYAN)→ Initialisation de la base Airflow...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) run --rm airflow-init
	@echo "$(GREEN)✓ Airflow initialisé$(RESET)"


# =============================================================================
# NETTOYAGE
# =============================================================================

clean:
	@echo "$(YELLOW)→ Suppression des conteneurs et images du projet...$(RESET)"
	$(COMPOSE) -f $(COMPOSE_FILE) down --rmi local
	@echo "$(GREEN)✓ Nettoyage terminé$(RESET)"

clean-volumes:
	@echo "$(RED)⚠️  Cette commande supprime les volumes (données PostgreSQL perdues !)$(RESET)"
	@read -p "Confirmer ? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(COMPOSE) -f $(COMPOSE_FILE) down -v --rmi local
	@echo "$(GREEN)✓ Volumes supprimés$(RESET)"

clean-all:
	@echo "$(RED)⚠️  Nettoyage complet du système Docker$(RESET)"
	@read -p "Confirmer ? [y/N] " confirm && [ "$$confirm" = "y" ] || exit 1
	$(COMPOSE) -f $(COMPOSE_FILE) down -v --rmi all
	docker system prune -f
	@echo "$(GREEN)✓ Système nettoyé$(RESET)"