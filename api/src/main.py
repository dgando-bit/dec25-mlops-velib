"""
velib_api.main — Point d'entrée de l'API FastAPI Vélib'.

Responsabilités :
    - instanciation de l'app FastAPI
    - lifespan : chargement du modèle MLflow au démarrage
    - enregistrement des routers
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI

from shared.logger import get_logger
from api.src.routers import health, predict
from api.src.services.model import load_model

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# LIFESPAN — startup / shutdown
# ─────────────────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Gère le cycle de vie de l'application.

    Startup : charge le modèle MLflow une seule fois.
    Shutdown : libère les ressources si nécessaire.
    """
    logger.info("Démarrage de l'API Vélib'...")
    try:
        load_model()
        logger.info("API prête.")
    except RuntimeError as e:
        # Le modèle n'est pas disponible — l'API démarre quand même
        # mais /ready retournera False jusqu'à ce qu'il soit chargé
        logger.warning(
            "Modèle MLflow non disponible au démarrage — %s. "
            "L'API démarre en mode dégradé.",
            str(e),
        )

    yield

    logger.info("Arrêt de l'API Vélib'.")


# ─────────────────────────────────────────────────────────────────────────────
# APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Vélib' MLOps API",
    description="API d'inférence pour la prédiction de disponibilité des stations Vélib'.",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Routers
app.include_router(health.router)
app.include_router(predict.router, prefix="/api/v1")