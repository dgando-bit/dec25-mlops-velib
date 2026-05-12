"""
shared — Code partagé entre les services du projet Vélib' MLOps.

Modules disponibles :
    - shared.config   : configuration Pydantic (settings unique pour tous les services)
    - shared.logger   : logger structuré (text/json) configurable par variable d'env
    - shared.utils    : helpers réutilisables (à venir)

Usage :
    from shared.config import settings
    from shared.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Hello", extra={"raw_dir": str(settings.raw_data_dir)})
"""

from shared.config import settings
from shared.logger import get_logger

__version__ = "0.1.0"
__all__ = ["settings", "get_logger"]