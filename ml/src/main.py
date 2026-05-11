from shared.logger import get_logger
from ml.src.models.train_model import train_model

logger = get_logger(__name__)

if __name__ == "__main__":
    logger.info("Lancement du script d'entraînement...")
    try:
        metrics = train_model()
        logger.info(f"✓ Entraînement terminé. Métriques : {metrics}")
    except Exception as e:
        logger.error(f"✗ Erreur lors de l'entraînement : {e}")
        raise
