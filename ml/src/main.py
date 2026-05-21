# import sys
#
# from ml.src.models.train_model import main
#
# if __name__ == "__main__":
#     sys.exit(main())
from shared.logger import get_logger

logger = get_logger(__name__)  # __name__ = nom du module automatiquement
logger.info("Lancement du module ML...")