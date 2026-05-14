import sys

from shared.logger import get_logger
from ml.src.models.train_model import main

logger = get_logger(__name__)

if __name__ == "__main__":
    sys.exit(main())
