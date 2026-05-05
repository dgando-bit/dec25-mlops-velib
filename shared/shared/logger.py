"""
shared.logger — Logger structuré pour tous les services Vélib'.

Caractéristiques :
    - format text (humain, défaut) ou json (machine, pour Loki/OpenTelemetry)
    - niveau configurable via settings.log_level
    - configuration idempotente (peut être appelé plusieurs fois sans dupliquer)
    - compatible avec ``logger.info("msg", extra={"clé": "valeur"})``

Usage :
    from shared.logger import get_logger
    logger = get_logger(__name__)
    logger.info("Téléchargement démarré", extra={"repo": "voroman/velib-ml-data"})

Format text (exemple) :
    2026-05-04 14:32:18 INFO  load_from_hf  Téléchargement démarré  repo=voroman/...

Format json (exemple, une ligne par log) :
    {"ts": "2026-05-04T14:32:18+00:00", "level": "INFO",
     "logger": "load_from_hf", "msg": "Téléchargement démarré",
     "repo": "voroman/velib-ml-data"}
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any

from shared.config import settings

# Marqueur global pour ne configurer qu'une fois la racine logging
_CONFIGURED = False

# Champs standards du LogRecord — on extrait tout le reste comme "extra"
_STANDARD_FIELDS = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "asctime", "taskName",
}


class _JsonFormatter(logging.Formatter):
    """Formate chaque LogRecord en une ligne JSON.

    Toutes les clés passées via ``extra=`` apparaissent à la racine du JSON.
    """

    def format(self, record: logging.LogRecord) -> str:
        base: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Récupère les "extra" (tout ce qui n'est pas un champ standard)
        for k, v in record.__dict__.items():
            if k not in _STANDARD_FIELDS and not k.startswith("_"):
                # pour rester JSON-safe, on str() tout ce qui n'est pas trivial
                base[k] = v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


class _TextFormatter(logging.Formatter):
    """Format humain : timestamp niveau logger message  k1=v1 k2=v2.

    Les champs ``extra`` sont concaténés en suffixe, séparés par deux espaces.
    """

    _BASE_FMT = "%(asctime)s %(levelname)-5s %(name)s  %(message)s"
    _DATE_FMT = "%Y-%m-%d %H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._BASE_FMT, datefmt=self._DATE_FMT)

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        # Suffixe avec les extras
        extras = {
            k: v for k, v in record.__dict__.items()
            if k not in _STANDARD_FIELDS and not k.startswith("_")
        }
        if extras:
            kv = "  ".join(f"{k}={v}" for k, v in extras.items())
            return f"{base}  {kv}"
        return base


def _configure_root() -> None:
    """Configure le logger racine une seule fois.

    Réécrit les handlers existants pour éviter les doublons si on est appelé
    plusieurs fois (ex. tests, reload).
    """
    global _CONFIGURED
    root = logging.getLogger()
    root.setLevel(settings.log_level)
    # Nettoyer les handlers déjà installés
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(stream=sys.stdout)
    if settings.log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(_TextFormatter())
    root.addHandler(handler)

    # Réduire le bruit de quelques librairies tierces verbeuses
    for noisy in ("urllib3", "huggingface_hub.file_download", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Retourne un logger configuré, prêt à l'emploi.

    Le premier appel configure le root logger. Les appels suivants sont
    idempotents.

    Args:
        name: nom du logger, typiquement ``__name__`` du module appelant.

    Returns:
        logging.Logger configuré selon ``settings.log_format`` et
        ``settings.log_level``.
    """
    if not _CONFIGURED:
        _configure_root()
    return logging.getLogger(name)