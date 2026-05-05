"""
ml.src.data.load_from_hf — Téléchargement du dataset brut depuis HuggingFace.

Refactoring de l'ancien ``load_all_raw.py`` adapté à l'architecture micro-services.

Différences avec l'ancienne version :
    - aucun secret hardcodé : token lu uniquement via env (HF_TOKEN)
    - sortie en parquet (compression zstd) au lieu de CSV
    - retry exponentiel via tenacity (configurable)
    - pas de mutation du fichier .env (idempotent, compatible Airflow)
    - logs structurés (text ou json selon shared.logger)
    - log humain consigné dans data/raw/.snapshots.log à chaque exécution
    - configuration via shared.config (Pydantic Settings)

Pipeline DVC :
    Stage  : load_from_hf (premier stage du pipeline)
    Sortie : data/raw/velib_snapshot_latest.parquet

Usage :
    # En CLI
    python -m ml.src.data.load_from_hf

    # En import depuis un autre module
    from ml.src.data.load_from_hf import load_from_hf
    df = load_from_hf()

Limites connues :
    - télécharge l'intégralité du dataset à chaque exécution (force_download=True)
    - acceptable tant que le dataset reste sous ~2 Go (cf. settings.hf_force_download)
    - à basculer à False quand la rotation HF aura créé un 2e fichier
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.errors import (
    EntryNotFoundError,
    GatedRepoError,
    RepositoryNotFoundError,
)
from huggingface_hub.utils import HfHubHTTPError
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential,
)

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLASSIFICATION DES ERREURS HF
# ─────────────────────────────────────────────────────────────────────────────
# Erreurs irrécupérables : on échoue tout de suite, retry inutile
_NON_RETRYABLE = (
    RepositoryNotFoundError,   # 404 sur le repo
    EntryNotFoundError,        # 404 sur un fichier
    GatedRepoError,            # 403 (token absent ou repo privé)
    PermissionError,
    ValueError,
)


def _is_retryable(exc: BaseException) -> bool:
    """Décide si une exception justifie un retry.

    Stratégie :
        - HfHubHTTPError : retry uniquement sur 429 (rate limit) ou 5xx
        - les exceptions réseau usuelles (ConnectionError, TimeoutError, OSError)
          sont retryables par défaut
        - tout le reste (404, 403, ValueError, ...) : pas de retry
    """
    if isinstance(exc, _NON_RETRYABLE):
        return False
    if isinstance(exc, HfHubHTTPError):
        # On regarde le status_code de la réponse sous-jacente si disponible
        resp = getattr(exc, "response", None)
        status = getattr(resp, "status_code", None) if resp is not None else None
        if status is None:
            return True   # incertain → on retente
        return status == 429 or 500 <= status < 600
    # Erreurs réseau bas niveau / OS — retryables
    if isinstance(exc, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# OPÉRATIONS HF AVEC RETRY
# ─────────────────────────────────────────────────────────────────────────────
def _retry_decorator():
    """Construit le décorateur tenacity à partir des settings."""
    return retry(
        retry=retry_if_exception(_is_retryable),
        stop=stop_after_attempt(settings.hf_max_retries),
        wait=wait_exponential(
            multiplier=settings.hf_retry_initial_wait_s,
            min=settings.hf_retry_initial_wait_s,
            max=120.0,
        ),
        reraise=True,
        before_sleep=lambda rs: logger.warning(
            "Retry HF",
            extra={
                "attempt": rs.attempt_number,
                "max": settings.hf_max_retries,
                "wait_s": round(rs.next_action.sleep, 1) if rs.next_action else None,
                "exc": type(rs.outcome.exception()).__name__ if rs.outcome else None,
            },
        ),
    )


# (l'import de retry_if_exception est en haut du fichier avec les autres imports tenacity)


def _list_raw_files() -> list[str]:
    """Liste les fichiers raw du dataset HF, triés alphabétiquement.

    Garantit l'ordre lexicographique, qui est aussi l'ordre temporel tant
    que la nomenclature ``dataset_velib_raw_NN.csv`` (NN sur 2 chiffres) tient.

    Returns:
        Liste triée des noms de fichiers (chemins relatifs au repo HF).

    Raises:
        FileNotFoundError: aucun fichier raw n'a été trouvé.
        HfHubHTTPError, RepositoryNotFoundError: après retries épuisés.
    """
    @_retry_decorator()
    def _call() -> list[str]:
        return list(list_repo_files(
            repo_id=settings.hf_repo,
            repo_type="dataset",
            token=settings.hf_token_value(),
        ))

    all_files = _call()
    raw_files = sorted(
        f for f in all_files
        if f.startswith(settings.hf_file_prefix) and f.endswith(".csv")
    )

    if not raw_files:
        raise FileNotFoundError(
            f"Aucun fichier {settings.hf_file_prefix}*.csv trouvé sur {settings.hf_repo}. "
            "Vérifier que la collecte HF Space tourne bien."
        )

    logger.info(
        "Fichiers raw détectés",
        extra={"count": len(raw_files), "files": ",".join(raw_files)},
    )
    return raw_files


def _download_one(filename: str, cache_dir: Path) -> Path:
    """Télécharge un fichier du dataset HF avec retry.

    Args:
        filename: nom du fichier dans le repo HF.
        cache_dir: dossier local où placer le fichier téléchargé.

    Returns:
        Chemin local du fichier téléchargé.
    """
    @_retry_decorator()
    def _call() -> str:
        return hf_hub_download(
            repo_id=settings.hf_repo,
            filename=filename,
            repo_type="dataset",
            token=settings.hf_token_value(),
            force_download=settings.hf_force_download,
            local_dir=str(cache_dir),
        )

    local_path = Path(_call())
    logger.debug(
        "Fichier téléchargé",
        extra={"filename": filename, "size_mb": round(local_path.stat().st_size / 1e6, 1)},
    )
    return local_path


# ─────────────────────────────────────────────────────────────────────────────
# CONCATÉNATION + DÉDOUBLONNAGE
# ─────────────────────────────────────────────────────────────────────────────
def _read_and_concat(files: list[Path]) -> pd.DataFrame:
    """Lit chaque CSV téléchargé et concatène en un seul DataFrame.

    Conserve la logique de l'ancien ``load_all_raw`` :
        - parse_dates=["datetime"]
        - drop_duplicates sur (station_id, datetime)
        - tri par (station_id, datetime)
    """
    dfs: list[pd.DataFrame] = []
    for path in files:
        df = pd.read_csv(path, parse_dates=["datetime"])
        dfs.append(df)
        logger.info(
            "CSV chargé",
            extra={"file": path.name, "rows": len(df)},
        )

    combined = (
        pd.concat(dfs, ignore_index=True)
        .drop_duplicates(subset=["station_id", "datetime"])
        .sort_values(["station_id", "datetime"])
        .reset_index(drop=True)
    )
    return combined


# ─────────────────────────────────────────────────────────────────────────────
# CONSIGNATION DU SNAPSHOT (log humain en plus du parquet)
# ─────────────────────────────────────────────────────────────────────────────
def _append_snapshot_log(df: pd.DataFrame, output_path: Path) -> None:
    """Append une ligne dans data/raw/.snapshots.log pour traçabilité humaine.

    Format CSV-like : timestamp,nb_lignes,nb_stations,date_min,date_max,sha256_short
    """
    log_path = settings.snapshot_log_path
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # SHA256 partiel du parquet (8 premiers caractères) — identification rapide
    h = hashlib.sha256()
    with output_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha_short = h.hexdigest()[:8]

    line = ",".join([
        datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        str(len(df)),
        str(df["station_id"].nunique()),
        str(df["datetime"].min()),
        str(df["datetime"].max()),
        sha_short,
        f"{output_path.stat().st_size / 1e6:.1f}MB",
    ])

    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# timestamp,rows,stations,date_min,date_max,sha256_short,size\n")
        f.write(line + "\n")

    logger.info("Snapshot log mis à jour", extra={"log_path": str(log_path), "sha": sha_short})


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────────────────────
def load_from_hf(write_to_disk: bool = True) -> pd.DataFrame:
    """Télécharge tout le dataset HF, concatène, écrit un parquet brut.

    Args:
        write_to_disk: si True (défaut), écrit le résultat dans
            ``settings.raw_snapshot_path`` et appende dans le snapshot log.

    Returns:
        DataFrame complet, dédupliqué, trié.

    Raises:
        FileNotFoundError: aucun fichier raw trouvé sur HF.
        HfHubHTTPError, RepositoryNotFoundError: après retries épuisés.
    """
    settings.ensure_directories()

    logger.info(
        "Démarrage load_from_hf",
        extra={
            "repo": settings.hf_repo,
            "force_download": settings.hf_force_download,
        },
    )

    # 1) Lister les fichiers raw disponibles sur HF
    files = _list_raw_files()

    # 2) Télécharger chacun dans un cache local
    cache_dir = settings.raw_data_dir / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    local_paths = [_download_one(f, cache_dir) for f in files]

    # 3) Lire, concaténer, dédupliquer, trier
    df = _read_and_concat(local_paths)

    logger.info(
        "Concaténation terminée",
        extra={
            "rows": len(df),
            "stations": df["station_id"].nunique(),
            "date_min": str(df["datetime"].min()),
            "date_max": str(df["datetime"].max()),
        },
    )

    # 4) Écriture parquet (compression zstd)
    if write_to_disk:
        out = settings.raw_snapshot_path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        logger.info(
            "Parquet écrit",
            extra={
                "path": str(out),
                "size_mb": round(out.stat().st_size / 1e6, 1),
            },
        )
        _append_snapshot_log(df, out)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """Point d'entrée CLI. Retourne le code de sortie pour le shell."""
    try:
        load_from_hf(write_to_disk=True)
        return 0
    except FileNotFoundError as e:
        logger.error("Aucun fichier raw sur HF", extra={"error": str(e)})
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception("Échec load_from_hf", extra={"error_type": type(e).__name__})
        return 1


if __name__ == "__main__":
    sys.exit(main())
