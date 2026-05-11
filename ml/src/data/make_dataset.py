"""
ml.src.data.make_dataset — Nettoyage du snapshot brut Vélib'.

Lit le parquet brut produit par ``load_from_hf`` et applique 4 étapes de
nettoyage de qualité de données :

    1. Coercition des booléens (is_renting, is_holiday, is_vacation)
    2. Calcul de total_capacity et du taux de remplissage
    3. Filtres de qualité :
        - is_renting == True
        - capacity > 0 (méta-donnée valide)
        - total_capacity > 0 (relevé exploitable)
        - taux ∈ [0, 100] (garde-fou défensif)
    4. Exclusion des stations sans variance significative (inentraînables)

Vérifications additionnelles :
    - assertion de cohérence avec capacity_status produit par le collector
      (warning si écart > tolerance, ne fait pas échouer le script)

Pipeline DVC :
    Stage  : make_dataset
    Entrée : data/raw/velib_snapshot_latest.parquet
    Sortie : data/interim/velib_cleaned_latest.parquet

Usage :
    # En CLI (depuis la racine du repo, avec .venv activé)
    python -m ml.src.data.make_dataset

    # En import depuis un autre module
    from ml.src.data.make_dataset import make_dataset
    df = make_dataset()
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone

import pandas as pd

from shared.config import settings
from shared.logger import get_logger
from shared.utils.data_cleaning import (
    coerce_booleans,
    compute_fill_rate,
    compute_total_capacity,
    filter_active_stations,
    filter_stations_with_variance,
    filter_valid_capacities,
    filter_valid_fill_rate,
)

logger = get_logger(__name__)


# Tolérance pour l'assertion de cohérence avec capacity_status du collector.
# 0.5 point de pourcentage : laisse passer les arrondis flottants éventuels.
_CAPACITY_STATUS_TOLERANCE = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# CONTRÔLE QUALITÉ — assertion de cohérence avec capacity_status
# ─────────────────────────────────────────────────────────────────────────────
def _assert_capacity_status_consistency(
    df: pd.DataFrame,
    recomputed_rate: pd.Series,
) -> None:
    """Vérifie que notre recalcul du taux est cohérent avec capacity_status.

    Si écart > tolerance sur ne serait-ce qu'une ligne, log un warning avec
    le détail. Ne fait jamais échouer le script (warning, pas exception).
    """
    if "capacity_status" not in df.columns:
        return
    existing = df["capacity_status"].astype(float)
    mask = recomputed_rate.notna() & existing.notna()
    diff = (recomputed_rate[mask] - existing[mask]).abs()
    above_tol = diff > _CAPACITY_STATUS_TOLERANCE
    n_diverge = int(above_tol.sum())
    if n_diverge > 0:
        logger.warning(
            "Divergence avec capacity_status du collector",
            extra={
                "rows_diverging": n_diverge,
                "max_diff_pct": float(diff.max()),
                "tolerance_pct": _CAPACITY_STATUS_TOLERANCE,
            },
        )
    else:
        logger.info(
            "Cohérence capacity_status validée",
            extra={"rows_compared": int(mask.sum())},
        )


# ─────────────────────────────────────────────────────────────────────────────
# CONSIGNATION DU CLEANING (log humain, miroir de .snapshots.log)
# ─────────────────────────────────────────────────────────────────────────────
def _append_cleaning_log(
    rows_in: int,
    rows_out: int,
    excluded_stations: list[int],
    output_path,
) -> None:
    """Append une ligne dans data/interim/.cleaning.log.

    Format CSV-like :
        timestamp,rows_in,rows_out,filter_rate_pct,excluded_stations_count,sha256_short,size
    """
    log_path = settings.interim_data_dir / ".cleaning.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha256()
    with output_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha_short = h.hexdigest()[:8]

    filter_rate = (1 - rows_out / rows_in) * 100 if rows_in > 0 else 0.0
    line = ",".join([
        datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        str(rows_in),
        str(rows_out),
        f"{filter_rate:.2f}",
        str(len(excluded_stations)),
        sha_short,
        f"{output_path.stat().st_size / 1e6:.1f}MB",
    ])

    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "# timestamp,rows_in,rows_out,filter_pct,"
                "excluded_stations,sha256_short,size\n"
            )
        f.write(line + "\n")

    logger.info(
        "Cleaning log mis à jour",
        extra={"log_path": str(log_path), "sha": sha_short},
    )


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────────────────────
def make_dataset(write_to_disk: bool = True) -> pd.DataFrame:
    """Lit le snapshot brut, applique les 4 étapes de nettoyage, écrit l'interim.

    Args:
        write_to_disk: si True (défaut), écrit le parquet nettoyé dans
            ``settings.interim_path`` et appende dans ``.cleaning.log``.

    Returns:
        DataFrame nettoyé, prêt pour le feature engineering.

    Raises:
        FileNotFoundError: si ``settings.raw_snapshot_path`` n'existe pas
            (lancer ``load_from_hf`` au préalable).
    """
# #     settings.ensure_directories()

    # ── Lecture parquet brut ──────────────────────────────────────────────
    raw_path = settings.raw_snapshot_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Snapshot brut absent : {raw_path}. "
            "Lancer d'abord : python -m ml.src.data.load_from_hf"
        )

    logger.info(
        "Démarrage make_dataset",
        extra={"input_path": str(raw_path)},
    )
    df = pd.read_parquet(raw_path)
    rows_in = len(df)
    logger.info(
        "Parquet brut chargé",
        extra={"rows": rows_in, "columns": len(df.columns)},
    )

    # ── Étape 1 : coercition des booléens ─────────────────────────────────
    df = coerce_booleans(df, ["is_renting", "is_holiday", "is_vacation"])

    # ── Étape 2 : calcul total_capacity et taux ───────────────────────────
    total_capacity = compute_total_capacity(df)
    taux = compute_fill_rate(df, total_capacity=total_capacity)

    # Contrôle qualité : cohérence avec capacity_status existant
    _assert_capacity_status_consistency(df, taux)

    df = df.assign(total_capacity=total_capacity, taux=taux)

    # ── Étape 3 : filtres de qualité ──────────────────────────────────────
    rows_before_filters = len(df)

    df = filter_active_stations(df)
    rows_after_active = len(df)
    logger.info(
        "Filtre is_renting==True",
        extra={
            "removed": rows_before_filters - rows_after_active,
            "remaining": rows_after_active,
        },
    )

    df = filter_valid_capacities(
        df, min_capacity=1, total_capacity=df["total_capacity"]
    )
    rows_after_capacities = len(df)
    logger.info(
        "Filtre capacity > 0 et total_capacity > 0",
        extra={
            "removed": rows_after_active - rows_after_capacities,
            "remaining": rows_after_capacities,
        },
    )

    df = filter_valid_fill_rate(df, rate_col="taux")
    rows_after_rate = len(df)
    logger.info(
        "Filtre taux ∈ [0, 100] (garde-fou défensif)",
        extra={
            "removed": rows_after_capacities - rows_after_rate,
            "remaining": rows_after_rate,
        },
    )

    # ── Étape 4 : exclusion des stations sans variance ────────────────────
    df, excluded_stations = filter_stations_with_variance(
        df,
        min_variance=settings.min_variance,
        rate_col="taux",
        station_col="station_id",
    )
    rows_after_variance = len(df)
    logger.info(
        "Filtre stations sans variance significative",
        extra={
            "min_variance": settings.min_variance,
            "excluded_stations_count": len(excluded_stations),
            "rows_removed": rows_after_rate - rows_after_variance,
            "remaining": rows_after_variance,
        },
    )

    # ── Bilan global ──────────────────────────────────────────────────────
    rows_out = len(df)
    filter_pct = (1 - rows_out / rows_in) * 100 if rows_in > 0 else 0.0
    logger.info(
        "Nettoyage terminé",
        extra={
            "rows_in": rows_in,
            "rows_out": rows_out,
            "rows_filtered_total": rows_in - rows_out,
            "filter_pct": round(filter_pct, 2),
            "stations_remaining": int(df["station_id"].nunique()),
        },
    )

    # ── Écriture parquet nettoyé ──────────────────────────────────────────
    if write_to_disk:
        out = settings.interim_path
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
        logger.info(
            "Parquet nettoyé écrit",
            extra={
                "path": str(out),
                "size_mb": round(out.stat().st_size / 1e6, 1),
            },
        )
        _append_cleaning_log(rows_in, rows_out, excluded_stations, out)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """Point d'entrée CLI. Retourne le code de sortie pour le shell."""
    try:
        make_dataset(write_to_disk=True)
        return 0
    except FileNotFoundError as e:
        logger.error("Snapshot brut absent", extra={"error": str(e)})
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception(
            "Échec make_dataset", extra={"error_type": type(e).__name__}
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
