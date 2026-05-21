"""
ml.src.data.make_dataset — Nettoyage du snapshot brut Vélib'.

Lit le parquet brut produit par ``load_from_hf`` et applique 4 étapes de
nettoyage de qualité de données via DuckDB (traitement en streaming,
empreinte mémoire ~10x inférieure à pandas).

    1. Coercition des booléens (is_renting, is_holiday, is_vacation)
    2. Calcul de total_capacity et du taux de remplissage
    3. Filtres de qualité :
        - is_renting == True
        - capacity > 0 (méta-donnée valide)
        - total_capacity > 0 (relevé exploitable)
        - taux ∈ [0, 100] (garde-fou défensif)
    4. Exclusion des stations sans variance significative (inentraînables)

Pipeline DVC :
    Stage  : make_dataset
    Entrée : data/raw/velib_snapshot_latest.parquet
    Sortie : data/interim/velib_cleaned_latest.parquet

Usage :
    python -m ml.src.data.make_dataset
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from shared.config import settings
from shared.logger import get_logger

logger = get_logger(__name__)

_CAPACITY_STATUS_TOLERANCE = 0.5


# ─────────────────────────────────────────────────────────────────────────────
# NETTOYAGE VIA DUCKDB
# ─────────────────────────────────────────────────────────────────────────────
# def _clean_with_duckdb(raw_path: Path) -> pd.DataFrame:
#     """Applique toutes les étapes de nettoyage via DuckDB en streaming.
#
#     DuckDB lit le parquet sans tout charger en RAM — idéal pour les gros fichiers.
#     """
#     con = duckdb.connect()
#
#     # ── Stats avant nettoyage ─────────────────────────────────────────────
#     rows_in = con.execute(
#         f"SELECT COUNT(*) FROM read_parquet('{raw_path}')"
#     ).fetchone()[0]
#     logger.info("Parquet brut chargé", extra={"rows": rows_in})
#
#     # ── Étape 1 & 2 : booléens + total_capacity + taux ───────────────────
#     # ── Étape 3 : filtres de qualité ──────────────────────────────────────
#     cleaned_query = f"""
#         WITH base AS (
#             SELECT *,
#                 -- Coercition booléens
#                 CASE
#                     WHEN TYPEOF(is_renting) = 'boolean' THEN is_renting
#                     WHEN CAST(is_renting AS VARCHAR) IN ('true', '1', 'True') THEN TRUE
#                     ELSE FALSE
#                 END AS is_renting_bool,
#
#                 -- total_capacity
#                 COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0) + COALESCE(numdocksavailable, 0)
#                     AS total_capacity_calc,
#
#                 -- taux de remplissage
#                 CASE
#                     WHEN (COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0) + COALESCE(numdocksavailable, 0)) > 0
#                     THEN ROUND(
#                         (COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0)) * 100.0
#                         / (COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0) + COALESCE(numdocksavailable, 0)),
#                         2
#                     )
#                     ELSE NULL
#                 END AS taux
#             FROM read_parquet('{raw_path}')
#         ),
#         filtered AS (
#             SELECT *
#             FROM base
#             WHERE
#                 is_renting_bool = TRUE
#                 AND capacity > 0
#                 AND total_capacity_calc > 0
#                 AND taux BETWEEN 0 AND 100
#         ),
#         station_variance AS (
#             SELECT
#                 station_id,
#                 VARIANCE(taux) AS var_taux
#             FROM filtered
#             GROUP BY station_id
#             HAVING VARIANCE(taux) >= {settings.min_variance}
#         )
#         SELECT f.*
#         FROM filtered f
#         INNER JOIN station_variance sv ON f.station_id = sv.station_id
#         ORDER BY f.station_id, f.datetime
#     """
#
#     logger.info("Application des filtres DuckDB...")
#     df = con.execute(cleaned_query).df()
#     con.close()
#
#     return df

def _clean_with_duckdb(raw_path: Path, out_path: Path) -> tuple[int, int]:
    """Applique les filtres et écrit directement en parquet via DuckDB.

    Returns:
        Tuple (rows_in, rows_out)
    """
    con = duckdb.connect()

    rows_in = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{raw_path}')"
    ).fetchone()[0]

    logger.info("Application des filtres DuckDB...", extra={"rows_in": rows_in})

    # Écriture directe parquet — jamais chargé en RAM
    con.execute(f"""
        COPY (
            WITH base AS (
                SELECT *,
                    CASE
                        WHEN TYPEOF(is_renting) = 'boolean' THEN is_renting
                        WHEN CAST(is_renting AS VARCHAR) IN ('true', '1', 'True') THEN TRUE
                        ELSE FALSE
                    END AS is_renting_bool,
                    COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0) + COALESCE(numdocksavailable, 0)
                        AS total_capacity_calc,
                    CASE
                        WHEN (COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0) + COALESCE(numdocksavailable, 0)) > 0
                        THEN ROUND(
                            (COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0)) * 100.0
                            / (COALESCE(bikes_ebike, 0) + COALESCE(bikes_mechanical, 0) + COALESCE(numdocksavailable, 0)),
                            2
                        )
                        ELSE NULL
                    END AS taux
                FROM read_parquet('{raw_path}')
            ),
            filtered AS (
                SELECT * FROM base
                WHERE is_renting_bool = TRUE
                  AND capacity > 0
                  AND total_capacity_calc > 0
                  AND taux BETWEEN 0 AND 100
            ),
            station_variance AS (
                SELECT station_id
                FROM filtered
                GROUP BY station_id
                HAVING VARIANCE(taux) >= {settings.min_variance}
            )
            SELECT f.*
            FROM filtered f
            INNER JOIN station_variance sv ON f.station_id = sv.station_id
            ORDER BY f.station_id, f.datetime
        ) TO '{out_path}' (FORMAT PARQUET, COMPRESSION 'zstd')
    """)

    rows_out = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_path}')"
    ).fetchone()[0]

    con.close()
    return rows_in, rows_out

# ─────────────────────────────────────────────────────────────────────────────
# STATS DE NETTOYAGE
# ─────────────────────────────────────────────────────────────────────────────
def _log_cleaning_stats(df: pd.DataFrame, rows_in: int) -> list[int]:
    """Loggue les stats de nettoyage et retourne les stations exclues."""
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
    return []


# ─────────────────────────────────────────────────────────────────────────────
# CONSIGNATION DU CLEANING
# ─────────────────────────────────────────────────────────────────────────────
def _append_cleaning_log(
    rows_in: int,
    rows_out: int,
    output_path: Path,
) -> None:
    """Append une ligne dans data/interim/.cleaning.log."""
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
        sha_short,
        f"{output_path.stat().st_size / 1e6:.1f}MB",
    ])

    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write("# timestamp,rows_in,rows_out,filter_pct,sha256_short,size\n")
        f.write(line + "\n")

    logger.info(
        "Cleaning log mis à jour",
        extra={"log_path": str(log_path), "sha": sha_short},
    )


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────────────────────
# def make_dataset(write_to_disk: bool = True) -> pd.DataFrame:
#     """Nettoie le snapshot brut via DuckDB et écrit le parquet interim.
#
#     Args:
#         write_to_disk: si True (défaut), écrit dans ``settings.interim_path``.
#
#     Returns:
#         DataFrame nettoyé.
#
#     Raises:
#         FileNotFoundError: si le snapshot brut n'existe pas.
#     """
#     settings.ensure_directories()
#
#     raw_path = settings.raw_snapshot_path
#     if not raw_path.exists():
#         raise FileNotFoundError(
#             f"Snapshot brut absent : {raw_path}. "
#             "Lancer d'abord : python -m ml.src.data.load_from_hf"
#         )
#
#     logger.info("Démarrage make_dataset", extra={"input_path": str(raw_path)})
#
#     # Nombre de lignes avant nettoyage (via DuckDB, pas de chargement RAM)
#     con = duckdb.connect()
#     rows_in = con.execute(
#         f"SELECT COUNT(*) FROM read_parquet('{raw_path}')"
#     ).fetchone()[0]
#     con.close()
#
#     # Nettoyage complet via DuckDB
#     df = _clean_with_duckdb(raw_path)
#     rows_out = len(df)
#
#     _log_cleaning_stats(df, rows_in)
#
#     if write_to_disk:
#         out = settings.interim_path
#         out.parent.mkdir(parents=True, exist_ok=True)
#         df.to_parquet(out, engine="pyarrow", compression="zstd", index=False)
#         logger.info(
#             "Parquet nettoyé écrit",
#             extra={
#                 "path": str(out),
#                 "size_mb": round(out.stat().st_size / 1e6, 1),
#             },
#         )
#         _append_cleaning_log(rows_in, rows_out, out)
#
#     return df

def make_dataset(write_to_disk: bool = True) -> pd.DataFrame:
    settings.ensure_directories()

    raw_path = settings.raw_snapshot_path
    if not raw_path.exists():
        raise FileNotFoundError(
            f"Snapshot brut absent : {raw_path}. "
            "Lancer d'abord : python -m ml.src.data.load_from_hf"
        )

    logger.info("Démarrage make_dataset", extra={"input_path": str(raw_path)})

    out = settings.interim_path
    out.parent.mkdir(parents=True, exist_ok=True)

    # Écriture directe parquet sans passer par pandas
    rows_in, rows_out = _clean_with_duckdb(raw_path, out)

    filter_pct = (1 - rows_out / rows_in) * 100 if rows_in > 0 else 0.0
    logger.info(
        "Nettoyage terminé",
        extra={
            "rows_in": rows_in,
            "rows_out": rows_out,
            "filter_pct": round(filter_pct, 2),
        },
    )
    logger.info(
        "Parquet nettoyé écrit",
        extra={"path": str(out), "size_mb": round(out.stat().st_size / 1e6, 1)},
    )

    _append_cleaning_log(rows_in, rows_out, out)

    # Retourne un DataFrame uniquement si demandé (pour compatibilité)
    if not write_to_disk:
        return duckdb.read_parquet(str(out)).df()

    return pd.DataFrame()  # caller n'a pas besoin du df en mode pipeline

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        make_dataset(write_to_disk=True)
        return 0
    except FileNotFoundError as e:
        logger.error("Snapshot brut absent", extra={"error": str(e)})
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception("Échec make_dataset", extra={"error_type": type(e).__name__})
        return 1


if __name__ == "__main__":
    sys.exit(main())