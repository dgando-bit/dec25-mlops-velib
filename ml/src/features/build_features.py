"""
ml.src.features.build_features — Pipeline de feature engineering Vélib'.

Optimisé mémoire : libération explicite entre chaque étape + types compacts.
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from shared.config import settings
from shared.logger import get_logger
from ml.src.utils._helpers import (
    add_capacity_group,
    add_post_split_features,
    add_residual_target,
    add_station_trend_avg,
    add_temporal_features,
    add_temporal_lags,
    add_weather_features,
    encode_calendar_booleans,
    extract_stations_geo,
    temporal_split,
    TARGET_RAW,
    TARGET_RESIDUAL,
)

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMISATION MÉMOIRE
# ─────────────────────────────────────────────────────────────────────────────
def _optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Réduit l'empreinte mémoire en choisissant les types les plus compacts.

    - float64 → float32 pour les features continues (précision suffisante)
    - int64   → int8/int16 selon la plage de valeurs
    - object  → category si peu de valeurs distinctes
    """
    out = df.copy()

    for col in out.select_dtypes(include="float64").columns:
        # Garde float64 uniquement pour datetime et les cibles
        if col not in ("taux", TARGET_RAW, TARGET_RESIDUAL, "station_trend_avg",
                       "lag_60min", "lag_240min", "lag_res_240min"):
            out[col] = out[col].astype("float32")

    for col in out.select_dtypes(include="int64").columns:
        col_min, col_max = out[col].min(), out[col].max()
        if col_min >= -128 and col_max <= 127:
            out[col] = out[col].astype("int8")
        elif col_min >= -32768 and col_max <= 32767:
            out[col] = out[col].astype("int16")
        else:
            out[col] = out[col].astype("int32")

    return out


def _log_memory(df: pd.DataFrame, label: str) -> None:
    """Loggue l'empreinte mémoire du DataFrame."""
    mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info(f"Mémoire [{label}]", extra={"mb": round(mb, 1), "rows": len(df)})


# ─────────────────────────────────────────────────────────────────────────────
# CONSIGNATION
# ─────────────────────────────────────────────────────────────────────────────
def _append_features_log(
    rows_in: int,
    train_rows: int,
    test_rows: int,
    n_features: int,
    train_path,
    test_path,
    station_trend_mode: str,
) -> None:
    log_path = settings.processed_data_dir / ".features.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    h = hashlib.sha256()
    with train_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha_short = h.hexdigest()[:8]

    line = ",".join([
        datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        str(rows_in), str(train_rows), str(test_rows),
        str(n_features), station_trend_mode, sha_short,
        f"{train_path.stat().st_size / 1e6:.1f}MB",
        f"{test_path.stat().st_size / 1e6:.1f}MB",
    ])

    is_new = not log_path.exists()
    with log_path.open("a", encoding="utf-8") as f:
        if is_new:
            f.write(
                "# timestamp,rows_in,train_rows,test_rows,n_features,"
                "station_trend_mode,train_sha256_short,train_size,test_size\n"
            )
        f.write(line + "\n")

    logger.info("Features log mis à jour", extra={"sha": sha_short})


def _save_metadata(station_trend_mode: str, n_features: int) -> None:
    metadata = {
        "station_trend_mode": station_trend_mode,
        "n_features": n_features,
        "build_timestamp": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
    }
    out_path = settings.processed_data_dir / "build_metadata.json"
    out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    logger.info("Metadata build sauvegardé", extra={"path": str(out_path)})


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE
# ─────────────────────────────────────────────────────────────────────────────
def build_features(
    write_to_disk: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Feature engineering optimisé mémoire avec libération explicite entre étapes."""
    settings.ensure_directories()

    cleaned_path = settings.interim_path
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Parquet nettoyé absent : {cleaned_path}. "
            "Lancer d'abord : python -m ml.src.data.make_dataset"
        )

    logger.info("Démarrage build_features", extra={"input_path": str(cleaned_path)})

    # ── Étape 1 : chargement avec colonnes sélectionnées ──────────────────
    # On ne charge que les colonnes nécessaires au FE
    needed_cols = [
        "station_id", "name", "lat", "lon", "capacity",
        "datetime", "taux",
        "apparent_temperature", "weather_code",
        "is_holiday", "is_vacation",
        # colonnes DuckDB ajoutées par make_dataset
        "is_renting_bool", "total_capacity_calc",
    ]
    # Lire toutes les colonnes disponibles (certaines peuvent manquer)
    df = pd.read_parquet(cleaned_path)
    available = [c for c in needed_cols if c in df.columns]
    df = df[available].copy()

    rows_in = len(df)
    logger.info("Parquet nettoyé chargé", extra={
        "rows": rows_in,
        "stations": int(df["station_id"].nunique()),
        "columns": len(df.columns),
    })

    # Normaliser datetime immédiatement
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.as_unit("ns")

    # Extraire stations_geo avant toute transformation (colonnes minimales)
    stations_geo = extract_stations_geo(df)
    logger.info("Table stations_geo extraite", extra={"stations": len(stations_geo)})

    # ── Étape 2 : features temporelles ────────────────────────────────────
    df = add_temporal_features(df)
    df = _optimize_dtypes(df)
    gc.collect()
    _log_memory(df, "après temporal_features")
    logger.info("Features temporelles ajoutées")

    # ── Étape 3 : profil de taille ────────────────────────────────────────
    df = add_capacity_group(df)
    gc.collect()
    logger.info("capacity_group ajoutée")

    # ── Étape 4 : features météo ──────────────────────────────────────────
    df = add_weather_features(df)
    df = _optimize_dtypes(df)
    gc.collect()
    logger.info("Features météo ajoutées")

    # ── Étape 5 : encodage booléens ───────────────────────────────────────
    df = encode_calendar_booleans(df)
    gc.collect()
    logger.info("Booléens calendaires encodés")

    # ── Étape 6 : lags temporels ──────────────────────────────────────────
    # C'est l'étape la plus lourde — on libère tout ce qu'on peut avant
    df = _optimize_dtypes(df)
    gc.collect()
    _log_memory(df, "avant lags")

    df = add_temporal_lags(df, lag_minutes=(60, 240))
    gc.collect()
    _log_memory(df, "après lags")

    # ── Étape 7 : split temporel ──────────────────────────────────────────
    train, test = temporal_split(df, test_size=settings.test_size)
    del df  # libère immédiatement le DataFrame complet
    gc.collect()
    _log_memory(train, "train après split")
    _log_memory(test, "test après split")

    # ── Étape 8 : post-split features (anti-fuite) ────────────────────────
    train, test = add_post_split_features(train, test)
    gc.collect()

    # ── Étape 9 : station_trend_avg (anti-fuite, cascade adaptative) ──────
    train, test, station_trend_mode = add_station_trend_avg(train, test, min_obs=3)
    gc.collect()

    # ── Étape 10 : cible résiduelle ───────────────────────────────────────
    train, test = add_residual_target(train, test)
    gc.collect()

    # Optimisation finale des types
    train = _optimize_dtypes(train)
    test = _optimize_dtypes(test)

    n_features = len(train.columns)
    logger.info("Feature engineering terminé", extra={
        "rows_in": rows_in,
        "train_rows": len(train),
        "test_rows": len(test),
        "n_features": n_features,
        "target_raw_sigma": round(float(train[TARGET_RAW].std()), 2),
        "target_residual_sigma": round(float(train[TARGET_RESIDUAL].std()), 2),
        "station_trend_mode": station_trend_mode,
    })

    # ── Écriture ──────────────────────────────────────────────────────────
    if write_to_disk:
        out_dir = settings.processed_data_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = settings.train_path
        test_path = settings.test_path
        stations_geo_path = out_dir / "stations_geo.parquet"

        train.to_parquet(train_path, engine="pyarrow", compression="zstd", index=False)
        test.to_parquet(test_path, engine="pyarrow", compression="zstd", index=False)
        stations_geo.to_parquet(
            stations_geo_path, engine="pyarrow", compression="zstd", index=False
        )

        logger.info("Parquets écrits", extra={
            "train_size_mb": round(train_path.stat().st_size / 1e6, 1),
            "test_size_mb": round(test_path.stat().st_size / 1e6, 1),
        })

        _append_features_log(
            rows_in=rows_in,
            train_rows=len(train),
            test_rows=len(test),
            n_features=n_features,
            train_path=train_path,
            test_path=test_path,
            station_trend_mode=station_trend_mode,
        )
        _save_metadata(station_trend_mode=station_trend_mode, n_features=n_features)

    return train, test, stations_geo


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    try:
        build_features(write_to_disk=True)
        return 0
    except FileNotFoundError as e:
        logger.error("Parquet nettoyé absent", extra={"error": str(e)})
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception("Échec build_features", extra={"error_type": type(e).__name__})
        return 1


if __name__ == "__main__":
    sys.exit(main())