"""
ml.src.features.build_features — Pipeline de feature engineering Vélib'.

Lit le parquet nettoyé produit par ``make_dataset`` et applique 9 étapes
de feature engineering pour produire les datasets train/test prêts à
l'entraînement.

Ordre des étapes (préserve la prévention des fuites) :

    1. Chargement du parquet nettoyé
    2. Features temporelles      (hour, dow, month, sin/cos, flags)
    3. Profil de taille           (capacity_group)
    4. Features météo             (severity, is_frozen, is_stormy)
    5. Encodage booléens cal.     (is_holiday, is_vacation → int)
    6. Lags temporels             (lag_60min, lag_240min via merge_asof)
    7. Split temporel             (quantile 1 - test_size)
    8. Post-split features        (morning_evening_ratio, temp_anomalie)
    9. station_trend_avg          (sur train, fallback 2 niveaux)
   10. Cible résiduelle           (residual_target, lag_res_240min)
   11. Export parquet train/test/stations_geo

Pipeline DVC :
    Stage  : build_features
    Entrée : data/interim/velib_cleaned_latest.parquet
    Sorties :
        - data/processed/train_preprocessed.parquet
        - data/processed/test_preprocessed.parquet
        - data/processed/stations_geo.parquet

Usage :
    # En CLI
    python -m ml.src.features.build_features

    # En import depuis un autre module
    from ml.src.features.build_features import build_features
    train, test, stations_geo = build_features()
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone

import pandas as pd

from shared.config import settings
from shared.logger import get_logger
from ml.src.features._helpers import (
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
# CONSIGNATION DES FEATURES (log humain miroir de .cleaning.log)
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
    """Append une ligne dans data/processed/.features.log."""
    log_path = settings.processed_data_dir / ".features.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Hash du fichier train (clé d'identification du dataset)
    h = hashlib.sha256()
    with train_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha_short = h.hexdigest()[:8]

    line = ",".join([
        datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        str(rows_in),
        str(train_rows),
        str(test_rows),
        str(n_features),
        station_trend_mode,
        sha_short,
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

    logger.info(
        "Features log mis à jour",
        extra={"log_path": str(log_path), "sha": sha_short, "mode": station_trend_mode},
    )


def _save_metadata(station_trend_mode: str, n_features: int) -> None:
    """Sauve les métadonnées du build pour que train_model.py puisse les lire.

    Notamment, le mode station_trend_avg utilisé sera tagué dans le run MLflow
    pour comparer plus tard les runs de modes différents.
    """
    import json
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
    """Lit le parquet nettoyé, exécute les 9 étapes de FE, écrit train/test.

    Args:
        write_to_disk: si True (défaut), écrit les 3 parquets dans
            ``settings.processed_data_dir`` et appende dans ``.features.log``.

    Returns:
        Tuple (train, test, stations_geo) — DataFrames prêts à l'entraînement.

    Raises:
        FileNotFoundError: si le parquet nettoyé n'existe pas
            (lancer ``make_dataset`` au préalable).
    """
    settings.ensure_directories()

    # ── Chargement du parquet nettoyé ─────────────────────────────────────
    cleaned_path = settings.interim_path
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Parquet nettoyé absent : {cleaned_path}. "
            "Lancer d'abord : python -m ml.src.data.make_dataset"
        )

    logger.info("Démarrage build_features", extra={"input_path": str(cleaned_path)})
    df = pd.read_parquet(cleaned_path)
    rows_in = len(df)
    logger.info(
        "Parquet nettoyé chargé",
        extra={
            "rows": rows_in,
            "stations": int(df["station_id"].nunique()),
            "columns": len(df.columns),
        },
    )

    # ── Étape 2 : features temporelles ────────────────────────────────────
    df = add_temporal_features(df)
    logger.info("Features temporelles ajoutées", extra={"new_cols": 10})

    # ── Étape 3 : profil de taille ────────────────────────────────────────
    df = add_capacity_group(df)
    logger.info("capacity_group ajoutée")

    # ── Étape 4 : features météo ──────────────────────────────────────────
    df = add_weather_features(df)
    logger.info("Features météo ajoutées",
                extra={"new_cols": "weather_severity,is_frozen,is_stormy"})

    # ── Étape 5 : encodage booléens calendaires ───────────────────────────
    df = encode_calendar_booleans(df)
    logger.info("Booléens calendaires encodés en int8")

    # ── Étape 6 : lags temporels (merge_asof) ─────────────────────────────
    df = add_temporal_lags(df, lag_minutes=(60, 240))

    # ── Étape 7 : split temporel ──────────────────────────────────────────
    train, test = temporal_split(df, test_size=settings.test_size)

    # ── Étape 8 : post-split features (anti-fuite) ────────────────────────
    train, test = add_post_split_features(train, test)

    # ── Étape 9 : station_trend_avg (anti-fuite, cascade adaptative) ──────
    train, test, station_trend_mode = add_station_trend_avg(train, test, min_obs=3)

    # ── Étape 10 : cible résiduelle + lag résiduel ────────────────────────
    train, test = add_residual_target(train, test)

    # ── Table stations_geo (pour Streamlit / API) ─────────────────────────
    stations_geo = extract_stations_geo(df)
    logger.info(
        "Table stations_geo extraite",
        extra={"stations": len(stations_geo)},
    )

    # ── Bilan global ──────────────────────────────────────────────────────
    n_features = len(train.columns)
    logger.info(
        "Feature engineering terminé",
        extra={
            "rows_in": rows_in,
            "train_rows": len(train),
            "test_rows": len(test),
            "stations_in_geo": len(stations_geo),
            "n_features": n_features,
            "target_raw_sigma": round(float(train[TARGET_RAW].std()), 2),
            "target_residual_sigma": round(float(train[TARGET_RESIDUAL].std()), 2),
        },
    )

    # ── Écriture ──────────────────────────────────────────────────────────
    if write_to_disk:
        out_dir = settings.processed_data_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        train_path = settings.train_path
        test_path = settings.test_path
        stations_geo_path = out_dir / "stations_geo.parquet"

        # Compression zstd cohérente avec le reste du pipeline
        train.to_parquet(train_path, engine="pyarrow", compression="zstd", index=False)
        test.to_parquet(test_path, engine="pyarrow", compression="zstd", index=False)
        stations_geo.to_parquet(
            stations_geo_path, engine="pyarrow", compression="zstd", index=False
        )

        logger.info(
            "Parquets écrits",
            extra={
                "train_path": str(train_path),
                "train_size_mb": round(train_path.stat().st_size / 1e6, 1),
                "test_path": str(test_path),
                "test_size_mb": round(test_path.stat().st_size / 1e6, 1),
                "stations_geo_path": str(stations_geo_path),
            },
        )

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
    """Point d'entrée CLI. Retourne le code de sortie pour le shell."""
    try:
        build_features(write_to_disk=True)
        return 0
    except FileNotFoundError as e:
        logger.error("Parquet nettoyé absent", extra={"error": str(e)})
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception(
            "Échec build_features", extra={"error_type": type(e).__name__}
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
