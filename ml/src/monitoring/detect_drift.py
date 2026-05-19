"""
detect_drift.py — Stage 6 DVC: détection de data drift via Evidently.

Inputs  : data/processed/train_preprocessed.parquet  (référence — distribution entraînement)
          data/processed/test_preprocessed.parquet   (courant   — distribution production proxy)
Outputs : data/outputs/drift/drift_report.html       (rapport Evidently interactif)
          data/outputs/drift/drift_metrics.json      (résumé : drift_share, n_drifted, alert)

Le drift est calculé sur les 24 features du modèle (FEATURES_FINAL).
Seuil d'alerte : > 30% des features ont drifté de façon statistiquement significative.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report

from ml.src.models._helpers import FEATURES_FINAL
from shared.logger import get_logger

logger = get_logger(__name__)

TRAIN_PATH = Path("data/processed/train_preprocessed.parquet")
TEST_PATH = Path("data/processed/test_preprocessed.parquet")
OUTPUT_DIR = Path("data/outputs/drift")
REPORT_PATH = OUTPUT_DIR / "drift_report.html"
METRICS_PATH = OUTPUT_DIR / "drift_metrics.json"

# Seuil : au-delà de 30% de features driftées, on lève une alerte dans les logs.
# Ce seuil ne bloque pas la promotion du modèle (informatif à ce stade).
# Seuil d'alerte drift : au-delà de 30% de features driftées, on lève un warning dans les logs.
# Ce seuil ne bloque pas la promotion du modèle (informatif à ce stade).
DRIFT_SHARE_THRESHOLD = 0.30

# NOTE MÉTHODOLOGIQUE — limitation connue de cette implémentation :
# "current" = test_preprocessed.parquet, issu du même snapshot HuggingFace que le train,
# splité par quantile temporel (80/20). Le drift mesuré reflète donc la dérive saisonnière
# intrinsèque au split (notamment sur 'month', 'apparent_temperature', 'weather_severity')
# et non une dérive de production réelle.
# Pour un monitoring temps réel, il faudrait comparer un nouveau snapshot HF
# (données fraîches post-entraînement) à une fenêtre glissante de référence.


def _extract_dataset_drift(result: dict) -> dict:
    """Extrait le résumé DatasetDriftMetric du rapport Evidently."""
    for metric in result.get("metrics", []):
        if "DatasetDrift" in metric.get("metric", ""):
            return metric["result"]
    raise KeyError("DatasetDriftMetric introuvable dans le rapport Evidently")


def main() -> None:
    logger.info("Chargement reference=%s current=%s", TRAIN_PATH, TEST_PATH)
    reference = pd.read_parquet(TRAIN_PATH, columns=FEATURES_FINAL)
    current = pd.read_parquet(TEST_PATH, columns=FEATURES_FINAL)
    logger.info(
        "reference: %d lignes | current: %d lignes | features: %d",
        len(reference), len(current), len(FEATURES_FINAL),
    )

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference, current_data=current)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report.save_html(str(REPORT_PATH))
    logger.info("Rapport HTML écrit: %s", REPORT_PATH)

    drift_result = _extract_dataset_drift(report.as_dict())
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "n_features": drift_result["number_of_columns"],
        "n_drifted": drift_result["number_of_drifted_columns"],
        "drift_share": round(drift_result["share_of_drifted_columns"], 4),
        "dataset_drift": drift_result["dataset_drift"],
        "drift_alert": drift_result["share_of_drifted_columns"] > DRIFT_SHARE_THRESHOLD,
    }

    METRICS_PATH.write_text(json.dumps(summary, indent=2))

    if summary["drift_alert"]:
        logger.warning(
            "DRIFT ALERT — %.0f%% des features ont drifté (%d/%d) — seuil=%.0f%%",
            summary["drift_share"] * 100,
            summary["n_drifted"],
            summary["n_features"],
            DRIFT_SHARE_THRESHOLD * 100,
        )
    else:
        logger.info(
            "Pas de drift significatif — %.0f%% des features (%d/%d) — seuil=%.0f%%",
            summary["drift_share"] * 100,
            summary["n_drifted"],
            summary["n_features"],
            DRIFT_SHARE_THRESHOLD * 100,
        )


if __name__ == "__main__":
    main()
