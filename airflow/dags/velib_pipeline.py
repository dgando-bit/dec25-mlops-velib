from __future__ import annotations

import hashlib
import logging
import os
import sys
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.email import EmailOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
SNAPSHOT_LOG = Path("/app/data/raw/.snapshots.log")
HASH_STATE_FILE = Path("/app/data/raw/.last_dag_hash")

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 0,                        # pas de retry — on alerte immédiatement
    "retry_delay": timedelta(minutes=5),
    "email": [os.getenv("AIRFLOW_ALERT_EMAIL", "mlops@yopmail.com")],
    "email_on_failure": False,
    "email_on_retry": False,
}


# ─────────────────────────────────────────────────────────────────────────────
# CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────
def on_failure_callback(context: dict) -> None:
    """Loggue l'échec — l'email est géré par email_on_failure=True."""
    task_id = context["task_instance"].task_id
    dag_id = context["dag"].dag_id
    log.error("Échec du pipeline — dag=%s task=%s", dag_id, task_id)


# ─────────────────────────────────────────────────────────────────────────────
# FONCTIONS DES TÂCHES
# ─────────────────────────────────────────────────────────────────────────────
def _sense_new_hf_file() -> bool:
    """Vérifie si un nouveau fichier est disponible sur HuggingFace.

    Stratégie : liste les fichiers du repo HF et compare avec l'état
    sauvegardé lors du dernier run. Retourne True si un nouveau fichier
    est détecté (ce qui débloque le sensor).
    """
    try:
        from huggingface_hub import list_repo_files
        from shared.config import get_settings

        settings = get_settings()
        files = sorted(
            f for f in list_repo_files(
                repo_id=settings.hf_repo,
                repo_type="dataset",
                token=settings.hf_token_value(),
            )
            if f.startswith(settings.hf_file_prefix) and f.endswith(".csv")
        )

        state_file = Path("/app/data/raw/.last_hf_files")
        current = "\n".join(files)

        if not state_file.exists():
            state_file.write_text(current)
            log.info("Premier run — %d fichiers HF détectés", len(files))
            return True

        previous = state_file.read_text().strip()
        if current.strip() != previous:
            state_file.write_text(current)
            log.info("Nouveau fichier HF détecté — lancement du pipeline")
            return True

        log.info("Aucun nouveau fichier HF — sensor en attente")
        return False

    except Exception as e:
        log.error("Erreur sensor HF : %s", e)
        return False


def _load_from_hf() -> None:
    """Télécharge les données depuis HuggingFace."""
    result = subprocess.run(
        [sys.executable, "-m", "ml.src.data.load_from_hf"],
        cwd="/app",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"load_from_hf échoué :\n{result.stderr}")
    log.info(result.stdout)


def _check_data_changed(**context) -> str:
    """Compare le hash SHA256 du dernier snapshot avec le run précédent.

    Retourne l'id de la tâche suivante :
        - 'make_dataset'  si les données ont changé → pipeline complet
        - 'skip_training' si les données n'ont pas changé → arrêt propre
    """
    if not SNAPSHOT_LOG.exists():
        log.warning("Snapshot log absent — on continue le pipeline")
        return "make_dataset"

    lines = [l for l in SNAPSHOT_LOG.read_text().splitlines() if not l.startswith("#")]
    if not lines:
        return "make_dataset"

    # Dernière ligne du log : timestamp,rows,stations,date_min,date_max,sha256_short,size
    last_sha = lines[-1].split(",")[5]

    if not HASH_STATE_FILE.exists():
        HASH_STATE_FILE.write_text(last_sha)
        log.info("Premier hash enregistré : %s — pipeline complet", last_sha)
        return "make_dataset"

    previous_sha = HASH_STATE_FILE.read_text().strip()

    if last_sha == previous_sha:
        log.info("Données inchangées (sha=%s) — skip entraînement", last_sha)
        return "skip_training"

    HASH_STATE_FILE.write_text(last_sha)
    log.info("Données changées : %s → %s — pipeline complet", previous_sha, last_sha)
    return "make_dataset"


def _make_dataset() -> None:
    """Nettoie les données brutes → interim."""
    result = subprocess.run(
        [sys.executable, "-m", "ml.src.data.make_dataset"],
        cwd="/app",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"make_dataset échoué :\n{result.stderr}")
    log.info(result.stdout)


def _build_features() -> None:
    """Feature engineering → train/test parquet."""
    result = subprocess.run(
        [sys.executable, "-m", "ml.src.features.build_features"],
        cwd="/app",
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"build_features échoué :\n{result.stderr}")
    log.info(result.stdout)


def _train_model() -> None:
    """Entraîne le modèle et le logue dans MLflow."""
    result = subprocess.run(
        [sys.executable, "-m", "ml.src.models.train_model"],
        cwd="/app",
        capture_output=True,
        text=True,
        env={**os.environ, "MLFLOW_TRACKING_URI": os.getenv("MLFLOW_TRACKING_URI", "http://mlflow-server:5000")},
    )
    if result.returncode != 0:
        raise RuntimeError(f"train_model échoué :\n{result.stderr}")
    log.info(result.stdout)


def _skip_training() -> None:
    """Tâche no-op — données inchangées, entraînement skippé."""
    log.info("Données inchangées — entraînement skippé.")


# ─────────────────────────────────────────────────────────────────────────────
# DAG
# ─────────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="velib_ml_pipeline",
    description="Pipeline ML Vélib — détection HF → load → clean → features → train",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(hours=6),  # poll toutes les 6h
    catchup=False,
    max_active_runs=1,                     # pas de runs parallèles
    tags=["velib", "ml", "production"],
    on_failure_callback=on_failure_callback,
) as dag:

    # 1. Sensor HuggingFace
    sense_new_data = PythonSensor(
        task_id="sense_new_data",
        python_callable=_sense_new_hf_file,
        poke_interval=60 * 30,   # vérifie toutes les 30 min
        timeout=60 * 60 * 5,     # timeout après 5h
        mode="reschedule",       # libère le worker entre les pokes
        soft_fail=False,
    )

    # 2. Téléchargement HuggingFace
    load_from_hf = PythonOperator(
        task_id="load_from_hf",
        python_callable=_load_from_hf,
        on_failure_callback=on_failure_callback,
    )

    # 3. Vérification du hash
    check_data_changed = BranchPythonOperator(
        task_id="check_data_changed",
        python_callable=_check_data_changed,
    )

    # 4a. Nettoyage des données
    make_dataset = PythonOperator(
        task_id="make_dataset",
        python_callable=_make_dataset,
        on_failure_callback=on_failure_callback,
    )

    # 4b. Skip si données inchangées
    skip_training = PythonOperator(
        task_id="skip_training",
        python_callable=_skip_training,
    )

    # 5. Feature engineering
    build_features = PythonOperator(
        task_id="build_features",
        python_callable=_build_features,
        on_failure_callback=on_failure_callback,
    )

    # 6. Entraînement
    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        on_failure_callback=on_failure_callback,
    )

    # 7. Notification succès
    notify_success = EmailOperator(
        task_id="notify_success",
        to=os.getenv("AIRFLOW_ALERT_EMAIL", "mlops@example.com"),
        subject="✅ Pipeline Vélib — Entraînement terminé",
        html_content="""
            <h3>Pipeline Vélib terminé avec succès</h3>
            <p>Le modèle a été réentraîné et enregistré dans MLflow.</p>
            <p><b>DAG :</b> velib_ml_pipeline</p>
            <p><b>Date :</b> {{ ds }}</p>
        """,
        trigger_rule="none_failed_min_one_success",
    )

    # ─── Dépendances ──────────────────────────────────────────────────────────
    #
    #  sense_new_data
    #       ↓
    #  load_from_hf
    #       ↓
    #  check_data_changed
    #       ↓              ↓
    #  make_dataset    skip_training
    #       ↓              ↓
    #  build_features      |
    #       ↓              |
    #  train_model         |
    #       ↓              ↓
    #        notify_success
    #
    var = sense_new_data >> load_from_hf >> check_data_changed
    var_1 = check_data_changed >> make_dataset >> build_features >> train_model >> notify_success
    var_2 = check_data_changed >> skip_training >> notify_success