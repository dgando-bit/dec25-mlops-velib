"""
airflow/dags/velib_pipeline.py — Pipeline ML Vélib orchestré par Airflow.
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.sensors.python import PythonSensor
from airflow.utils.task_group import TaskGroup
from airflow.utils.trigger_rule import TriggerRule

log = logging.getLogger(__name__)

SNAPSHOT_LOG = Path("/app/data/raw/.snapshots.log")
HASH_STATE_FILE = Path("/app/data/raw/.last_dag_hash")
ALERT_EMAIL = os.getenv("AIRFLOW_ALERT_EMAIL", "mlops@example.com")

DEFAULT_ARGS = {
    "owner": "mlops",
    "retries": 0,
    "retry_delay": timedelta(minutes=5),
    "email": [ALERT_EMAIL],
    "email_on_failure": False,
    "email_on_retry": False,
}


def on_failure_callback(context: dict) -> None:
    task_id = context["task_instance"].task_id
    dag_id = context["dag"].dag_id
    log.error("Échec du pipeline — dag=%s task=%s", dag_id, task_id)


def _run(module: str, env_extra: dict | None = None) -> None:
    """Lance un module Python en subprocess et lève une exception si échec."""
    env = {**os.environ, **(env_extra or {})}
    result = subprocess.run(
        [sys.executable, "-m", module],
        cwd="/app",
        capture_output=True,
        text=True,
        env=env,
    )
    if result.stdout:
        log.info(result.stdout)
    if result.returncode != 0:
        raise RuntimeError(
            f"{module} échoué (code {result.returncode}):\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def _sense_new_hf_file() -> bool:
    try:
        from airflow.models import Variable
        if Variable.get("dev_mode", default_var="false") == "true":
            log.info("Mode dev — sensor bypassed")
            return True

        from huggingface_hub import list_repo_files
        from shared.config import settings

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
    _run("ml.src.data.load_from_hf")


def _check_data_changed(**context) -> str:
    """Retourne 'ml_pipeline.make_dataset' ou 'skip_training'."""
    from airflow.models import Variable
    if Variable.get("force_training", default_var="false") == "true":
        log.info("Mode force training — _check_data_changed bypassed")
        return "ml_pipeline.make_dataset"

    if not SNAPSHOT_LOG.exists():
        log.warning("Snapshot log absent — on continue le pipeline")
        return "ml_pipeline.make_dataset"

    lines = [l for l in SNAPSHOT_LOG.read_text().splitlines() if not l.startswith("#")]
    if not lines:
        return "ml_pipeline.make_dataset"

    last_sha = lines[-1].split(",")[5]

    if not HASH_STATE_FILE.exists():
        HASH_STATE_FILE.write_text(last_sha)
        log.info("Premier hash enregistré : %s — pipeline complet", last_sha)
        return "ml_pipeline.make_dataset"

    previous_sha = HASH_STATE_FILE.read_text().strip()

    if last_sha == previous_sha:
        log.info("Données inchangées (sha=%s) — skip", last_sha)
        return "skip_training"

    HASH_STATE_FILE.write_text(last_sha)
    log.info("Données changées : %s → %s", previous_sha, last_sha)
    return "ml_pipeline.make_dataset"


def _make_dataset() -> None:
    _run("ml.src.data.make_dataset")


def _build_features() -> None:
    _run("ml.src.features.build_features")


def _train_model() -> None:
    _run(
        "ml.src.models.train_model",
        env_extra={"MLFLOW_TRACKING_URI": os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow-server:5000"
        )},
    )


def _skip_training() -> None:
    log.info("Données inchangées — entraînement skippé.")


def _notify(**context) -> None:
    """Email adaptatif : succès / skip / échec."""
    from airflow.utils.email import send_email

    dag_run = context["dag_run"]
    run_id = dag_run.run_id
    ds = context["ds"]

    task_instances = dag_run.get_task_instances()
    failed = [t for t in task_instances if t.state == "failed"]
    skipped = any(
        t.task_id == "skip_training" and t.state == "success"
        for t in task_instances
    )

    if failed:
        tasks_html = "".join(
            f"<li><code>{t.task_id}</code> — {t.state}</li>" for t in failed
        )
        subject = f"❌ Pipeline Vélib — Échec ({ds})"
        body = f"""
            <h2 style="color:#C62828">❌ ÉCHEC — Pipeline Vélib</h2>
            <p>Le pipeline a échoué sur les tâches suivantes :</p>
            <ul>{tasks_html}</ul>
            <p><b>Run ID :</b> {run_id}</p>
            <p><b>Date :</b> {ds}</p>
            <p>Consulter les logs dans Airflow pour le détail.</p>
        """

    elif skipped:
        subject = f"⏭️ Pipeline Vélib — Skip ({ds})"
        body = f"""
            <h2 style="color:#F57C00">⏭️ SKIPPÉ — Pipeline Vélib</h2>
            <p>Aucune nouvelle donnée détectée sur HuggingFace.</p>
            <p>Le réentraînement a été skippé — le modèle en production reste inchangé.</p>
            <p><b>Run ID :</b> {run_id}</p>
            <p><b>Date :</b> {ds}</p>
        """

    else:
        subject = f"✅ Pipeline Vélib — Succès ({ds})"
        body = f"""
            <h2 style="color:#2E7D32">✅ SUCCÈS — Pipeline Vélib</h2>
            <p>Le modèle a été réentraîné et enregistré dans MLflow
            avec l'alias <b>staging</b>.</p>
            <p><b>Run ID :</b> {run_id}</p>
            <p><b>Date :</b> {ds}</p>
            <p>Consulter MLflow pour valider et promouvoir en production.</p>
        """

    send_email(to=ALERT_EMAIL, subject=subject, html_content=body)
    log.info("Email envoyé — subject=%s", subject)


# ─────────────────────────────────────────────────────────────────────────────
# DAG
# ─────────────────────────────────────────────────────────────────────────────
with DAG(
    dag_id="velib_ml_pipeline",
    description="Pipeline ML Vélib — détection HF → load → clean → features → train",
    default_args=DEFAULT_ARGS,
    start_date=datetime(2026, 1, 1),
    schedule=timedelta(hours=6),
    catchup=False,
    max_active_runs=1,
    tags=["velib", "ml", "production"],
    on_failure_callback=on_failure_callback,
) as dag:

    sense_new_data = PythonSensor(
        task_id="sense_new_data",
        python_callable=_sense_new_hf_file,
        poke_interval=60 * 30,
        timeout=60 * 60 * 5,
        mode="reschedule",
        soft_fail=False,
    )

    load_from_hf = PythonOperator(
        task_id="load_from_hf",
        python_callable=_load_from_hf,
        execution_timeout=timedelta(hours=1),
        on_failure_callback=on_failure_callback,
    )

    check_data_changed = BranchPythonOperator(
        task_id="check_data_changed",
        python_callable=_check_data_changed,
    )

    # ── Groupe ML ────────────────────────────────────────────────────────
    with TaskGroup(
        group_id="ml_pipeline",
        tooltip="Nettoyage → Feature Engineering → Entraînement",
    ) as ml_group:

        make_dataset = PythonOperator(
            task_id="make_dataset",
            python_callable=_make_dataset,
            execution_timeout=timedelta(minutes=30),
            on_failure_callback=on_failure_callback,
        )

        build_features = PythonOperator(
            task_id="build_features",
            python_callable=_build_features,
            execution_timeout=timedelta(hours=1),
            on_failure_callback=on_failure_callback,
        )

        train_model = PythonOperator(
            task_id="train_model",
            python_callable=_train_model,
            execution_timeout=timedelta(hours=2),
            on_failure_callback=on_failure_callback,
        )

        make_dataset >> build_features >> train_model

    skip_training = PythonOperator(
        task_id="skip_training",
        python_callable=_skip_training,
    )

    # ALL_DONE : s'exécute quoi qu'il arrive (succès, skip, échec)
    notify = PythonOperator(
        task_id="notify",
        python_callable=_notify,
        trigger_rule=TriggerRule.ALL_DONE,
    )

    # ── Dépendances ──────────────────────────────────────────────────────
    #
    #  sense_new_data
    #       ↓
    #  load_from_hf
    #       ↓
    #  check_data_changed
    #       ↓                    ↓
    #  [ml_pipeline]        skip_training
    #    make_dataset             ↓
    #       ↓                    |
    #    build_features          |
    #       ↓                    |
    #    train_model             |
    #       ↓                    ↓
    #              notify
    #
    sense_new_data >> load_from_hf >> check_data_changed
    check_data_changed >> ml_group >> notify
    check_data_changed >> skip_training >> notify