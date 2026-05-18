"""
dag_velib_pipeline.py — Orchestration Airflow du pipeline MLOps Vélib'.

DAG  : velib_pipeline
Cron : 0 4 * * *  (tous les jours à 04h00 UTC — heure creuse)
Exec : LocalExecutor (le scheduler lance les tâches directement)

Flux :
    check_hf ──► check_mlflow ──► dvc_repro ──► reload_model ──► smoke_test

Notes d'architecture :
- Le scheduler monte /var/run/docker.sock et le répertoire du projet (HOST_PROJECT_ROOT)
  pour pouvoir lancer 'docker compose run --rm --no-deps ml_training dvc repro'.
- dvc_repro délègue l'intégralité du pipeline DVC (5 stages) au conteneur ml_training,
  qui dispose de toutes les dépendances Python et de l'accès à MLflow et aux volumes data.
- reload_model et smoke_test ciblent api:8000 directement (trafic interne réseau mlops-net).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator

HOST_ROOT = os.getenv("HOST_PROJECT_ROOT", "")
API_INTERNAL = "http://api:8000"
MLFLOW_INTERNAL = "http://mlflow-server:5000"

default_args = {
    "owner": "velib-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry": False,
}

with DAG(
    dag_id="velib_pipeline",
    description="Pipeline MLOps Vélib' : DVC repro complet + rechargement modèle API",
    schedule="0 4 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    default_args=default_args,
    tags=["velib", "mlops", "dvc"],
) as dag:

    check_hf = BashOperator(
        task_id="check_hf_connectivity",
        bash_command=(
            "curl -sf --max-time 15 https://huggingface.co > /dev/null "
            "&& echo 'HuggingFace joignable'"
        ),
        retries=3,
        retry_delay=timedelta(minutes=2),
    )

    check_mlflow = BashOperator(
        task_id="check_mlflow_health",
        bash_command=(
            f"curl -sf --max-time 10 {MLFLOW_INTERNAL}/health > /dev/null "
            "&& echo 'MLflow OK'"
        ),
    )

    dvc_repro = BashOperator(
        task_id="dvc_repro",
        bash_command=(
            f"cd {HOST_ROOT} && "
            "docker compose run --rm --no-deps ml_training dvc repro"
        ),
        execution_timeout=timedelta(hours=3),
        retries=0,
    )

    reload_model = BashOperator(
        task_id="reload_model",
        bash_command=(
            f"RESPONSE=$(curl -sf -X POST {API_INTERNAL}/model/reload) "
            "&& echo \"$RESPONSE\" "
            "&& echo \"$RESPONSE\" | jq -e '.alias == \"staging\"' > /dev/null"
        ),
    )

    smoke_test = BashOperator(
        task_id="smoke_test",
        bash_command=(
            f"curl -sf {API_INTERNAL}/health | grep -q '\"status\":\\s*\"ok\"' "
            "&& echo 'API status: ok'"
        ),
    )

    check_hf >> check_mlflow >> dvc_repro >> reload_model >> smoke_test
