"""
dag_velib_pipeline.py — Vélib' MLOps orchestration DAG.

DAG  : velib_pipeline
Cron : 0 4 * * *  (every day at 04:00 UTC — off-peak)
Exec : LocalExecutor (the scheduler runs tasks directly)

Flow :
    preflight_checks ──► dvc_status_check ──► dvc_repro ──►
        parse_metrics ──► gate_metrics ──► branch_on_gate ──►
            [promote_model | skip_promotion] ──► smoke_test

Architecture notes:
- The scheduler mounts /var/run/docker.sock and the host project dir
  (HOST_PROJECT_ROOT) so it can run 'docker compose run --rm --no-deps
  ml_training ...' via Docker-out-of-Docker.
- dvc_repro delegates the full DVC pipeline (5 stages) to the ml_training
  container, which already has all Python deps and access to MLflow/volumes.
- reload_model and smoke_test target api:8000 directly (internal mlops-net).
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from airflow import DAG
from airflow.decorators import task, task_group
from airflow.exceptions import AirflowException
from airflow.models import TaskInstance
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.trigger_rule import TriggerRule

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

API_INTERNAL = "http://api:8000"
MLFLOW_INTERNAL = "http://mlflow-server:5000"

# Quality gate thresholds — tuned against the baseline run ea250421
# (taux_r2=0.833, taux_mae=8.32). A 10-pt drop on R² or +50% MAE blocks
# promotion so the previous staging model keeps serving.
GATE_MIN_R2 = 0.75
GATE_MAX_MAE = 12.0
GATE_MAX_MAPE_PCT = 50.0

HOST_ROOT = os.getenv("HOST_PROJECT_ROOT", "").strip()
if not HOST_ROOT:
    # Fail loud at DAG parse time — avoids cryptic "cd: missing operand"
    # errors at task runtime hours later.
    raise AirflowException(
        "HOST_PROJECT_ROOT env var is not set on the airflow-scheduler "
        "service. Add it to .env and recreate the container."
    )

METRICS_PATH = Path(HOST_ROOT) / "data" / "outputs" / "metrics.json"

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Failure callback
# ---------------------------------------------------------------------------


def _on_failure(context: dict[str, Any]) -> None:
    """Structured JSON log on task failure — easy to grep / forward to Slack."""
    ti: TaskInstance = context["task_instance"]
    payload = {
        "event": "task_failed",
        "dag_id": ti.dag_id,
        "task_id": ti.task_id,
        "run_id": context.get("run_id"),
        "try_number": ti.try_number,
        "log_url": ti.log_url,
        "exception": str(context.get("exception")),
    }
    logger.error(json.dumps(payload))


# ---------------------------------------------------------------------------
# DAG definition
# ---------------------------------------------------------------------------

default_args = {
    "owner": "velib-mlops",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
    "email_on_failure": False,
    "email_on_retry": False,
    "on_failure_callback": _on_failure,
}

DAG_DOC = """
### velib_pipeline

Daily MLOps pipeline for the **Vélib' fill-rate predictor**.

**Stages:**
1. `preflight_checks` — HuggingFace + MLflow + API are reachable.
2. `dvc_status_check` — short-circuit if nothing changed (saves 5-30 min).
3. `dvc_repro` — full DVC pipeline (5 stages) inside `ml_training`.
4. `parse_metrics` — reads `data/outputs/metrics.json`, pushes XCom dict.
5. `gate_metrics` — validates R², MAE, MAPE against configured thresholds.
6. `branch_on_gate` — routes to `promote_model` or `skip_promotion`.
7. `smoke_test` — `GET /health` confirms the API is `ok` in both branches.

A failed gate does **not** crash the DAG: the previous staging model keeps
serving, the run completes with a clear skip branch, and metrics are logged.
"""

with DAG(
    dag_id="velib_pipeline",
    description="MLOps pipeline: DVC repro + quality-gated model promotion",
    schedule="0 4 * * *",
    start_date=datetime(2025, 1, 1),
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(hours=7),
    default_args=default_args,
    tags=["velib", "mlops", "dvc", "mlflow"],
    doc_md=DAG_DOC,
) as dag:

    # ----- 1. Preflight checks (parallel, grouped) -----------------------

    @task_group(group_id="preflight_checks")
    def preflight_checks() -> None:
        BashOperator(
            task_id="check_hf_connectivity",
            bash_command=(
                "curl -sf --max-time 15 https://huggingface.co > /dev/null "
                "&& echo 'HuggingFace reachable'"
            ),
            retries=3,
            retry_delay=timedelta(minutes=2),
            doc_md="Verifies HuggingFace Hub is reachable before pulling snapshots.",
        )
        BashOperator(
            task_id="check_mlflow_health",
            bash_command=(
                f"curl -sf --max-time 10 {MLFLOW_INTERNAL}/health > /dev/null "
                "&& echo 'MLflow OK'"
            ),
            retries=3,
            retry_delay=timedelta(minutes=1),
            doc_md="Verifies MLflow server is up before the training stage.",
        )
        BashOperator(
            task_id="check_api_alive",
            # API may be 'degraded' at this point — we just need it to respond.
            bash_command=(
                f"curl -sf --max-time 5 {API_INTERNAL}/health "
                "| jq -e 'has(\"status\")' > /dev/null "
                "&& echo 'API alive'"
            ),
            retries=2,
            retry_delay=timedelta(minutes=1),
            doc_md="Verifies the FastAPI service responds before smoke test.",
        )

    # ----- 2. Short-circuit if DVC has nothing to reproduce --------------

    # ignore_downstream_trigger_rules=False : le ShortCircuit respecte les
    # trigger rules des tâches downstream (NONE_FAILED_MIN_ONE_SUCCESS sur
    # smoke_test). Les tâches skippées propagent leur état normalement plutôt
    # que d'être forcées skipped indépendamment de leur trigger rule.
    @task.short_circuit(task_id="dvc_status_check", ignore_downstream_trigger_rules=False)
    def dvc_status_check() -> bool:
        """
        Returns True if DVC detects changes that require a pipeline rerun.
        Returns False to short-circuit all downstream tasks when the pipeline
        is already up-to-date — avoids unnecessary XGBoost training runs.
        """
        result = subprocess.run(
            [
                "docker", "compose", "run", "--rm", "--no-deps",
                "ml_training", "dvc", "status",
            ],
            cwd=HOST_ROOT,
            capture_output=True,
            text=True,
            timeout=300,
        )
        stdout = (result.stdout or "").strip()
        logger.info("dvc status:\n%s", stdout)
        return "up to date" not in stdout.lower()

    # ----- 3. DVC pipeline reproduction ----------------------------------

    dvc_repro = BashOperator(
        task_id="dvc_repro",
        bash_command=(
            f"cd {HOST_ROOT} && "
            "docker compose run --rm --no-deps ml_training dvc repro"
        ),
        execution_timeout=timedelta(hours=3),
        retries=1,
        retry_delay=timedelta(minutes=5),
        retry_exponential_backoff=True,
        doc_md="Runs the 5-stage DVC pipeline inside the `ml_training` container.",
    )

    # ----- 4. Quality gate on metrics ------------------------------------

    @task(task_id="parse_metrics")
    def parse_metrics() -> dict[str, Any]:
        """Reads metrics.json produced by dvc_repro and pushes typed XCom."""
        if not METRICS_PATH.exists():
            raise AirflowException(f"metrics.json not found at {METRICS_PATH}")
        try:
            data = json.loads(METRICS_PATH.read_text())
            parsed = {
                "run_id": data["run_id"],
                "r2": float(data["taux_r2"]),
                "mae": float(data["taux_mae"]),
                "mape": float(data["taux_mape_pct"]),
                "timestamp": data["timestamp"],
            }
        except (json.JSONDecodeError, KeyError, ValueError) as exc:
            raise AirflowException(
                f"metrics.json is malformed or missing expected keys: {exc}"
            ) from exc
        logger.info("Parsed metrics: %s", json.dumps(parsed))
        return parsed

    @task(task_id="gate_metrics")
    def gate_metrics(metrics: dict[str, Any]) -> bool:
        """Validates new model metrics against quality thresholds."""
        r2, mae, mape = metrics["r2"], metrics["mae"], metrics["mape"]
        checks = {
            "r2_ok": r2 >= GATE_MIN_R2,
            "mae_ok": mae <= GATE_MAX_MAE,
            "mape_ok": mape <= GATE_MAX_MAPE_PCT,
        }
        passed = all(checks.values())
        logger.info(
            "Gate %s — run_id=%s r2=%.3f (>= %.2f), mae=%.2f (<= %.2f), "
            "mape=%.1f%% (<= %.1f%%) — checks=%s",
            "PASSED" if passed else "FAILED",
            metrics["run_id"],
            r2, GATE_MIN_R2, mae, GATE_MAX_MAE, mape, GATE_MAX_MAPE_PCT, checks,
        )
        return passed

    @task.branch(task_id="branch_on_gate")
    def branch_on_gate(gate_passed: bool) -> str:
        return "deployment.promote_model" if gate_passed else "deployment.skip_promotion"

    # ----- 5. Deployment: promote OR skip --------------------------------

    @task_group(group_id="deployment")
    def deployment() -> None:
        promote = BashOperator(
            task_id="promote_model",
            bash_command=(
                f"RESPONSE=$(curl -sf -X POST {API_INTERNAL}/model/reload) "
                "&& echo \"$RESPONSE\" "
                "&& echo \"$RESPONSE\" | jq -e '.alias == \"staging\"' > /dev/null"
            ),
            doc_md="Reloads the API's in-memory model from the MLflow `staging` alias.",
        )

        skip = EmptyOperator(
            task_id="skip_promotion",
            doc_md=(
                "Gate failed — previous staging model keeps serving. "
                "Check metrics.json and the latest MLflow run for root cause."
            ),
        )

        # smoke_test runs in BOTH branches: the API must stay healthy whether
        # we promoted or skipped.
        smoke = BashOperator(
            task_id="smoke_test",
            bash_command=(
                f"curl -sf {API_INTERNAL}/health "
                "| jq -e '.status == \"ok\"' > /dev/null "
                "&& echo 'API status: ok'"
            ),
            trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS,
            retries=2,
            retry_delay=timedelta(minutes=1),
            doc_md="Final liveness check — runs after promote OR skip.",
        )

        [promote, skip] >> smoke

    # ----- Wiring --------------------------------------------------------

    pre = preflight_checks()
    dvc_check = dvc_status_check()
    metrics_xcom = parse_metrics()
    gate_xcom = gate_metrics(metrics_xcom)
    branch = branch_on_gate(gate_xcom)
    deploy = deployment()

    pre >> dvc_check >> dvc_repro >> metrics_xcom >> gate_xcom >> branch >> deploy
