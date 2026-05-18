import os
import requests
from typing import Any

API_BASE = os.getenv("API_URL", "http://api:8000")
MLFLOW_BASE = os.getenv("MLFLOW_URL", "http://mlflow-server:5000")
PROMETHEUS_BASE = os.getenv("PROMETHEUS_URL", "http://prometheus:9090")
GRAFANA_BASE = os.getenv("GRAFANA_URL", "http://grafana:3000")
AIRFLOW_BASE = os.getenv("AIRFLOW_URL", "http://airflow-webserver:8080")
NGINX_BASE = os.getenv("NGINX_URL", "http://nginx:80")

TIMEOUT = 5


def _get(url: str, timeout: int = TIMEOUT) -> tuple[int, Any]:
    try:
        r = requests.get(url, timeout=timeout)
        ct = r.headers.get("content-type", "")
        body = r.json() if "json" in ct else r.text
        return r.status_code, body
    except requests.exceptions.ConnectionError:
        return 0, "connexion refusée"
    except requests.exceptions.Timeout:
        return 408, "timeout"
    except Exception as e:
        return -1, str(e)


def _post(url: str, payload: dict, timeout: int = TIMEOUT) -> tuple[int, Any]:
    try:
        r = requests.post(url, json=payload, timeout=timeout)
        ct = r.headers.get("content-type", "")
        body = r.json() if "json" in ct else r.text
        return r.status_code, body
    except requests.exceptions.ConnectionError:
        return 0, "connexion refusée"
    except requests.exceptions.Timeout:
        return 408, "timeout"
    except Exception as e:
        return -1, str(e)


# ── API FastAPI ──────────────────────────────────────────────────────────────
def api_health() -> tuple[int, Any]:
    return _get(f"{API_BASE}/health")

def api_model_info() -> tuple[int, Any]:
    return _get(f"{API_BASE}/model/info")

def api_predict(features: dict) -> tuple[int, Any]:
    return _post(f"{API_BASE}/predict", features)

def api_predict_batch(items: list[dict]) -> tuple[int, Any]:
    return _post(f"{API_BASE}/predict/batch", {"items": items})

def api_reload() -> tuple[int, Any]:
    return _post(f"{API_BASE}/model/reload", {})

def api_metrics() -> tuple[int, str]:
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=TIMEOUT)
        return r.status_code, r.text
    except Exception as e:
        return -1, str(e)


# ── MLflow ───────────────────────────────────────────────────────────────────
def mlflow_health() -> tuple[int, Any]:
    # MLflow n'a pas de /health officiel — on utilise l'API expériments
    return _get(f"{MLFLOW_BASE}/api/2.0/mlflow/experiments/list?max_results=1")

def mlflow_models() -> tuple[int, Any]:
    return _get(f"{MLFLOW_BASE}/api/2.0/mlflow/registered-models/list?max_results=5")


# ── Prometheus ───────────────────────────────────────────────────────────────
def prometheus_health() -> tuple[int, Any]:
    return _get(f"{PROMETHEUS_BASE}/-/healthy")

def prometheus_targets() -> tuple[int, Any]:
    return _get(f"{PROMETHEUS_BASE}/api/v1/targets")


# ── Grafana ──────────────────────────────────────────────────────────────────
def grafana_health() -> tuple[int, Any]:
    return _get(f"{GRAFANA_BASE}/api/health")


# ── Airflow ──────────────────────────────────────────────────────────────────
def airflow_health() -> tuple[int, Any]:
    return _get(f"{AIRFLOW_BASE}/health")


# ── Nginx ────────────────────────────────────────────────────────────────────
def nginx_404() -> tuple[int, str]:
    try:
        r = requests.get(f"{NGINX_BASE}/this-route-does-not-exist-404", timeout=TIMEOUT)
        return r.status_code, r.text[:200]
    except Exception as e:
        return -1, str(e)


def nginx_rate_limit(n: int = 25) -> tuple[int, int]:
    """Envoie n requêtes rapides, retourne (nb_429, nb_total)."""
    count_429 = 0
    for _ in range(n):
        try:
            r = requests.get(f"{NGINX_BASE}/health", timeout=2)
            if r.status_code == 429:
                count_429 += 1
        except Exception:
            pass
    return count_429, n
