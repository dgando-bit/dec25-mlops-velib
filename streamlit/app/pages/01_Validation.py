import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
from utils import api_client
from utils.stations import STATIONS, build_features

st.set_page_config(page_title="Validation · Vélib' MLOps", page_icon="🏥", layout="wide")

HOST_ROOT = os.getenv("HOST_PROJECT_ROOT", "")

st.markdown("## 🏥 Validation des services")
st.caption("Tests fonctionnels en temps réel · Tests unitaires pytest par service")
st.divider()


# ── Helpers ───────────────────────────────────────────────────────────────────
def status_badge(code: int) -> str:
    if code in (200, 201, 204):
        return "🟢 OK"
    if code == 0:
        return "🔴 Hors ligne"
    if code == 408:
        return "🟡 Timeout"
    return f"🟡 HTTP {code}"


def show_result(code: int, body, label: str = "", description: str = ""):
    st.markdown(f"**{label}** — {status_badge(code)}")
    if description:
        st.caption(description)
    if isinstance(body, dict):
        st.json(body, expanded=False)
    elif isinstance(body, str) and body:
        with st.expander("Réponse brute", expanded=False):
            st.text(body[:1000])


def run_pytest_in_docker(target: str) -> tuple[int, str]:
    if not HOST_ROOT:
        return -1, "HOST_PROJECT_ROOT non défini — impossible de lancer docker compose."
    cmd_map = {
        "api": ["docker", "compose", "run", "--rm", "--no-deps",
                "api", "pytest", "api/tests/", "--tb=short", "-q", "--color=no"],
        "ml":  ["docker", "compose", "run", "--rm", "--no-deps",
                "ml_training", "pytest", "ml/tests/", "shared/tests/",
                "--tb=short", "-q", "--color=no"],
    }
    cmd = cmd_map.get(target)
    if not cmd:
        return -1, "cible inconnue"
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=180, cwd=HOST_ROOT
        )
        output = result.stdout + (("\n--- stderr ---\n" + result.stderr) if result.stderr.strip() else "")
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return -1, "Timeout (180 s)"
    except FileNotFoundError:
        return -1, "Commande `docker` introuvable dans ce conteneur."
    except Exception as e:
        return -1, str(e)


# ── Tabs ─────────────────────────────────────────────────────────────────────
tab_api, tab_mlflow, tab_prom, tab_graf, tab_airflow, tab_nginx, tab_pytest = st.tabs([
    "🔌 API FastAPI",
    "📊 MLflow",
    "📈 Prometheus",
    "📉 Grafana",
    "⚙️ Airflow",
    "🔀 Nginx",
    "🧪 Tests pytest",
])

# ─── API FastAPI ──────────────────────────────────────────────────────────────
with tab_api:
    st.markdown("### API FastAPI — tests fonctionnels")
    st.markdown(
        "L'API FastAPI est exposée via Nginx sur `:8080`. "
        "Ces tests appellent `api:8000` directement depuis le réseau Docker interne."
    )
    st.divider()

    if st.button("▶ Lancer les tests API", key="btn_api"):
        with st.spinner("Appels en cours…"):

            code, body = api_client.api_health()
            show_result(code, body, "GET /health",
                        "Vérifie que l'API est opérationnelle et que le modèle XGBoost est chargé en mémoire. "
                        "Retourne `ok` ou `degraded` si le modèle est absent.")

            st.divider()
            code, body = api_client.api_model_info()
            show_result(code, body, "GET /model/info",
                        "Retourne les métadonnées du modèle actif : nom, alias MLflow (`staging`), "
                        "version, run_id, framework et nombre de features (24).")

            st.divider()
            sample = build_features(
                station=STATIONS[0],
                hour=8, dow=0, month=5,
                temperature=15.0, temp_anomalie=0.5,
                weather_severity=0, is_frozen=0, is_stormy=0,
                lag_60min=50.0, lag_240min=48.0,
                is_holiday=0, is_vacation=0,
            )
            code, body = api_client.api_predict(sample)
            show_result(code, body, "POST /predict — Gare du Nord, lundi 8h",
                        "Prédiction unitaire : 24 features pré-calculées → résidu prédit → taux reconstruit "
                        "(résidu + station_trend_avg). Retourne aussi le niveau d'alerte : green / yellow / red.")

            st.divider()
            sample2 = build_features(
                station=STATIONS[4],  # Pigalle
                hour=18, dow=4, month=5,
                temperature=17.0, temp_anomalie=1.0,
                weather_severity=0, is_frozen=0, is_stormy=0,
                lag_60min=35.0, lag_240min=40.0,
                is_holiday=0, is_vacation=0,
            )
            code, body = api_client.api_predict_batch([sample, sample2])
            show_result(code, body, "POST /predict/batch — 2 stations (Gare du Nord + Pigalle, vendredi 18h)",
                        "Prédiction batch : jusqu'à 2 000 stations en une seule requête. "
                        "Retourne la liste des prédictions dans le même ordre que les inputs.")

            st.divider()
            code, body = api_client.api_metrics()
            st.markdown(f"**GET /metrics** — {status_badge(code)}")
            st.caption(
                "Endpoint Prometheus : expose les métriques de l'API en format text/plain — "
                "nombre de requêtes par endpoint, latence (histogramme), taux d'erreur, statut du modèle."
            )
            if code == 200:
                lines = [l for l in body.split("\n") if l and not l.startswith("#")]
                with st.expander(f"{len(lines)} métriques exposées", expanded=False):
                    st.text("\n".join(lines[:30]))

# ─── MLflow ───────────────────────────────────────────────────────────────────
with tab_mlflow:
    st.markdown("### MLflow — tests fonctionnels")
    st.markdown(
        "MLflow est accessible via Nginx sur `localhost:5000`. "
        "Les tests internes ciblent `mlflow-server:5000` via le réseau Docker."
    )
    st.divider()

    if st.button("▶ Lancer les tests MLflow", key="btn_mlflow"):
        with st.spinner("Connexion à MLflow…"):

            code, body = api_client.mlflow_health()
            show_result(code, body, "GET /api/2.0/mlflow/experiments/list",
                        "Vérifie que le serveur MLflow est joignable et que la base PostgreSQL est accessible. "
                        "Liste les expériences enregistrées (chaque `dvc repro` crée un nouveau run).")

            st.divider()
            code, body = api_client.mlflow_models()
            if code == 200 and isinstance(body, dict):
                models = body.get("registered_models", [])
                st.markdown(f"**GET /api/2.0/mlflow/registered-models/list** — {status_badge(code)} — {len(models)} modèle(s)")
                st.caption(
                    "Vérifie que le modèle `velib_fill_rate_predictor` est enregistré dans le Registry "
                    "et que l'alias `staging` pointe vers la dernière version entraînée."
                )
                for m in models:
                    aliases = [a["alias"] for a in m.get("aliases", [])]
                    st.markdown(f"  - `{m['name']}` · aliases : **{aliases or '—'}**")
            else:
                show_result(code, body, "GET registered-models",
                            "Vérification du Registry MLflow.")

# ─── Prometheus ──────────────────────────────────────────────────────────────
with tab_prom:
    st.markdown("### Prometheus — tests fonctionnels")
    st.markdown(
        "Prometheus scrape `api:8000/metrics` toutes les 15 s "
        "et stocke les métriques pendant 15 jours. UI sur `localhost:9090`."
    )
    st.divider()

    if st.button("▶ Lancer les tests Prometheus", key="btn_prom"):
        with st.spinner("Interrogation Prometheus…"):

            code, body = api_client.prometheus_health()
            st.markdown(f"**GET /-/healthy** — {status_badge(code)}")
            st.caption("Endpoint de liveness de Prometheus. Retourne `Prometheus Server is Healthy.` si opérationnel.")

            st.divider()
            code, body = api_client.prometheus_targets()
            if code == 200 and isinstance(body, dict):
                targets = body.get("data", {}).get("activeTargets", [])
                st.markdown(f"**GET /api/v1/targets** — {status_badge(code)} — {len(targets)} cible(s)")
                st.caption(
                    "Liste les cibles de scrape configurées dans `prometheus.yml`. "
                    "Chaque cible doit être `up` pour que les métriques soient collectées."
                )
                for t in targets:
                    health = "🟢" if t.get("health") == "up" else "🔴"
                    last = t.get("lastScrape", "—")[:19]
                    st.markdown(
                        f"  {health} `{t.get('labels', {}).get('job', '?')}` "
                        f"— {t.get('scrapeUrl', '—')} — dernier scrape : {last}"
                    )
            else:
                show_result(code, body, "GET /api/v1/targets", "Liste des cibles de scrape.")

# ─── Grafana ─────────────────────────────────────────────────────────────────
with tab_graf:
    st.markdown("### Grafana — tests fonctionnels")
    st.markdown(
        "Le dashboard est provisionné automatiquement au démarrage depuis "
        "`deployments/grafana/provisioning/` (datasource + panels). UI sur `localhost:3000`."
    )
    st.divider()

    if st.button("▶ Lancer les tests Grafana", key="btn_graf"):
        with st.spinner("Connexion à Grafana…"):

            code, body = api_client.grafana_health()
            show_result(code, body, "GET /api/health",
                        "Vérifie que Grafana est opérationnel et que sa base de données interne (SQLite) "
                        "est accessible. Retourne la version et l'état de la base.")
            if code == 200 and isinstance(body, dict):
                st.success(
                    f"Grafana {body.get('version', '?')} — "
                    f"base de données : {body.get('database', '?')}"
                )

# ─── Airflow ─────────────────────────────────────────────────────────────────
with tab_airflow:
    st.markdown("### Airflow — tests fonctionnels")
    st.markdown(
        "Airflow orchestre le pipeline quotidien à 04h00 UTC. "
        "DAG `velib_pipeline` : `check_hf → check_mlflow → dvc_repro → reload_model → smoke_test`. "
        "UI sur `localhost:8090`."
    )
    st.divider()

    if st.button("▶ Lancer les tests Airflow", key="btn_airflow"):
        with st.spinner("Connexion à Airflow…"):

            code, body = api_client.airflow_health()
            show_result(code, body, "GET /health",
                        "Vérifie l'état du scheduler (LocalExecutor) et de la metabase PostgreSQL. "
                        "Les deux doivent être `healthy` pour que le DAG puisse s'exécuter.")
            if code == 200 and isinstance(body, dict):
                meta = body.get("metadatabase", {})
                sched = body.get("scheduler", {})
                c1, c2 = st.columns(2)
                c1.metric("Metabase PostgreSQL", meta.get("status", "?"))
                c2.metric("Scheduler", sched.get("status", "?"))

# ─── Nginx ────────────────────────────────────────────────────────────────────
with tab_nginx:
    st.markdown("### Nginx — pages d'erreur personnalisées")
    st.markdown(
        "Nginx est le point d'entrée unique de toute la stack. "
        "Il gère le rate limiting et sert des pages d'erreur brandées Vélib' MLOps "
        "pour les codes 404, 429 et 50x."
    )
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Test 404 — Route introuvable")
        st.caption(
            "Appel vers une route inexistante sur l'API (`/this-route-does-not-exist`). "
            "Nginx intercepte le 404 retourné par FastAPI et sert la page personnalisée "
            "`deployments/nginx/errors/404.html`."
        )
        if st.button("▶ Tester 404", key="btn_404"):
            code, body = api_client.nginx_404()
            if code == 404:
                st.success(f"✅ HTTP {code} — page 404 personnalisée reçue")
                with st.expander("Aperçu HTML"):
                    st.code(body[:500], language="html")
            else:
                st.warning(f"Code reçu : {code}")

    with col2:
        st.markdown("#### Test 429 — Rate limit")
        st.caption(
            "Envoi de 25 requêtes en rafale vers `/health` via Nginx. "
            "Le rate limit est configuré à 10 req/s avec un burst de 20 — "
            "les requêtes excédentaires reçoivent HTTP 429 avec la page `429.html`."
        )
        if st.button("▶ Tester 429 (25 requêtes)", key="btn_429"):
            with st.spinner("Envoi de 25 requêtes…"):
                n429, total = api_client.nginx_rate_limit(25)
            if n429 > 0:
                st.success(f"✅ {n429}/{total} requêtes → HTTP 429 (rate limit actif)")
            else:
                st.info(f"0/{total} requêtes throttlées — le burst de 20 a tout absorbé. Réessayez rapidement.")

    st.divider()
    st.markdown("#### Récapitulatif des pages d'erreur")
    st.markdown("""
| Code | Fichier | Déclencheur |
|------|---------|-------------|
| **404** | `deployments/nginx/errors/404.html` | Route inexistante — interceptée via `proxy_intercept_errors on` |
| **429** | `deployments/nginx/errors/429.html` | Rate limiting dépassé (`limit_req_status 429`) |
| **50x** | `deployments/nginx/errors/50x.html` | API indisponible ou erreur interne upstream |
""")

# ─── Tests pytest ─────────────────────────────────────────────────────────────
with tab_pytest:
    st.markdown("### Tests unitaires pytest")
    st.markdown("""
**94 tests** répartis sur 6 fichiers — exécutables sans la stack complète (pas de MLflow, pas de PostgreSQL) :

| Périmètre | Fichier | Tests |
|-----------|---------|-------|
| Schémas Pydantic | `api/tests/test_schemas.py` | 13 |
| Endpoints FastAPI | `api/tests/test_endpoints.py` | 14 |
| Nettoyage données | `ml/tests/test_data_cleaning.py` | 18 |
| Feature engineering | `ml/tests/test_build_features.py` | 16 |
| Inférence & alertes | `ml/tests/test_predict_model.py` | 10 |
| Configuration | `shared/tests/test_config.py` | 12 |
""")
    st.caption(
        "Stratégie : les tests API injectent un vrai XGBRegressor entraîné sur données synthétiques "
        "(fixture session-scoped). Les tests ML valident le nettoyage, le feature engineering et la logique "
        "d'alerte sans dépendance réseau."
    )
    st.divider()

    col_api, col_ml = st.columns(2)

    with col_api:
        st.markdown("#### Tests API (27 tests)")
        st.caption(
            "Valide les schémas Pydantic (validation stricte des 25 champs) et les 6 endpoints FastAPI "
            "via un TestClient avec modèle XGBoost injecté — sans démarrer la vraie stack."
        )
        st.code("make test-unit-api", language="bash")
        if st.button("▶ Lancer (docker compose run)", key="btn_pytest_api"):
            if not HOST_ROOT:
                st.error("HOST_PROJECT_ROOT non défini dans l'environnement.")
            else:
                with st.spinner("pytest api/tests/ en cours… (~20 s)"):
                    rc, output = run_pytest_in_docker("api")
                if rc == 0:
                    st.success(f"✅ Tous les tests passent (code {rc})")
                else:
                    st.error(f"❌ Échec (code {rc})")
                st.code(output, language="text")

    with col_ml:
        st.markdown("#### Tests ML + shared (56 tests)")
        st.caption(
            "Valide les fonctions de nettoyage, le feature engineering (encodages cycliques, lags, flags), "
            "la reconstruction du taux (`résidu + tendance`) et les seuils d'alerte green/yellow/red."
        )
        st.code("make test-unit-ml", language="bash")
        if st.button("▶ Lancer (docker compose run)", key="btn_pytest_ml"):
            if not HOST_ROOT:
                st.error("HOST_PROJECT_ROOT non défini dans l'environnement.")
            else:
                with st.spinner("pytest ml/tests/ shared/tests/ en cours… (~30 s)"):
                    rc, output = run_pytest_in_docker("ml")
                if rc == 0:
                    st.success(f"✅ Tous les tests passent (code {rc})")
                else:
                    st.error(f"❌ Échec (code {rc})")
                st.code(output, language="text")
