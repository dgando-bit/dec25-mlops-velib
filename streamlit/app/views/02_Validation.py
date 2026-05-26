import sys
import os
import subprocess

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import streamlit.components.v1 as components
from pathlib import Path
from utils import api_client
from utils.stations import STATIONS, build_features

st.set_page_config(page_title="Validation · Vélib' MLOps", page_icon="🏥", layout="wide")

HOST_ROOT = os.getenv("HOST_PROJECT_ROOT", "")
BASE_DIR = Path(__file__).parent.parent  # app/


# ── SVG helper ───────────────────────────────────────────────────────────────

@st.cache_data
def _load_svg(rel_path: str) -> str | None:
    full = BASE_DIR / rel_path
    return full.read_text(encoding="utf-8") if full.exists() else None


def _render_svg_zoomable(svg_path: str, height: int = 480, key: str = "diag-val"):
    svg = _load_svg(svg_path)
    if svg is None:
        st.warning(f"SVG introuvable : `{svg_path}`")
        return
    html = f"""
    <div id="{key}" style="width:100%;height:{height}px;border:1px solid #e6e6e6;
         border-radius:14px;overflow:hidden;background:#ffffff;">
      {svg}
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/svg-pan-zoom/3.6.1/svg-pan-zoom.min.js"></script>
    <script>
      (function() {{
        var el = document.querySelector('#{key} svg');
        if (!el) return;
        el.setAttribute('width',  '100%');
        el.setAttribute('height', '100%');
        el.style.maxWidth = 'none';
        svgPanZoom(el, {{
          controlIconsEnabled: true, fit: true, center: true,
          minZoom: 0.3, maxZoom: 12, mouseWheelZoomEnabled: true
        }});
      }})();
    </script>
    """
    components.html(html, height=height + 10, scrolling=False)


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


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("## 🏥 Validation des services")
st.caption("Tests fonctionnels en temps réel · Tests unitaires pytest par service")
st.divider()

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

    st.divider()
    st.markdown("#### Architecture de la stack d'observabilité")
    st.caption(
        "3 jobs de scrape (velib-api :8000/metrics, node-exporter :9100, self-scrape :9090), "
        "TSDB 15 jours, datasource auto-provisionnée, "
        "2 règles d'alerte (CPU > 90 % / 5 min, RAM > 80 % / 2 min)."
    )
    _render_svg_zoomable("assets/diagrams/05_observability.svg", height=480, key="diag-obs")

# ─── Grafana ─────────────────────────────────────────────────────────────────
with tab_graf:
    st.markdown("### Grafana — tests fonctionnels")
    st.markdown(
        "Le dashboard est provisionné automatiquement au démarrage depuis "
        "`deployments/grafana/provisioning/` (datasource + panels). UI sur `localhost:3000`."
    )
    st.divider()

    st.markdown("#### Dashboard provisionné automatiquement")
    st.caption(
        "Le dashboard est chargé au démarrage depuis `deployments/grafana/provisioning/` "
        "(datasource Prometheus + 10 panels). Aucune configuration manuelle requise."
    )
    st.markdown("""
| Panel | Métrique |
|-------|---------|
| Requêtes / min | `rate(http_requests_total[1m])` |
| Taux d'erreur 5xx | `rate(http_requests_total{status=~"5.."}[1m])` |
| Latence p95 | `histogram_quantile(0.95, ...)` |
| Modèle chargé | `velib_model_loaded` |
| Version modèle | `velib_model_version` |
| R² modèle | `velib_model_r2` |
| MAE modèle | `velib_model_mae` |
| Latences par percentile | p50 / p90 / p99 |
| Codes HTTP | répartition 2xx / 4xx / 5xx |
| Rechargements modèle | `velib_model_reloads_total` |
""")
    st.markdown(
        '<div class="link-btn"><a href="http://localhost:3000" target="_blank">Ouvrir Grafana ↗</a></div>',
        unsafe_allow_html=True,
    )

# ─── Airflow ─────────────────────────────────────────────────────────────────
with tab_airflow:
    st.markdown("### Airflow — tests fonctionnels")
    st.markdown(
        "Airflow orchestre le pipeline quotidien à 04h00 UTC (cron `0 4 * * *`). "
        "DAG `velib_pipeline` — TaskFlow API Airflow 2.x, LocalExecutor, `dagrun_timeout=7h`. "
        "UI sur `localhost:8090`."
    )
    st.divider()

    st.markdown("#### DAG `velib_pipeline` — orchestration quotidienne")
    st.caption("Exécution automatique tous les jours à 04h00 UTC via LocalExecutor.")
    st.code("""
preflight_checks  (@task_group — parallèle)
  ├─ check_hf_connectivity
  ├─ check_mlflow_health
  └─ check_api_alive
        ▼
dvc_status_check  (@task.short_circuit)
  → court-circuit si dvc status = up-to-date
        ▼
dvc_repro  (BashOperator — exec_timeout 3h, 1 retry exponentiel)
  → 5 stages DVC : load_from_hf → make_dataset → dataviz
                   → build_features → train_model → detect_drift
        ▼                              ▼
parse_metrics (@task)          parse_drift (@task — parallèle)
  → XCom : run_id, r2,           → XCom : n_drifted_features,
    mae, mape, timestamp           share_drifted (informatif)
        ▼─────────────────────────────┘
gate_metrics (@task)
  → seuils : R² ≥ 0.75 | MAE ≤ 12.0 | MAPE ≤ 50 %
        ▼
branch_on_gate (@task.branch)
  ├─ deployment.promote_model   → POST /model/reload + jq alias=staging
  └─ deployment.skip_promotion  → EmptyOperator (modèle précédent conservé)
        ▼  (TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS)
deployment.smoke_test  → GET /health | jq '.status == "ok"'
""", language="text")
    st.markdown("""
| Tâche | Rôle | Retries |
|-------|------|---------|
| `preflight_checks` | HF + MLflow + API joignables (parallèle) | 3 / 1 / 1 |
| `dvc_status_check` | Court-circuit si pipeline à jour | 0 |
| `dvc_repro` | Pipeline complet (6 stages dont detect_drift) | 1 exp. |
| `parse_metrics` | Lecture metrics.json → XCom typé | 0 |
| `parse_drift` | Lecture drift_metrics.json → XCom typé | 0 |
| `gate_metrics` | Quality gate R²/MAE/MAPE (bloquant) | 0 |
| `branch_on_gate` | Branche selon résultat du gate | 0 |
| `promote_model` | `POST /model/reload` + validation jq | 1 |
| `skip_promotion` | No-op — modèle précédent reste en staging | 0 |
| `smoke_test` | `GET /health` → `status == "ok"` | 1 |
""")
    st.markdown(
        '<div class="link-btn"><a href="http://localhost:8090" target="_blank">Ouvrir Airflow ↗</a></div>',
        unsafe_allow_html=True,
    )

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
