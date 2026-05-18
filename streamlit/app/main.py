import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from utils import api_client

st.set_page_config(
    page_title="Vélib' MLOps",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #f0f8eb; }
  .service-card {
    background: #fff;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    margin-bottom: 0.5rem;
    border-left: 4px solid #5bac3a;
  }
  .link-btn a {
    display: inline-block;
    background: #5bac3a;
    color: #fff !important;
    padding: 0.45rem 1.1rem;
    border-radius: 7px;
    font-size: 0.85rem;
    font-weight: 600;
    text-decoration: none;
    margin: 0.2rem;
    transition: opacity .15s;
  }
  .link-btn a:hover { opacity: 0.85; }
  .link-btn-secondary a {
    background: #edf2f7;
    color: #4a5568 !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────
st.markdown("## 🚲 Vélib' MLOps")
st.caption("Système de prédiction du taux de remplissage · Formation ML Engineer — DataScientest (déc. 2025)")
st.divider()

# ── Live status ──────────────────────────────────────────────────────────────
st.markdown("### État des services")


@st.cache_data(ttl=15, show_spinner=False)
def fetch_statuses() -> dict:
    checks = {
        "API FastAPI": api_client.api_health,
        "MLflow": api_client.mlflow_health,
        "Prometheus": api_client.prometheus_health,
        "Grafana": api_client.grafana_health,
        "Airflow": api_client.airflow_health,
    }
    results = {}
    for name, fn in checks.items():
        code, _ = fn()
        if code in (200, 201, 204):
            results[name] = ("🟢", "OK", "#d4edda")
        elif code == 0:
            results[name] = ("🔴", "Hors ligne", "#f8d7da")
        elif code == 408:
            results[name] = ("🟡", "Timeout", "#fff3cd")
        else:
            results[name] = ("🟡", f"HTTP {code}", "#fff3cd")
    return results


with st.spinner("Vérification des services…"):
    statuses = fetch_statuses()

cols = st.columns(len(statuses))
for col, (name, (icon, label, color)) in zip(cols, statuses.items()):
    with col:
        st.markdown(
            f'<div style="background:{color};border-radius:8px;padding:0.6rem 1rem;text-align:center;">'
            f'<div style="font-size:1.4rem">{icon}</div>'
            f'<div style="font-weight:700;font-size:0.82rem">{name}</div>'
            f'<div style="font-size:0.75rem;color:#555">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.caption("Rafraîchissement automatique toutes les 15 s — appuyez sur ↺ pour forcer.")
st.divider()

# ── Architecture ─────────────────────────────────────────────────────────────
st.markdown("### Architecture")

left, right = st.columns([1.2, 1])

with left:
    st.code("""
  Internet
     │
  ┌──────── Nginx (point d'entrée unique) ────────┐
  │  :8080 → API FastAPI                          │
  │  :5000 → MLflow UI                            │
  │  :8888 → JupyterLab                           │
  │  :9090 → Prometheus                           │
  │  :3000 → Grafana                              │
  │  :8090 → Airflow UI                           │
  │  :8501 → Streamlit (cette app)                │
  └───────────────────────────────────────────────┘
     │
  Pipeline DVC (5 stages, orchestré par Airflow)
  load_from_hf → make_dataset → build_features → train_model
                                                      │
                                              MLflow Registry
                                              alias: staging
""", language="text")

with right:
    st.markdown("#### Stack")
    stack = [
        ("Langage", "Python 3.12"),
        ("API", "FastAPI + Uvicorn"),
        ("ML", "XGBoost 3.0.2"),
        ("Tracking", "MLflow 2.20.3"),
        ("Données", "DVC + HuggingFace"),
        ("Orchestration", "Airflow 2.11.2"),
        ("Monitoring", "Prometheus + Grafana"),
        ("Proxy", "Nginx"),
        ("Tests", "pytest (94 tests)"),
        ("Conteneurs", "Docker Compose v2"),
    ]
    for tech, val in stack:
        st.markdown(
            f'<div style="display:flex;justify-content:space-between;'
            f'padding:0.3rem 0.5rem;border-bottom:1px solid #edf2f7;font-size:0.875rem">'
            f'<span style="color:#718096">{tech}</span>'
            f'<span style="font-weight:600">{val}</span></div>',
            unsafe_allow_html=True,
        )

st.divider()

# ── Métriques modèle ─────────────────────────────────────────────────────────
st.markdown("### Modèle ML — Dernières métriques")
m1, m2, m3, m4 = st.columns(4)
m1.metric("R² (taux reconstruit)", "0.833")
m2.metric("MAE", "8.32 pp")
m3.metric("RMSE", "12.08 pp")
m4.metric("Stations couvertes", "1 492")
st.caption("Run `ea250421` · XGBoost sur résidu (taux − station_trend_avg) · 24 features · 15 mai 2026")
st.divider()

# ── Accès direct aux interfaces natives ──────────────────────────────────────
st.markdown("### Interfaces natives")

ui_services = [
    ("MLflow UI", "http://localhost:5000", "Expériences, Registry, artefacts", True),
    ("Grafana", "http://localhost:3000", "Dashboard monitoring temps réel", True),
    ("Airflow UI", "http://localhost:8090", "DAG velib_pipeline, runs, logs", True),
    ("JupyterLab", "http://localhost:8888", "Notebooks d'exploration", True),
    ("Prometheus", "http://localhost:9090", "Métriques brutes, alertes", False),
    ("Swagger API", "http://localhost:8080/docs", "Documentation OpenAPI interactive", True),
]

cols = st.columns(3)
for i, (name, url, desc, primary) in enumerate(ui_services):
    with cols[i % 3]:
        btn_class = "link-btn" if primary else "link-btn link-btn-secondary"
        st.markdown(
            f'<div class="service-card">'
            f'<div style="font-weight:700;margin-bottom:0.25rem">{name}</div>'
            f'<div style="font-size:0.8rem;color:#718096;margin-bottom:0.6rem">{desc}</div>'
            f'<div class="{btn_class}"><a href="{url}" target="_blank">Ouvrir ↗</a></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

st.divider()
st.caption("🚲 Vélib' MLOps · DataScientest promotion décembre 2025 · kumnito")
