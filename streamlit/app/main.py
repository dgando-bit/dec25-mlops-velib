import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import plotly.graph_objects as go
from collections import Counter
from datetime import date, timedelta
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

# ── Dataset HuggingFace ───────────────────────────────────────────────────────
st.markdown("### Données — Continuité des relevés HuggingFace")


@st.cache_data(ttl=300, show_spinner=False)
def fetch_hf_data() -> tuple[int, dict, bool, list]:
    info_code, info = api_client.hf_dataset_info()
    if info_code != 200:
        return info_code, {}, False, []
    ok, dates = api_client.hf_dataset_commits()
    return info_code, info, ok, dates


with st.spinner("Récupération de l'historique HuggingFace… (première visite : ~15 s, puis mis en cache 5 min)"):
    hf_code, hf_info, hf_ok, hf_dates = fetch_hf_data()

col_badge, col_chart = st.columns([1, 3], gap="large")

with col_badge:
    if hf_code == 200:
        st.markdown(
            '<div style="background:#d4edda;border-radius:10px;padding:1.2rem;text-align:center;">'
            '<div style="font-size:2rem">🟢</div>'
            '<div style="font-weight:700;color:#2e7d32;font-size:0.9rem">HuggingFace</div>'
            '<div style="font-size:0.78rem;color:#555;margin-top:0.3rem">Dataset accessible</div>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div style="background:#f8d7da;border-radius:10px;padding:1.2rem;text-align:center;">'
            '<div style="font-size:2rem">🔴</div>'
            '<div style="font-weight:700;color:#c53030;font-size:0.9rem">HuggingFace</div>'
            f'<div style="font-size:0.78rem;color:#555;margin-top:0.3rem">HTTP {hf_code}</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    if hf_info:
        created = hf_info.get("created_at", "")[:10]
        modified = hf_info.get("last_modified", "")[:10]
        if created:
            st.metric("Collecte depuis", created)
        if modified:
            st.metric("Dernier relevé", modified)

    if hf_dates:
        first_day = hf_dates[0]
        last_day = hf_dates[-1]
        total_days = (last_day - first_day).days + 1
        st.metric("Total relevés", f"{len(hf_dates):,}")
        st.metric("Jours couverts", f"{total_days} j")

with col_chart:
    if hf_ok and hf_dates:
        counts = Counter(hf_dates)
        first_day = hf_dates[0]
        last_day = hf_dates[-1]
        all_days = [first_day + timedelta(days=i) for i in range((last_day - first_day).days + 1)]
        y_vals = [counts.get(d, 0) for d in all_days]

        EXPECTED = 288
        colors = [
            "#5bac3a" if v >= 200
            else "#f6a623" if v >= 50
            else "#e53e3e"
            for v in y_vals
        ]

        fig = go.Figure()
        fig.add_bar(x=all_days, y=y_vals, marker_color=colors, name="Relevés / jour")
        fig.add_hline(
            y=EXPECTED, line_dash="dot", line_color="#718096", line_width=1,
            annotation_text="288 / jour (cible)", annotation_position="top right",
            annotation_font_size=11,
        )
        fig.update_layout(
            title="Continuité de la collecte — relevés par jour",
            xaxis_title=None,
            yaxis_title="Relevés",
            height=270,
            margin=dict(t=45, b=20, l=40, r=20),
            paper_bgcolor="#f7f8fa",
            plot_bgcolor="#fff",
            showlegend=False,
            font={"family": "sans-serif", "size": 12},
        )
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(gridcolor="#edf2f7", range=[0, EXPECTED + 40])
        st.plotly_chart(fig, use_container_width=True)
        st.caption(
            "Un relevé est collecté toutes les 5 min via cron-job.org → HuggingFace Space → HuggingFace Datasets. "
            "🟢 ≥ 200 · 🟡 50–200 · 🔴 < 50 — le dernier jour en cours est partiel."
        )
    elif not hf_ok:
        st.info("Impossible de récupérer l'historique HuggingFace.")

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
        ("Tracking", "MLflow 2.22.0"),
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
    ("Prometheus", "http://localhost:9090", "Métriques brutes, alertes", False),
]

cols = st.columns(4)
for i, (name, url, desc, primary) in enumerate(ui_services):
    with cols[i % 4]:
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
st.caption("🚲 Vélib' MLOps · DataScientest promotion décembre 2025")
