import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from collections import Counter
from datetime import date, timedelta
from pathlib import Path
from utils import api_client

BASE_DIR = Path(__file__).parent.parent  # pages/ → app/

st.set_page_config(
    page_title="Background MLOps · Vélib'",
    page_icon="🚲",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data
def _load_svg(rel_path: str) -> str | None:
    full = BASE_DIR / rel_path
    return full.read_text(encoding="utf-8") if full.exists() else None


def _render_svg_zoomable(svg_path: str, height: int = 520):
    svg = _load_svg(svg_path)
    if svg is None:
        st.warning(f"SVG introuvable : `{svg_path}`")
        return
    html = f"""
    <div id="diag-bg" style="width:100%;height:{height}px;border:1px solid #e6e6e6;
         border-radius:14px;overflow:hidden;background:#ffffff;">
      {svg}
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/svg-pan-zoom/3.6.1/svg-pan-zoom.min.js"></script>
    <script>
      (function() {{
        var el = document.querySelector('#diag-bg svg');
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

# ── Architecture des conteneurs ──────────────────────────────────────────────
st.markdown("### Architecture des conteneurs")
st.caption(
    "Topologie réseau des 17 services Docker sur mlops-net. "
    "Nginx route les 8 ports exposés vers leurs upstreams. "
    "CeleryExecutor : scheduler + worker consomment le broker Redis indépendamment."
)
_render_svg_zoomable("assets/diagrams/01_containers.svg", height=540)
st.divider()

# ── Métriques modèle ─────────────────────────────────────────────────────────
st.markdown("### Modèle ML — Dernières métriques")


@st.cache_data(ttl=30, show_spinner=False)
def fetch_model_metrics() -> dict:
    code, info = api_client.api_model_info()
    if code != 200 or not isinstance(info, dict):
        return {}
    run_id = info.get("run_id", "")
    result = {
        "version": info.get("version", "?"),
        "run_id": run_id,
        "n_features": info.get("n_features", 24),
    }
    if not run_id:
        return result
    mcode, mdata = api_client.mlflow_run_metrics(run_id)
    if mcode == 200 and mdata:
        result.update(mdata.get("metrics", {}))
        start_ms = mdata.get("info", {}).get("start_time")
        if start_ms:
            result["run_date"] = date.fromtimestamp(int(start_ms) / 1000).isoformat()
    return result


with st.spinner("Récupération des métriques modèle…"):
    model_metrics = fetch_model_metrics()

m1, m2, m3, m4 = st.columns(4)
if model_metrics:
    r2 = model_metrics.get("taux_r2")
    mae = model_metrics.get("taux_mae")
    rmse = model_metrics.get("taux_rmse")
    mape = model_metrics.get("taux_mape_pct")
    m1.metric("R² (taux reconstruit)", f"{r2:.3f}" if r2 is not None else "N/A")
    m2.metric("MAE", f"{mae:.2f} pp" if mae is not None else "N/A")
    m3.metric("RMSE", f"{rmse:.2f} pp" if rmse is not None else "N/A")
    m4.metric("MAPE", f"{mape:.1f} %" if mape is not None else "N/A")
    run_id = model_metrics.get("run_id", "")
    version = model_metrics.get("version", "?")
    n_features = model_metrics.get("n_features", 24)
    run_date = model_metrics.get("run_date", "—")
    st.caption(
        f"Run `{run_id[:8]}` · v{version} · "
        f"XGBoost sur résidu (taux − station_trend_avg) · {n_features} features · {run_date}"
    )
else:
    m1.metric("R² (taux reconstruit)", "—")
    m2.metric("MAE", "—")
    m3.metric("RMSE", "—")
    m4.metric("MAPE", "—")
    st.caption("API ou MLflow inaccessible — métriques indisponibles.")
st.divider()

# ── Accès direct aux interfaces natives ──────────────────────────────────────
st.markdown("### Interfaces natives")

ui_services = [
    ("MLflow UI",   "http://localhost:5000", "Expériences, Registry, artefacts",    True),
    ("Grafana",     "http://localhost:3000", "Dashboard monitoring temps réel",      True),
    ("Airflow UI",  "http://localhost:8090", "DAG velib_pipeline, runs, logs",       True),
    ("Prometheus",  "http://localhost:9090", "Métriques brutes, alertes",            False),
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
