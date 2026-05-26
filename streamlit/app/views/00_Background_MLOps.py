import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from pathlib import Path
from utils import api_client

BASE_DIR = Path(__file__).parent.parent  # views/ → app/

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

# ── Architecture des conteneurs ──────────────────────────────────────────────
st.markdown("### Architecture des conteneurs")
st.caption(
    "Topologie réseau des 17 services Docker sur mlops-net. "
    "Nginx route les 8 ports exposés vers leurs upstreams. "
    "CeleryExecutor : scheduler + worker consomment le broker Redis indépendamment."
)
_render_svg_zoomable("assets/diagrams/01_containers.svg", height=540)
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
