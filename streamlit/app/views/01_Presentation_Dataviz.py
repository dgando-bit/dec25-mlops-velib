import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from collections import Counter
from datetime import timedelta
from pathlib import Path
from plotly.subplots import make_subplots
from utils import api_client

st.set_page_config(
    page_title="Présentation & Dataviz - Vélib' MLOps",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #f0f8eb; }
  .pres-header {
    background: linear-gradient(135deg, #1b5e20 0%, #2e7d32 60%, #388e3c 100%);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
  }
  .pres-header h1 { margin: 0 0 0.4rem 0; font-size: 1.8rem; }
  .pres-header p  { margin: 0; opacity: 0.9; font-size: 0.95rem; }
  .kpi-box {
    background: #fff;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    border-top: 3px solid #5bac3a;
    text-align: center;
  }
  .kpi-val  { font-size: 1.5rem; font-weight: 700; color: #2e7d32; }
  .kpi-lbl  { font-size: 0.78rem; color: #718096; margin-top: 0.2rem; }
  .section-title {
    color: #2d6a1f;
    font-size: 1.05rem;
    font-weight: 700;
    border-bottom: 2px solid #c8e6c9;
    padding-bottom: 0.3rem;
    margin: 1.4rem 0 1rem 0;
  }
  .data-warning {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 6px;
    padding: 0.8rem 1rem;
    font-size: 0.88rem;
    color: #5d4037;
  }
  .hint-box {
    font-size: 0.80rem;
    color: #888;
    text-align: right;
    margin-top: 0.3rem;
  }
</style>
""", unsafe_allow_html=True)

# ─── Charte graphique (identique à dataviz.py) ───────────────────────────────

COLORS = {
    "primary":     "#2E7D32",
    "secondary":   "#F57C00",
    "neutral":     "#78909C",
    "danger":      "#C62828",
    "info":        "#1976D2",
    "success":     "#388E3C",
    "warning":     "#F9A825",
    "weather_sev": ["#FFC107", "#90A4AE", "#64B5F6", "#1976D2", "#6A1B9A"],
    "dow":         ["#1976D2"] * 5 + ["#F57C00"] * 2,
}

DAY_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
SEV_LABELS = ["0 Clair", "1 Nuageux", "2 Pluie légère", "3 Pluie forte", "4 Extrême"]

PLOT_TEMPLATE = "plotly_white"
PLOT_FONT = dict(family="Arial, sans-serif", size=12)
PLOT_HEIGHT = 450

BASE_DIR = Path(__file__).parent.parent  # app/


# ─── Helpers inline (pas d'import ml.src.*) ──────────────────────────────────

def _apply_weather_severity(weather_code: pd.Series) -> dict:
    code = weather_code.fillna(-1).astype(int)
    severity = pd.Series(0, index=code.index, dtype="int8")
    severity.loc[(code >= 4) & (code <= 19)] = 1
    severity.loc[((code >= 20) & (code <= 29)) | ((code >= 40) & (code <= 59))] = 2
    severity.loc[((code >= 60) & (code <= 69)) | ((code >= 80) & (code <= 84))] = 3
    severity.loc[((code >= 30) & (code <= 39)) | ((code >= 70) & (code <= 79)) | (code >= 85)] = 4
    frozen_codes = {56, 57, 66, 67, 85, 86, *range(70, 80)}
    is_frozen = code.isin(frozen_codes).astype("int8")
    stormy_codes = {17, 18, 19, 29, *range(90, 100)}
    is_stormy = code.isin(stormy_codes).astype("int8")
    return {"weather_severity": severity, "is_frozen": is_frozen, "is_stormy": is_stormy}


def _apply_layout(fig: go.Figure, title: str, height: int = PLOT_HEIGHT) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Arial", color="#212121")),
        template=PLOT_TEMPLATE,
        font=PLOT_FONT,
        height=height,
        margin=dict(l=60, r=40, t=60, b=50),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Arial",
                        bordercolor=COLORS["neutral"]),
    )
    return fig


# ─── 7 fonctions render (portage fidèle de dataviz.py) ───────────────────────

def render_temporal_coverage(df_raw: pd.DataFrame) -> go.Figure:
    daily = df_raw.groupby(df_raw["datetime"].dt.date).size().reset_index(name="count")
    daily.columns = ["date", "count"]
    dow_counts = df_raw.groupby(df_raw["datetime"].dt.dayofweek).size()
    n_days = (df_raw["datetime"].max() - df_raw["datetime"].min()).days
    n_weeks = round(n_days / 7, 1)
    n_stations = int(df_raw["station_id"].nunique())
    ticks_per_day = 24 * 60 // 5
    max_theoretical = n_stations * ticks_per_day
    y_min_data = int(daily["count"].min())
    y_max_data = int(daily["count"].max())
    y_axis_max = max(y_max_data, max_theoretical) * 1.05
    y_axis_min = y_min_data * 0.92

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Relevés par jour calendaire — {n_days} jours ({n_weeks} semaines)",
            "Relevés par jour de semaine",
        ),
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["count"], mode="lines+markers",
            line=dict(color=COLORS["info"], width=2),
            marker=dict(size=8, color=COLORS["info"], line=dict(color="white", width=1)),
            name="Relevés/jour",
            hovertemplate="<b>%{x|%a %d %b}</b><br>%{y:,} relevés<extra></extra>",
        ),
        row=1, col=1,
    )
    fig.add_hline(
        y=max_theoretical,
        line=dict(color=COLORS["success"], dash="dash", width=1.5),
        annotation_text=(
            f"Max théorique : {max_theoretical:,} "
            f"({n_stations} stations × {ticks_per_day} ticks/jour)"
        ),
        annotation_position="top right",
        annotation=dict(font=dict(size=10, color=COLORS["success"])),
        row=1, col=1,
    )
    fig.add_annotation(
        text=f"Echelle Y adaptée à la plage observée (début à {y_axis_min:,.0f})",
        xref="x domain", yref="y domain", x=0.02, y=0.04,
        showarrow=False,
        font=dict(size=10, color=COLORS["neutral"], family="Arial"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor=COLORS["neutral"],
        borderwidth=1, borderpad=4, row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=DAY_FR, y=dow_counts.values,
            marker=dict(color=COLORS["dow"], line=dict(color="white", width=1)),
            text=[f"{v:,}" for v in dow_counts.values],
            textposition="outside", textfont=dict(size=10),
            name="Relevés/jour de semaine",
            hovertemplate="<b>%{x}</b><br>%{y:,} relevés<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Nb relevés", row=1, col=1, tickformat=",",
                     range=[y_axis_min, y_axis_max])
    fig.update_xaxes(title_text="Jour", row=1, col=2)
    fig.update_yaxes(title_text="Nb relevés", row=1, col=2, tickformat=",")
    return _apply_layout(fig, "1. Couverture temporelle de la collecte", height=420)


def render_fill_rate_distribution(df: pd.DataFrame) -> go.Figure:
    mean_val = df["taux"].mean()
    median_val = df["taux"].median()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribution globale du taux",
            "Distribution par station (top 20 par volume)",
        ),
        column_widths=[0.50, 0.50],
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Histogram(
            x=df["taux"], nbinsx=50,
            marker=dict(color=COLORS["info"], line=dict(color="white", width=0.3)),
            name="Distribution",
            hovertemplate="Bin: %{x}%<br>Fréquence: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_vline(x=mean_val, line=dict(color=COLORS["danger"], dash="dash", width=2),
                  annotation_text=f"Moyenne {mean_val:.1f}%", annotation_position="top",
                  row=1, col=1)
    fig.add_vline(x=median_val, line=dict(color=COLORS["secondary"], dash="dot", width=2),
                  annotation_text=f"Médiane {median_val:.1f}%",
                  annotation_position="top right", row=1, col=1)
    top_stations = df["station_id"].value_counts().head(20).index.tolist()
    sample = df.loc[df["station_id"].isin(top_stations)].copy()
    for sid in top_stations:
        vals = sample.loc[sample["station_id"] == sid, "taux"]
        name = df.loc[df["station_id"] == sid, "name"].iloc[0]
        fig.add_trace(
            go.Box(
                y=vals, name=str(sid),
                marker=dict(color=COLORS["primary"]), line=dict(width=1),
                boxmean=True,
                hovertemplate=(
                    f"<b>{name}</b> (id={sid})<br>Taux: %{{y:.1f}}%<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="Taux (%)", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=1, tickformat=",")
    fig.update_xaxes(title_text="Station ID (top 20 par volume)", row=1, col=2, tickangle=45)
    fig.update_yaxes(title_text="Taux (%)", row=1, col=2)
    return _apply_layout(fig, "2. Distribution du taux de remplissage", height=480)


def render_temporal_patterns(df: pd.DataFrame) -> go.Figure:
    df_local = df.copy()
    df_local["hour"] = df_local["datetime"].dt.hour
    df_local["day_of_week"] = df_local["datetime"].dt.dayofweek
    hourly = df_local.groupby("hour")["taux"].mean()
    daily = df_local.groupby("day_of_week")["taux"].mean()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Taux moyen par heure", "Taux moyen par jour de semaine"),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Scatter(
            x=hourly.index, y=hourly.values, mode="lines+markers",
            line=dict(color=COLORS["info"], width=2.5),
            marker=dict(size=7, color=COLORS["info"]),
            name="Taux moyen",
            hovertemplate="<b>%{x}h</b><br>Taux moyen: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    for h_start, h_end, label in [(7, 9, "Pointe matin"), (17, 19, "Pointe soir")]:
        fig.add_vrect(
            x0=h_start, x1=h_end,
            fillcolor=COLORS["primary"], opacity=0.12, line_width=0,
            annotation_text=label, annotation_position="top",
            annotation=dict(font=dict(size=10, color=COLORS["primary"])),
            row=1, col=1,
        )
    fig.add_trace(
        go.Bar(
            x=DAY_FR, y=daily.values,
            marker=dict(color=COLORS["dow"], line=dict(color="white", width=1)),
            text=[f"{v:.1f}%" for v in daily.values], textposition="outside",
            hovertemplate="<b>%{x}</b><br>Taux moyen: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Heure", row=1, col=1, tickmode="linear", dtick=2)
    fig.update_yaxes(title_text="Taux moyen (%)", row=1, col=1)
    fig.update_xaxes(title_text="Jour", row=1, col=2)
    fig.update_yaxes(title_text="Taux moyen (%)", row=1, col=2)
    return _apply_layout(fig, "3. Profils temporels", height=420)


def render_weather_impact(df: pd.DataFrame) -> go.Figure:
    weather_features = _apply_weather_severity(df["weather_code"])
    df_local = df.assign(weather_severity=weather_features["weather_severity"])
    sev_counts = df_local["weather_severity"].value_counts().sort_index()
    mean_by_sev = df_local.groupby("weather_severity")["taux"].mean()
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Répartition des conditions météo", "Taux moyen par condition"),
        horizontal_spacing=0.10,
    )
    sev_idx = sev_counts.index.tolist()
    fig.add_trace(
        go.Bar(
            x=[SEV_LABELS[i] for i in sev_idx],
            y=sev_counts.values,
            marker=dict(color=[COLORS["weather_sev"][i] for i in sev_idx],
                        line=dict(color="white", width=1)),
            text=[f"{v:,}" for v in sev_counts.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:,} relevés<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    sev2 = mean_by_sev.index.tolist()
    fig.add_trace(
        go.Bar(
            x=[SEV_LABELS[i] for i in sev2],
            y=mean_by_sev.values,
            marker=dict(color=[COLORS["weather_sev"][i] for i in sev2],
                        line=dict(color="white", width=1)),
            text=[f"{v:.1f}%" for v in mean_by_sev.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>Taux moyen: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Condition", row=1, col=1)
    fig.update_yaxes(title_text="Nb relevés", row=1, col=1, tickformat=",")
    fig.update_xaxes(title_text="Condition", row=1, col=2)
    fig.update_yaxes(title_text="Taux moyen (%)", row=1, col=2)
    return _apply_layout(fig, "4. Impact des conditions météo", height=440)


def render_temp_anomaly(df: pd.DataFrame) -> go.Figure:
    month = df["datetime"].dt.month
    monthly_mean = df.groupby(month)["apparent_temperature"].transform("mean")
    temp_anom = df["apparent_temperature"] - monthly_mean

    # Pré-agrégation serveur : 60 bins → 60 valeurs envoyées au browser (vs 11M+)
    counts, edges = np.histogram(temp_anom, bins=60)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribution de l'anomalie thermique",
            "Anomalie thermique vs taux de remplissage (échantillon 8 000 pts)",
        ),
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Bar(
            x=bin_centers, y=counts,
            marker=dict(color=COLORS["info"], line=dict(color="white", width=0.3)),
            hovertemplate="Anomalie: %{x:.1f}°C<br>Fréquence: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_vline(x=0, line=dict(color=COLORS["danger"], dash="dash", width=2),
                  annotation_text="0 = normale du mois", annotation_position="top",
                  row=1, col=1)
    sample_idx = np.random.default_rng(42).choice(len(df), size=min(8000, len(df)), replace=False)
    sample = df.iloc[sample_idx]
    sample_anom = temp_anom.iloc[sample_idx]
    fig.add_trace(
        go.Scattergl(
            x=sample_anom, y=sample["taux"], mode="markers",
            marker=dict(size=4, color=COLORS["info"], opacity=0.25, line=dict(width=0)),
            hovertemplate="Anomalie: %{x:.1f}°C<br>Taux: %{y:.1f}%<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )
    fig.update_xaxes(title_text="Écart à la normale (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=1, tickformat=",")
    fig.update_xaxes(title_text="Anomalie (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Taux (%)", row=1, col=2)
    return _apply_layout(fig, "5. Anomalie thermique", height=440)


def render_station_profiles(df: pd.DataFrame) -> go.Figure:
    df_local = df.copy()
    df_local["hour"] = df_local["datetime"].dt.hour
    morning_avg = (
        df_local.loc[df_local["hour"].isin([7, 8, 9])]
        .groupby("station_id")["taux"].mean()
    )
    evening_avg = (
        df_local.loc[df_local["hour"].isin([17, 18, 19])]
        .groupby("station_id")["taux"].mean()
    )
    station_ratio = (morning_avg / evening_avg.replace(0, np.nan)).dropna()
    n_res = int((station_ratio > 1.2).sum())
    n_bur = int((station_ratio < 0.8).sum())
    n_mix = int(((station_ratio >= 0.8) & (station_ratio <= 1.2)).sum())

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribution du ratio matin/soir par station",
            "Répartition des profils fonctionnels",
        ),
        specs=[[{"type": "xy"}, {"type": "domain"}]],
        horizontal_spacing=0.10,
    )
    fig.add_trace(
        go.Histogram(
            x=station_ratio.values, nbinsx=40,
            marker=dict(color=COLORS["info"], line=dict(color="white", width=0.3)),
            hovertemplate="Ratio: %{x:.2f}<br>Stations: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_vline(x=1.2, line=dict(color=COLORS["success"], dash="dash", width=2),
                  annotation_text=f"Résidentiel >1.2 ({n_res})",
                  annotation_position="top right", row=1, col=1)
    fig.add_vline(x=0.8, line=dict(color=COLORS["danger"], dash="dash", width=2),
                  annotation_text=f"Bureaux <0.8 ({n_bur})",
                  annotation_position="top left", row=1, col=1)
    fig.add_vline(x=1.0, line=dict(color=COLORS["neutral"], dash="dot", width=1),
                  row=1, col=1)
    pie_data = [
        (n_res, "Résidentiel", COLORS["success"]),
        (n_mix, "Mixte",       COLORS["info"]),
        (n_bur, "Bureaux",     COLORS["secondary"]),
    ]
    pie_data = [(n, lbl, col) for n, lbl, col in pie_data if n > 0]
    if pie_data:
        fig.add_trace(
            go.Pie(
                values=[d[0] for d in pie_data],
                labels=[d[1] for d in pie_data],
                marker=dict(colors=[d[2] for d in pie_data],
                            line=dict(color="white", width=2)),
                textinfo="label+percent+value",
                textfont=dict(size=12),
                hovertemplate=(
                    "<b>%{label}</b><br>%{value} stations (%{percent})<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )
    fig.update_xaxes(title_text="Ratio matin / soir", row=1, col=1)
    fig.update_yaxes(title_text="Nb stations", row=1, col=1)
    return _apply_layout(fig, "6. Profils fonctionnels des stations", height=460)


def render_vacation_impact(df: pd.DataFrame) -> go.Figure:
    vac_mean = df.groupby(df["is_vacation"].astype(bool))["taux"].mean()
    if len(vac_mean) < 2:
        fig = go.Figure()
        fig.add_annotation(
            text="Données insuffisantes (période contenant uniquement vacances ou hors-vacances)",
            xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["neutral"]),
        )
        return _apply_layout(fig, "7. Impact des vacances scolaires", height=380)
    labels = ["Hors vacances", "Vacances scolaires"]
    values = [float(vac_mean.get(False, np.nan)), float(vac_mean.get(True, np.nan))]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels, y=values,
            marker=dict(color=[COLORS["info"], COLORS["secondary"]],
                        line=dict(color="white", width=1)),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside", textfont=dict(size=14),
            width=0.4,
            hovertemplate="<b>%{x}</b><br>Taux moyen: %{y:.2f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text="Période")
    fig.update_yaxes(title_text="Taux moyen (%)")
    return _apply_layout(fig, "7. Impact des vacances scolaires", height=400)


# ─── Chargement des données ───────────────────────────────────────────────────

@st.cache_data(ttl=600, show_spinner=False)
def load_parquets() -> tuple[pd.DataFrame | None, pd.DataFrame | None, str]:
    root = os.environ.get("HOST_PROJECT_ROOT", "")
    if not root:
        return None, None, "Variable HOST_PROJECT_ROOT non définie dans l'environnement."
    root_path = Path(root)
    raw_path     = root_path / "data" / "raw"     / "velib_snapshot_latest.parquet"
    interim_path = root_path / "data" / "interim" / "velib_cleaned_latest.parquet"
    missing = []
    if not raw_path.exists():
        missing.append(str(raw_path))
    if not interim_path.exists():
        missing.append(str(interim_path))
    if missing:
        return None, None, (
            "Fichiers parquet introuvables :\n" + "\n".join(f"  • {p}" for p in missing)
            + "\n\nLancer `make pipeline` pour générer les données."
        )
    try:
        df_raw   = pd.read_parquet(raw_path)
        df_clean = pd.read_parquet(interim_path)
    except Exception as exc:
        return None, None, f"Erreur de lecture parquet : {exc}"
    return df_raw, df_clean, ""


# ─── SVG zoomable (portage de 03_Architecture.py) ────────────────────────────

@st.cache_data
def _load_svg(rel_path: str) -> str | None:
    full = BASE_DIR / rel_path
    return full.read_text(encoding="utf-8") if full.exists() else None


def _render_svg_zoomable(svg_path: str, height: int = 540, key: str = "diag-pres"):
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
          controlIconsEnabled: true,
          fit: true,
          center: true,
          minZoom: 0.3,
          maxZoom: 12,
          mouseWheelZoomEnabled: true
        }});
      }})();
    </script>
    """
    components.html(html, height=height + 10, scrolling=False)


# ─── En-tête ─────────────────────────────────────────────────────────────────

st.markdown("""
<div class="pres-header">
  <h1>Vélib' MLOps — Présentation du projet</h1>
  <p>
    Formation ML Engineer · DataScientest · Promotion décembre 2025<br>
    Prédiction du taux de remplissage des stations Vélib' Paris
    à partir d'un pipeline ML complet orchestré avec Airflow et MLflow.
  </p>
</div>
""", unsafe_allow_html=True)

# ─── Contexte projet ─────────────────────────────────────────────────────────

with st.expander("Contexte & Objectif", expanded=True):
    col_obj, col_stack = st.columns([3, 2], gap="large")
    with col_obj:
        st.markdown("""
**Problématique** : prévoir si une station Vélib' sera disponible (trop pleine ou trop vide)
dans les prochaines minutes, afin d'aider les usagers à anticiper leurs déplacements.

**Approche ML** : modèle XGBoost entraîné sur le *résidu* (taux observé − tendance historique par station).
Cette décomposition permet d'isoler le signal conjoncturel (météo, événements, anomalies)
de la tendance structurelle, et produit un modèle nettement plus précis.

**Pipeline** (6 stages DVC orchestrés par Airflow, toutes les 6 h) :
- `load_from_hf` → collecte depuis HuggingFace
- `make_dataset` → nettoyage, calcul du taux de remplissage
- `build_features` → 24 features (temporelles, météo, lags, géo)
- `train_model` → XGBoost + MLflow Registry (alias `staging`)
- `detect_drift` → Evidently (train vs test)
- `dataviz` → rapport HTML interactif
        """)
    with col_stack:
        st.markdown("**Stack technique**")
        stack = [
            ("Modèle ML", "XGBoost 3.0.2"),
            ("Tracking", "MLflow 2.22.0"),
            ("Orchestration", "Airflow 2.10.4 (Celery)"),
            ("API", "FastAPI + Uvicorn"),
            ("Données", "Pandas 2.3.3 + DVC"),
            ("Monitoring", "Prometheus + Grafana"),
            ("Serving", "Nginx + Streamlit"),
            ("Source", "HuggingFace Datasets"),
        ]
        for label, val in stack:
            st.markdown(f"- **{label}** : {val}")

st.divider()

# ─── Schémas de référence ─────────────────────────────────────────────────────

st.markdown('<div class="section-title">Schémas de référence</div>', unsafe_allow_html=True)
tab_pip, tab_flow = st.tabs(["Pipeline ML de bout en bout", "Flux de données & artefacts"])

with tab_pip:
    st.caption(
        "6 stages DVC enchaînés : load_from_hf → make_dataset → build_features → "
        "train_model (XGBoost) → detect_drift (Evidently). dataviz produit un rapport HTML en parallèle. "
        "Le modèle est enregistré dans le MLflow Registry sous l'alias staging."
    )
    _render_svg_zoomable("assets/diagrams/02_pipeline.svg", height=500, key="diag-pipeline")

with tab_flow:
    st.caption(
        "Provenance et stockage des artefacts : snapshots HuggingFace → data/ (bind mount DVC) "
        "→ mlflow-db Postgres (métadonnées de runs) + mlflow/artifacts (modèle PKL). "
        "Le MLflow Registry expose l'alias staging à l'API."
    )
    _render_svg_zoomable("assets/diagrams/03_data_flow.svg", height=500, key="diag-dataflow")

st.divider()

# ─── KPIs modèle ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-title">Métriques du modèle en production</div>', unsafe_allow_html=True)


@st.cache_data(ttl=60, show_spinner=False)
def _fetch_model_kpis() -> dict:
    from datetime import date
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


with st.spinner("Récupération des métriques…"):
    kpis = _fetch_model_kpis()

kpi_defs = [
    ("R² (taux reconstruit)", "taux_r2",        "{:.3f}",    "Variance expliquée — 1.0 = parfait"),
    ("MAE",                   "taux_mae",        "{:.2f} pp", "Erreur absolue moyenne (points de pourcentage)"),
    ("RMSE",                  "taux_rmse",       "{:.2f} pp", "Racine de l'erreur quadratique moyenne"),
    ("MAPE",                  "taux_mape_pct",   "{:.1f} %",  "Erreur relative moyenne"),
]

kpi_cols = st.columns(4)
for col, (label, key, fmt, tooltip) in zip(kpi_cols, kpi_defs):
    val = kpis.get(key)
    display = fmt.format(val) if val is not None else "—"
    with col:
        st.markdown(
            f'<div class="kpi-box">'
            f'<div class="kpi-val">{display}</div>'
            f'<div class="kpi-lbl">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.caption(tooltip)

if kpis.get("run_id"):
    st.caption(
        f"Run `{kpis['run_id'][:8]}` · v{kpis.get('version', '?')} · "
        f"{kpis.get('n_features', 24)} features · {kpis.get('run_date', '—')}"
    )
else:
    st.caption("API ou MLflow inaccessible — métriques indisponibles.")

st.divider()

# ─── Exploration des données ──────────────────────────────────────────────────

st.markdown('<div class="section-title">Exploration des données — Visualisations interactives</div>',
            unsafe_allow_html=True)

with st.spinner("Chargement des données (parquet)…"):
    df_raw, df_clean, data_err = load_parquets()

if data_err:
    st.markdown(
        f'<div class="data-warning"><b>Données non disponibles</b><br><pre>{data_err}</pre></div>',
        unsafe_allow_html=True,
    )
else:
    # Métadonnées du dataset
    period_start = df_raw["datetime"].min().date()
    period_end   = df_raw["datetime"].max().date()
    n_raw    = len(df_raw)
    n_clean  = len(df_clean)
    n_sta    = int(df_clean["station_id"].nunique())
    taux_avg = df_clean["taux"].mean()

    m_cols = st.columns(5)
    for col, (label, val) in zip(m_cols, [
        ("Période",          f"{period_start} → {period_end}"),
        ("Relevés bruts",    f"{n_raw:,}"),
        ("Relevés nettoyés", f"{n_clean:,}"),
        ("Stations",         f"{n_sta:,}"),
        ("Taux moyen",       f"{taux_avg:.1f} %"),
    ]):
        with col:
            st.markdown(
                f'<div class="kpi-box">'
                f'<div class="kpi-val" style="font-size:1.1rem">{val}</div>'
                f'<div class="kpi-lbl">{label}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    TABS = [
        ("Couverture temporelle",      render_temporal_coverage,    "df_raw"),
        ("Distribution du taux",       render_fill_rate_distribution, "df_clean"),
        ("Profils temporels",          render_temporal_patterns,    "df_clean"),
        ("Impact météo",               render_weather_impact,       "df_clean"),
        ("Anomalie thermique",         render_temp_anomaly,         "df_clean"),
        ("Profils stations",           render_station_profiles,     "df_clean"),
        ("Vacances scolaires",         render_vacation_impact,      "df_clean"),
    ]

    tabs = st.tabs([t[0] for t in TABS])
    data_map = {"df_raw": df_raw, "df_clean": df_clean}

    for tab, (label, render_fn, df_key) in zip(tabs, TABS):
        with tab:
            try:
                with st.spinner(f"Génération du graphique «{label}»…"):
                    fig = render_fn(data_map[df_key])
                st.plotly_chart(fig, use_container_width=True)
            except Exception as exc:
                st.error(f"Erreur lors du rendu du graphique : {exc}")

st.divider()

# ─── Continuité HuggingFace ───────────────────────────────────────────────────

st.markdown('<div class="section-title">Continuité de la collecte — HuggingFace</div>',
            unsafe_allow_html=True)


@st.cache_data(ttl=300, show_spinner=False)
def _fetch_hf() -> tuple[int, dict, bool, list]:
    info_code, info = api_client.hf_dataset_info()
    if info_code != 200:
        return info_code, {}, False, []
    ok, dates = api_client.hf_dataset_commits()
    return info_code, info, ok, dates


with st.spinner("Récupération de l'historique HuggingFace… (mise en cache 5 min)"):
    hf_code, hf_info, hf_ok, hf_dates = _fetch_hf()

hf_col_meta, hf_col_chart = st.columns([1, 3], gap="large")

with hf_col_meta:
    badge_bg = "#d4edda" if hf_code == 200 else "#f8d7da"
    badge_icon = "🟢" if hf_code == 200 else "🔴"
    badge_lbl  = "Dataset accessible" if hf_code == 200 else f"HTTP {hf_code}"
    st.markdown(
        f'<div style="background:{badge_bg};border-radius:10px;padding:1.2rem;text-align:center;">'
        f'<div style="font-size:2rem">{badge_icon}</div>'
        f'<div style="font-weight:700;font-size:0.9rem">HuggingFace</div>'
        f'<div style="font-size:0.78rem;color:#555;margin-top:0.3rem">{badge_lbl}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    if hf_info:
        if hf_info.get("created_at"):
            st.metric("Collecte depuis", hf_info["created_at"][:10])
        if hf_info.get("last_modified"):
            st.metric("Dernier relevé",  hf_info["last_modified"][:10])
    if hf_dates:
        first_day = hf_dates[0]
        last_day  = hf_dates[-1]
        total_days = (last_day - first_day).days + 1
        st.metric("Total commits",  f"{len(hf_dates):,}")
        st.metric("Jours couverts", f"{total_days} j")

with hf_col_chart:
    if hf_ok and hf_dates:
        counts   = Counter(hf_dates)
        first_d  = hf_dates[0]
        last_d   = hf_dates[-1]
        all_days = [first_d + timedelta(days=i) for i in range((last_d - first_d).days + 1)]
        y_vals   = [counts.get(d, 0) for d in all_days]
        EXPECTED = 288
        bar_colors = [
            "#5bac3a" if v >= 200 else "#f6a623" if v >= 50 else "#e53e3e"
            for v in y_vals
        ]
        fig_hf = go.Figure()
        fig_hf.add_bar(x=all_days, y=y_vals, marker_color=bar_colors, name="Commits / jour")
        fig_hf.add_hline(
            y=EXPECTED, line_dash="dot", line_color="#718096", line_width=1,
            annotation_text="288 / jour (cible)", annotation_position="top right",
            annotation_font_size=11,
        )
        fig_hf.update_layout(
            title="Continuité de la collecte — commits par jour",
            xaxis_title=None, yaxis_title="Commits",
            height=280,
            margin=dict(t=45, b=20, l=40, r=20),
            paper_bgcolor="#f7f8fa", plot_bgcolor="#fff",
            showlegend=False,
            font={"family": "sans-serif", "size": 12},
        )
        fig_hf.update_xaxes(showgrid=False)
        fig_hf.update_yaxes(gridcolor="#edf2f7", range=[0, EXPECTED + 40])
        st.plotly_chart(fig_hf, use_container_width=True)
        st.caption(
            "Relevé toutes les 5 min via cron-job.org → HuggingFace Space → HuggingFace Datasets. "
            "🟢 ≥ 200 · 🟡 50–200 · 🔴 < 50 — le dernier jour en cours est partiel."
        )
    else:
        st.info("Historique HuggingFace indisponible.")

st.divider()

# ─── Vue d'ensemble architecture ─────────────────────────────────────────────

st.markdown('<div class="section-title">Vue d\'ensemble de l\'architecture</div>',
            unsafe_allow_html=True)
st.caption(
    "Schéma systémique : ingestion HuggingFace → pipeline ML → MLflow Registry → "
    "API serving, avec la couche observabilité en parallèle. "
    "Molette pour zoomer · clic-glisser pour déplacer."
)

_render_svg_zoomable("assets/diagrams/00_overview.svg", height=560)


st.divider()
st.caption("Vélib' MLOps · DataScientest promotion décembre 2025")
