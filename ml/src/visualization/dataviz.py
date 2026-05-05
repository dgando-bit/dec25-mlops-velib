"""
ml.src.visualization.dataviz — Exploration visuelle interactive du dataset Vélib'.

Refactoring de l'ancien ``01_dataviz.py`` adapté à l'architecture micro-services :
    - Plotly au lieu de matplotlib (HTML interactif au lieu de PNG statique)
    - Lit le parquet brut ET le parquet nettoyé (chacun a son rôle)
    - Charte graphique custom Vélib' centralisée
    - Pas d'échantillonnage : les 5M lignes sont rendues telles quelles
    - Sortie : 1 fichier HTML unifié avec navigation par onglets

Les 7 graphiques produits, fidèles à l'ancien code :
    1. Couverture temporelle (à partir du raw, pour détecter les trous de collecte)
    2. Distribution du taux de remplissage (histogramme + boxplot top 20)
    3. Profils temporels (taux par heure + taux par jour de semaine)
    4. Impact météo (répartition conditions + taux moyen par condition)
    5. Anomalie thermique (distribution + scatter vs taux)
    6. Profils fonctionnels stations (ratio matin/soir + camembert)
    7. Impact vacances scolaires (bar chart vacances vs hors vacances)

Pipeline DVC :
    Stage  : dataviz
    Entrées : data/raw/velib_snapshot_latest.parquet (graphe 1)
              data/interim/velib_cleaned_latest.parquet (graphes 2-7)
    Sortie  : data/outputs/plots/dataviz_report.html

Usage :
    # En CLI
    python -m ml.src.visualization.dataviz

    # En import depuis un autre module (Streamlit par exemple)
    from ml.src.visualization.dataviz import (
        render_temporal_coverage, render_fill_rate_distribution,
        render_temporal_patterns, render_weather_impact,
        render_temp_anomaly, render_station_profiles, render_vacation_impact,
    )
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from shared.config import settings
from shared.logger import get_logger
from shared.utils.data_cleaning import apply_weather_severity

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CHARTE GRAPHIQUE VÉLIB'
# ─────────────────────────────────────────────────────────────────────────────
# Palette inspirée du branding Vélib' (vert dominant) + couleurs ergonomiques
# pour les états (chaud/froid, weekend/semaine, sévérité météo).
COLORS = {
    "primary":      "#2E7D32",   # vert Vélib'
    "secondary":    "#F57C00",   # orange (weekend, ebike)
    "neutral":      "#78909C",   # gris-bleu (mixte)
    "danger":       "#C62828",   # rouge (alerte, moyenne)
    "info":         "#1976D2",   # bleu (information)
    "success":      "#388E3C",   # vert foncé (résidentiel)
    "warning":      "#F9A825",   # jaune (mixte)

    # Sévérité météo (5 niveaux 0-4)
    "weather_sev": ["#FFC107", "#90A4AE", "#64B5F6", "#1976D2", "#6A1B9A"],

    # Jours de la semaine (5 jours ouvrés bleus + 2 weekend orange)
    "dow": ["#1976D2"] * 5 + ["#F57C00"] * 2,
}

DAY_FR = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
SEV_LABELS = ["0 Clair", "1 Nuageux", "2 Pluie légère", "3 Pluie forte", "4 Extrême"]

# Template Plotly cohérent pour tous les graphes
PLOT_TEMPLATE = "plotly_white"
PLOT_FONT = dict(family="Arial, sans-serif", size=12)
PLOT_HEIGHT = 450


def _apply_layout(fig: go.Figure, title: str, height: int = PLOT_HEIGHT) -> go.Figure:
    """Applique la charte graphique commune à toutes les figures."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, family="Arial", color="#212121")),
        template=PLOT_TEMPLATE,
        font=PLOT_FONT,
        height=height,
        margin=dict(l=60, r=40, t=60, b=50),
        hoverlabel=dict(
            bgcolor="white", font_size=12, font_family="Arial",
            bordercolor=COLORS["neutral"],
        ),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE 1 — COUVERTURE TEMPORELLE (utilise le RAW, pas le cleaned)
# ─────────────────────────────────────────────────────────────────────────────
def render_temporal_coverage(df_raw: pd.DataFrame) -> go.Figure:
    """Graphe 1 : nombre de relevés par jour calendaire + par jour de semaine.

    Utilise le snapshot **brut** (avant nettoyage) parce que ce graphe sert
    à détecter les trous de collecte — les filtres du nettoyage masqueraient
    des anomalies.
    """
    daily = df_raw.groupby(df_raw["datetime"].dt.date).size().reset_index(
        name="count"
    )
    daily.columns = ["date", "count"]

    dow_counts = df_raw.groupby(df_raw["datetime"].dt.dayofweek).size()

    n_days = (df_raw["datetime"].max() - df_raw["datetime"].min()).days
    n_weeks = round(n_days / 7, 1)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f"Relevés par jour calendaire — {n_days} jours ({n_weeks} semaines)",
            "Relevés par jour de semaine",
        ),
        column_widths=[0.65, 0.35],
        horizontal_spacing=0.10,
    )

    # Col 1 : aire temporelle
    fig.add_trace(
        go.Scatter(
            x=daily["date"], y=daily["count"],
            mode="lines",
            fill="tozeroy",
            line=dict(color=COLORS["info"], width=1.5),
            fillcolor="rgba(25, 118, 210, 0.25)",
            name="Relevés/jour",
            hovertemplate="<b>%{x|%a %d %b}</b><br>%{y:,} relevés<extra></extra>",
        ),
        row=1, col=1,
    )

    # Col 2 : barres par jour de semaine
    fig.add_trace(
        go.Bar(
            x=DAY_FR,
            y=dow_counts.values,
            marker=dict(color=COLORS["dow"], line=dict(color="white", width=1)),
            text=[f"{v:,}" for v in dow_counts.values],
            textposition="outside",
            textfont=dict(size=10),
            name="Relevés/jour de semaine",
            hovertemplate="<b>%{x}</b><br>%{y:,} relevés<extra></extra>",
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Nb relevés", row=1, col=1, tickformat=",")
    fig.update_xaxes(title_text="Jour", row=1, col=2)
    fig.update_yaxes(title_text="Nb relevés", row=1, col=2, tickformat=",")

    return _apply_layout(fig, "1. Couverture temporelle de la collecte", height=420)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE 2 — DISTRIBUTION DU TAUX DE REMPLISSAGE
# ─────────────────────────────────────────────────────────────────────────────
def render_fill_rate_distribution(df: pd.DataFrame) -> go.Figure:
    """Graphe 2 : histogramme global + boxplot top 20 stations.

    Boxplot fait via groupby (au lieu de la liste compréhension O(n²)
    de l'ancien code).
    """
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

    # Col 1 : histogramme
    fig.add_trace(
        go.Histogram(
            x=df["taux"],
            nbinsx=50,
            marker=dict(color=COLORS["info"], line=dict(color="white", width=0.3)),
            name="Distribution",
            hovertemplate="Bin: %{x}%<br>Fréquence: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    # Lignes verticales moyenne/médiane
    fig.add_vline(
        x=mean_val, line=dict(color=COLORS["danger"], dash="dash", width=2),
        annotation_text=f"Moyenne {mean_val:.1f}%", annotation_position="top",
        row=1, col=1,
    )
    fig.add_vline(
        x=median_val, line=dict(color=COLORS["secondary"], dash="dot", width=2),
        annotation_text=f"Médiane {median_val:.1f}%",
        annotation_position="top right",
        row=1, col=1,
    )

    # Col 2 : boxplot top 20 stations (par volume de relevés)
    top_stations = df["station_id"].value_counts().head(20).index.tolist()
    sample = df.loc[df["station_id"].isin(top_stations)].copy()
    # Ordonner par volume décroissant pour la lisibilité
    sample["station_id"] = pd.Categorical(
        sample["station_id"], categories=top_stations, ordered=True
    )

    for sid in top_stations:
        vals = sample.loc[sample["station_id"] == sid, "taux"]
        # Récupère le nom pour le hover (1 fois par station, pas par point)
        name = df.loc[df["station_id"] == sid, "name"].iloc[0]
        fig.add_trace(
            go.Box(
                y=vals,
                name=str(sid),
                marker=dict(color=COLORS["primary"]),
                line=dict(width=1),
                boxmean=True,
                hovertemplate=(
                    f"<b>{name}</b> (id={sid})<br>"
                    "Taux: %{y:.1f}%<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Taux (%)", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=1, tickformat=",")
    fig.update_xaxes(title_text="Station ID (top 20 par volume)", row=1, col=2,
                     tickangle=45)
    fig.update_yaxes(title_text="Taux (%)", row=1, col=2)

    return _apply_layout(fig, "2. Distribution du taux de remplissage", height=480)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE 3 — PROFILS TEMPORELS
# ─────────────────────────────────────────────────────────────────────────────
def render_temporal_patterns(df: pd.DataFrame) -> go.Figure:
    """Graphe 3 : taux moyen par heure + taux moyen par jour de semaine."""
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

    # Col 1 : courbe par heure
    fig.add_trace(
        go.Scatter(
            x=hourly.index, y=hourly.values,
            mode="lines+markers",
            line=dict(color=COLORS["info"], width=2.5),
            marker=dict(size=7, color=COLORS["info"]),
            name="Taux moyen",
            hovertemplate="<b>%{x}h</b><br>Taux moyen: %{y:.2f}%<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    # Zones de pointe matin/soir
    for h_start, h_end, label in [(7, 9, "Pointe matin"), (17, 19, "Pointe soir")]:
        fig.add_vrect(
            x0=h_start, x1=h_end,
            fillcolor=COLORS["primary"], opacity=0.12, line_width=0,
            annotation_text=label, annotation_position="top",
            annotation=dict(font=dict(size=10, color=COLORS["primary"])),
            row=1, col=1,
        )

    # Col 2 : barres par jour
    fig.add_trace(
        go.Bar(
            x=DAY_FR, y=daily.values,
            marker=dict(color=COLORS["dow"], line=dict(color="white", width=1)),
            text=[f"{v:.1f}%" for v in daily.values],
            textposition="outside",
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


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE 4 — IMPACT MÉTÉO
# ─────────────────────────────────────────────────────────────────────────────
def render_weather_impact(df: pd.DataFrame) -> go.Figure:
    """Graphe 4 : répartition des conditions météo + taux moyen par condition.

    Calcule la sévérité météo via le helper partagé (cohérent avec
    build_features.py qui utilisera le même).
    """
    weather_features = apply_weather_severity(df["weather_code"])
    df_local = df.assign(weather_severity=weather_features["weather_severity"])

    sev_counts = df_local["weather_severity"].value_counts().sort_index()
    mean_by_sev = df_local.groupby("weather_severity")["taux"].mean()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Répartition des conditions météo",
            "Taux moyen par condition",
        ),
        horizontal_spacing=0.10,
    )

    # Col 1 : répartition
    sev_idx = sev_counts.index.tolist()
    fig.add_trace(
        go.Bar(
            x=[SEV_LABELS[i] for i in sev_idx],
            y=sev_counts.values,
            marker=dict(
                color=[COLORS["weather_sev"][i] for i in sev_idx],
                line=dict(color="white", width=1),
            ),
            text=[f"{v:,}" for v in sev_counts.values],
            textposition="outside",
            hovertemplate="<b>%{x}</b><br>%{y:,} relevés<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )

    # Col 2 : taux moyen par condition
    sev2 = mean_by_sev.index.tolist()
    fig.add_trace(
        go.Bar(
            x=[SEV_LABELS[i] for i in sev2],
            y=mean_by_sev.values,
            marker=dict(
                color=[COLORS["weather_sev"][i] for i in sev2],
                line=dict(color="white", width=1),
            ),
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


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE 5 — ANOMALIE THERMIQUE
# ─────────────────────────────────────────────────────────────────────────────
def render_temp_anomaly(df: pd.DataFrame) -> go.Figure:
    """Graphe 5 : distribution de l'anomalie thermique + scatter anomalie vs taux.

    Anomalie = température ressentie - moyenne mensuelle.
    Permet de détecter si un jour anormalement chaud/froid affecte l'usage.
    """
    df_local = df.copy()
    df_local["month"] = df_local["datetime"].dt.month
    monthly_mean = df_local.groupby("month")["apparent_temperature"].transform("mean")
    df_local["temp_anom"] = df_local["apparent_temperature"] - monthly_mean

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            "Distribution de l'anomalie thermique",
            "Anomalie thermique vs taux de remplissage (échantillon 8000 pts)",
        ),
        horizontal_spacing=0.10,
    )

    # Col 1 : histogramme anomalie
    fig.add_trace(
        go.Histogram(
            x=df_local["temp_anom"],
            nbinsx=60,
            marker=dict(color=COLORS["info"], line=dict(color="white", width=0.3)),
            hovertemplate="Anomalie: %{x:.1f}°C<br>Fréquence: %{y:,}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    fig.add_vline(
        x=0, line=dict(color=COLORS["danger"], dash="dash", width=2),
        annotation_text="0 = normale du mois", annotation_position="top",
        row=1, col=1,
    )

    # Col 2 : scatter (échantillonné car 5M points en HTML serait trop lourd)
    sample = df_local.sample(min(8000, len(df_local)), random_state=42)
    fig.add_trace(
        go.Scattergl(  # Scattergl = WebGL, beaucoup plus rapide qu'un Scatter classique
            x=sample["temp_anom"],
            y=sample["taux"],
            mode="markers",
            marker=dict(
                size=4, color=COLORS["info"], opacity=0.25,
                line=dict(width=0),
            ),
            hovertemplate=(
                "Anomalie: %{x:.1f}°C<br>"
                "Taux: %{y:.1f}%<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1, col=2,
    )

    fig.update_xaxes(title_text="Écart à la normale (°C)", row=1, col=1)
    fig.update_yaxes(title_text="Fréquence", row=1, col=1, tickformat=",")
    fig.update_xaxes(title_text="Anomalie (°C)", row=1, col=2)
    fig.update_yaxes(title_text="Taux (%)", row=1, col=2)

    return _apply_layout(fig, "5. Anomalie thermique", height=440)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE 6 — PROFILS FONCTIONNELS DES STATIONS
# ─────────────────────────────────────────────────────────────────────────────
def render_station_profiles(df: pd.DataFrame) -> go.Figure:
    """Graphe 6 : distribution du ratio matin/soir + camembert des profils.

    Résidentiel (ratio > 1.2) : se vide le matin (plein la nuit)
    Bureaux (ratio < 0.8)     : se remplit le matin (vide la nuit)
    Mixte (entre 0.8 et 1.2)  : équilibré
    """
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
        specs=[[{"type": "xy"}, {"type": "domain"}]],   # domain pour le pie
        horizontal_spacing=0.10,
    )

    # Col 1 : histogramme du ratio
    fig.add_trace(
        go.Histogram(
            x=station_ratio.values,
            nbinsx=40,
            marker=dict(color=COLORS["info"], line=dict(color="white", width=0.3)),
            hovertemplate="Ratio: %{x:.2f}<br>Stations: %{y}<extra></extra>",
            showlegend=False,
        ),
        row=1, col=1,
    )
    # Lignes seuils
    fig.add_vline(
        x=1.2, line=dict(color=COLORS["success"], dash="dash", width=2),
        annotation_text=f"Résidentiel >1.2 ({n_res})",
        annotation_position="top right",
        row=1, col=1,
    )
    fig.add_vline(
        x=0.8, line=dict(color=COLORS["danger"], dash="dash", width=2),
        annotation_text=f"Bureaux <0.8 ({n_bur})",
        annotation_position="top left",
        row=1, col=1,
    )
    fig.add_vline(
        x=1.0, line=dict(color=COLORS["neutral"], dash="dot", width=1),
        row=1, col=1,
    )

    # Col 2 : camembert
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
                    "<b>%{label}</b><br>"
                    "%{value} stations (%{percent})<extra></extra>"
                ),
                showlegend=False,
            ),
            row=1, col=2,
        )

    fig.update_xaxes(title_text="Ratio matin / soir", row=1, col=1)
    fig.update_yaxes(title_text="Nb stations", row=1, col=1)

    return _apply_layout(fig, "6. Profils fonctionnels des stations", height=460)


# ─────────────────────────────────────────────────────────────────────────────
# GRAPHE 7 — IMPACT VACANCES SCOLAIRES
# ─────────────────────────────────────────────────────────────────────────────
def render_vacation_impact(df: pd.DataFrame) -> go.Figure:
    """Graphe 7 : taux moyen vacances scolaires vs hors vacances."""
    vac_mean = df.groupby(df["is_vacation"].astype(bool))["taux"].mean()

    if len(vac_mean) < 2:
        # Cas où on n'a que vacances ou que hors-vacances dans la fenêtre
        fig = go.Figure()
        fig.add_annotation(
            text="Données insuffisantes (la période ne contient que vacances ou que hors-vacances)",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color=COLORS["neutral"]),
        )
        return _apply_layout(fig, "7. Impact des vacances scolaires", height=380)

    labels = ["Hors vacances", "Vacances scolaires"]
    values = [
        float(vac_mean.get(False, np.nan)),
        float(vac_mean.get(True, np.nan)),
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels, y=values,
            marker=dict(
                color=[COLORS["info"], COLORS["secondary"]],
                line=dict(color="white", width=1),
            ),
            text=[f"{v:.1f}%" for v in values],
            textposition="outside",
            textfont=dict(size=14),
            width=0.4,
            hovertemplate="<b>%{x}</b><br>Taux moyen: %{y:.2f}%<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text="Période")
    fig.update_yaxes(title_text="Taux moyen (%)")

    return _apply_layout(fig, "7. Impact des vacances scolaires", height=400)


# ─────────────────────────────────────────────────────────────────────────────
# RAPPORT HTML UNIFIÉ AVEC NAVIGATION PAR ONGLETS
# ─────────────────────────────────────────────────────────────────────────────
def _build_unified_report(
    figures: list[tuple[str, go.Figure]],
    metadata: dict,
) -> str:
    """Construit le HTML unifié avec navigation par onglets.

    Args:
        figures: liste de (titre_onglet, figure plotly)
        metadata: dict avec infos du dataset (rows, stations, période, etc.)

    Returns:
        HTML complet en string, prêt à être écrit sur disque.
    """
    # Convertir chaque figure en div HTML autonome (sans HTML wrapper)
    # On ne charge plotly.js qu'une fois, dans le 1er div
    figs_html = []
    for i, (title, fig) in enumerate(figures):
        include_js = "cdn" if i == 0 else False
        div = fig.to_html(
            include_plotlyjs=include_js,
            full_html=False,
            div_id=f"plot-{i}",
            config={"displayModeBar": True, "displaylogo": False},
        )
        figs_html.append((title, div))

    # CSS de la page (charte Vélib')
    css = """
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: 'Arial', sans-serif;
            margin: 0;
            padding: 0;
            background: #FAFAFA;
            color: #212121;
        }
        header {
            background: linear-gradient(90deg, #2E7D32, #388E3C);
            color: white;
            padding: 20px 40px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        header h1 {
            margin: 0;
            font-size: 24px;
            font-weight: 600;
        }
        header .subtitle {
            margin-top: 4px;
            font-size: 13px;
            opacity: 0.9;
        }
        .metadata {
            display: flex;
            gap: 24px;
            padding: 12px 40px;
            background: white;
            border-bottom: 1px solid #E0E0E0;
            font-size: 13px;
            flex-wrap: wrap;
        }
        .metadata .item {
            color: #616161;
        }
        .metadata .item strong {
            color: #212121;
        }
        .tabs {
            display: flex;
            gap: 4px;
            padding: 0 40px;
            background: white;
            border-bottom: 2px solid #E0E0E0;
            overflow-x: auto;
            white-space: nowrap;
        }
        .tab {
            padding: 12px 16px;
            cursor: pointer;
            border-bottom: 3px solid transparent;
            transition: all 0.2s;
            font-size: 13px;
            font-weight: 500;
            color: #616161;
            user-select: none;
        }
        .tab:hover { color: #2E7D32; background: #F1F8E9; }
        .tab.active {
            color: #2E7D32;
            border-bottom-color: #2E7D32;
            background: #F1F8E9;
        }
        .tab-content {
            display: none;
            padding: 24px 40px;
        }
        .tab-content.active { display: block; }
        footer {
            margin-top: 24px;
            padding: 16px 40px;
            border-top: 1px solid #E0E0E0;
            font-size: 12px;
            color: #9E9E9E;
            text-align: center;
        }
    </style>
    """

    # JS pour la navigation par onglets (vanilla, pas de framework)
    js = """
    <script>
        document.addEventListener('DOMContentLoaded', function () {
            var tabs = document.querySelectorAll('.tab');
            var contents = document.querySelectorAll('.tab-content');
            tabs.forEach(function (tab) {
                tab.addEventListener('click', function () {
                    var idx = this.getAttribute('data-tab');
                    tabs.forEach(function (t) { t.classList.remove('active'); });
                    contents.forEach(function (c) { c.classList.remove('active'); });
                    this.classList.add('active');
                    document.getElementById('content-' + idx).classList.add('active');
                    // Force Plotly à recalculer la taille (sinon les onglets cachés apparaissent rétrécis)
                    if (window.Plotly) {
                        var plot = document.querySelector('#content-' + idx + ' [id^="plot-"]');
                        if (plot) { window.Plotly.Plots.resize(plot); }
                    }
                });
            });
        });
    </script>
    """

    # Onglets
    tabs_html = "".join(
        f'<div class="tab {"active" if i == 0 else ""}" data-tab="{i}">'
        f'{title}</div>'
        for i, (title, _) in enumerate(figs_html)
    )

    # Contenus
    contents_html = "".join(
        f'<div class="tab-content {"active" if i == 0 else ""}" id="content-{i}">'
        f'{html_div}</div>'
        for i, (_, html_div) in enumerate(figs_html)
    )

    # Métadonnées
    metadata_html = "".join(
        f'<div class="item"><strong>{k}</strong>: {v}</div>'
        for k, v in metadata.items()
    )

    # Assemblage
    html = f"""<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <title>Vélib' MLOps — Rapport DataViz</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {css}
</head>
<body>
    <header>
        <h1>🚲 Vélib' MLOps — Rapport d'exploration visuelle</h1>
        <div class="subtitle">Pipeline de prédiction du taux de remplissage des stations</div>
    </header>
    <div class="metadata">
        {metadata_html}
    </div>
    <nav class="tabs">
        {tabs_html}
    </nav>
    <main>
        {contents_html}
    </main>
    <footer>
        Généré automatiquement par <code>ml/src/visualization/dataviz.py</code>
        — Plotly {go.__version__ if hasattr(go, "__version__") else ""}
    </footer>
    {js}
</body>
</html>
"""
    return html


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLIQUE — ORCHESTRATEUR
# ─────────────────────────────────────────────────────────────────────────────
def run_dataviz(write_to_disk: bool = True) -> Path:
    """Lit les parquets, génère les 7 figures, écrit le rapport HTML unifié.

    Args:
        write_to_disk: si True (défaut), écrit le HTML dans
            ``settings.plots_dir / 'dataviz_report.html'``.

    Returns:
        Chemin du fichier HTML produit.

    Raises:
        FileNotFoundError: si raw ou cleaned parquet n'existent pas.
    """
    settings.ensure_directories()

    # ── Lecture des deux parquets (raw + cleaned) ────────────────────────
    raw_path = settings.raw_snapshot_path
    cleaned_path = settings.interim_path

    if not raw_path.exists():
        raise FileNotFoundError(
            f"Snapshot brut absent : {raw_path}. "
            "Lancer d'abord : python -m ml.src.data.load_from_hf"
        )
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"Parquet nettoyé absent : {cleaned_path}. "
            "Lancer d'abord : python -m ml.src.data.make_dataset"
        )

    logger.info(
        "Démarrage run_dataviz",
        extra={"raw_path": str(raw_path), "cleaned_path": str(cleaned_path)},
    )

    df_raw = pd.read_parquet(raw_path)
    df_clean = pd.read_parquet(cleaned_path)

    logger.info(
        "Parquets chargés",
        extra={
            "raw_rows": len(df_raw),
            "cleaned_rows": len(df_clean),
            "stations": int(df_clean["station_id"].nunique()),
        },
    )

    # ── Génération des 7 figures ──────────────────────────────────────────
    figures: list[tuple[str, go.Figure]] = []

    for label, render_fn, df_arg in [
        ("1. Couverture",   render_temporal_coverage,    df_raw),
        ("2. Distribution", render_fill_rate_distribution, df_clean),
        ("3. Profils temp.", render_temporal_patterns,    df_clean),
        ("4. Météo",        render_weather_impact,        df_clean),
        ("5. Anomalie temp.", render_temp_anomaly,        df_clean),
        ("6. Profils stations", render_station_profiles,  df_clean),
        ("7. Vacances",     render_vacation_impact,        df_clean),
    ]:
        try:
            logger.info(f"Génération {label}", extra={"graph": label})
            fig = render_fn(df_arg)
            figures.append((label, fig))
        except Exception as e:  # noqa: BLE001
            logger.exception(
                f"Échec génération {label}",
                extra={"graph": label, "error_type": type(e).__name__},
            )
            # On continue avec les autres graphes — un échec ne doit pas
            # tout bloquer
            continue

    if not figures:
        raise RuntimeError("Aucun graphe n'a pu être généré, abort.")

    # ── Construction du rapport unifié ────────────────────────────────────
    metadata = {
        "Période":       f"{df_raw['datetime'].min().date()} → {df_raw['datetime'].max().date()}",
        "Relevés bruts": f"{len(df_raw):,}",
        "Relevés nettoyés": f"{len(df_clean):,}",
        "Stations":      f"{int(df_clean['station_id'].nunique()):,}",
        "Taux moyen":    f"{df_clean['taux'].mean():.1f}%",
        "Taux médian":   f"{df_clean['taux'].median():.1f}%",
    }

    html = _build_unified_report(figures, metadata)

    # ── Écriture ──────────────────────────────────────────────────────────
    if write_to_disk:
        out = settings.plots_dir / "dataviz_report.html"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(html, encoding="utf-8")
        logger.info(
            "Rapport HTML écrit",
            extra={
                "path": str(out),
                "size_mb": round(out.stat().st_size / 1e6, 2),
                "n_figures": len(figures),
            },
        )
        return out

    return Path()  # placeholder si write_to_disk=False


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def main() -> int:
    """Point d'entrée CLI. Retourne le code de sortie pour le shell."""
    try:
        run_dataviz(write_to_disk=True)
        return 0
    except FileNotFoundError as e:
        logger.error("Parquet absent", extra={"error": str(e)})
        return 2
    except Exception as e:  # noqa: BLE001
        logger.exception("Échec run_dataviz", extra={"error_type": type(e).__name__})
        return 1


if __name__ == "__main__":
    sys.exit(main())
