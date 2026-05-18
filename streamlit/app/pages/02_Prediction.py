import sys
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st
import plotly.graph_objects as go
from utils import api_client
from utils.stations import STATIONS_BY_NAME, STATION_NAMES, build_features

st.set_page_config(page_title="Prédiction · Vélib' MLOps", page_icon="🔮", layout="wide")

st.markdown("## 🔮 Prédiction — MVP métier")
st.caption(
    "Sélectionnez une station, définissez le contexte temporel et météo, "
    "et obtenez la prédiction de taux de remplissage via l'API FastAPI."
)
st.divider()

# ── ALERTE_LEVEL helpers ──────────────────────────────────────────────────────
ALERT_CONFIG = {
    "green":  {"label": "Équilibrée",      "color": "#5bac3a", "bg": "#d4edda", "icon": "🟢"},
    "yellow": {"label": "Tension modérée", "color": "#f6a623", "bg": "#fff3cd", "icon": "🟡"},
    "red":    {"label": "Critique",        "color": "#e53e3e", "bg": "#f8d7da", "icon": "🔴"},
}

DOW_LABELS = ["Lundi", "Mardi", "Mercredi", "Jeudi", "Vendredi", "Samedi", "Dimanche"]
WEATHER_LABELS = {
    0: "0 — Dégagé",
    1: "1 — Nuageux",
    2: "2 — Pluie légère",
    3: "3 — Pluie forte",
    4: "4 — Orage / neige",
}


def make_gauge(taux: float, alert: str) -> go.Figure:
    color = ALERT_CONFIG[alert]["color"]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(taux, 1),
        number={"suffix": " %", "font": {"size": 40, "color": color}},
        title={"text": "Taux de remplissage prédit", "font": {"size": 15, "color": "#718096"}},
        domain={"x": [0, 1], "y": [0, 1]},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#cbd5e0"},
            "bar": {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 10],  "color": "#ffcdd2"},
                {"range": [10, 30], "color": "#ffe0b2"},
                {"range": [30, 70], "color": "#c8e6c9"},
                {"range": [70, 90], "color": "#ffe0b2"},
                {"range": [90, 100],"color": "#ffcdd2"},
            ],
        },
    ))
    fig.update_layout(
        margin=dict(t=60, b=20, l=20, r=20),
        height=280,
        paper_bgcolor="#f7f8fa",
        font={"family": "sans-serif"},
    )
    return fig


# ── Layout ────────────────────────────────────────────────────────────────────
left_col, right_col = st.columns([1, 1.3], gap="large")

# ──── Inputs ──────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("### Paramètres")

    station_name = st.selectbox(
        "Station Vélib'",
        options=STATION_NAMES,
        index=0,
        help="15 stations représentatives de Paris",
    )
    station = STATIONS_BY_NAME[station_name]

    st.caption(
        f"📍 Lat {station['lat']:.4f} · Lon {station['lon']:.4f} · "
        f"{station['capacity']} places · tendance historique ≈ {station['station_trend_avg']:.0f} %"
    )

    st.divider()
    st.markdown("**Contexte temporel**")

    now = datetime.now(timezone.utc)
    use_now = st.checkbox("Utiliser l'heure actuelle (UTC)", value=True)
    if use_now:
        hour = now.hour
        dow = now.weekday()
        month = now.month
        st.info(f"Heure : {hour:02d}h · {DOW_LABELS[dow]} · mois {month}")
    else:
        c1, c2, c3 = st.columns(3)
        hour = c1.number_input("Heure (0–23)", 0, 23, 8, step=1)
        dow = c2.selectbox("Jour", range(7), format_func=lambda i: DOW_LABELS[i])
        month = c3.number_input("Mois", 1, 12, now.month, step=1)

    st.divider()
    st.markdown("**Contexte météo**")

    temperature = st.slider("Température ressentie (°C)", -10.0, 40.0, 16.0, 0.5)
    temp_anomalie = st.slider("Anomalie thermique (°C vs normale)", -8.0, 8.0, 0.0, 0.5)
    weather_severity = st.select_slider(
        "Sévérité météo",
        options=[0, 1, 2, 3, 4],
        value=0,
        format_func=lambda v: WEATHER_LABELS[v],
    )
    is_frozen = int(temperature <= 2)
    is_stormy = int(weather_severity >= 4)
    if is_frozen:
        st.warning("❄️ Gel détecté (température ≤ 2 °C) → is_frozen = 1")
    if is_stormy:
        st.warning("⛈️ Orage détecté (sévérité 4) → is_stormy = 1")

    st.divider()
    st.markdown("**Occupation récente**")
    st.caption("Ces lags représentent le taux observé il y a 60 min et 240 min.")
    lag_60 = st.slider("Taux il y a 60 min (%)", 0.0, 100.0, station["station_trend_avg"], 1.0)
    lag_240 = st.slider("Taux il y a 240 min (%)", 0.0, 100.0, station["station_trend_avg"], 1.0)

    st.divider()
    st.markdown("**Calendaire**")
    c1, c2 = st.columns(2)
    is_holiday = int(c1.checkbox("Jour férié"))
    is_vacation = int(c2.checkbox("Vacances scolaires"))

    st.divider()
    predict_btn = st.button("🔮 Prédire", type="primary", use_container_width=True)


# ──── Résultats ────────────────────────────────────────────────────────────────
with right_col:
    st.markdown("### Résultat")

    if predict_btn:
        features = build_features(
            station=station,
            hour=int(hour), dow=int(dow), month=int(month),
            temperature=float(temperature),
            temp_anomalie=float(temp_anomalie),
            weather_severity=int(weather_severity),
            is_frozen=is_frozen, is_stormy=is_stormy,
            lag_60min=float(lag_60), lag_240min=float(lag_240),
            is_holiday=is_holiday, is_vacation=is_vacation,
        )

        with st.spinner("Appel POST /predict…"):
            code, body = api_client.api_predict(features)

        if code == 200 and isinstance(body, dict):
            taux = body["taux_predicted"]
            residual = body["residual_predicted"]
            alert = body["alert_level"]
            cfg = ALERT_CONFIG[alert]

            # Gauge
            st.plotly_chart(make_gauge(taux, alert), use_container_width=True)

            # Badge alerte
            st.markdown(
                f'<div style="background:{cfg["bg"]};border-radius:10px;padding:1rem 1.5rem;'
                f'text-align:center;margin-bottom:1rem;">'
                f'<div style="font-size:2rem">{cfg["icon"]}</div>'
                f'<div style="font-weight:800;font-size:1.1rem;color:{cfg["color"]}">'
                f'Station {cfg["label"].upper()}</div>'
                f'<div style="color:#718096;font-size:0.85rem;margin-top:0.3rem">'
                f'Niveau d\'alerte opérationnel : <code>{alert}</code></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Métriques détaillées
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("Taux prédit", f"{taux:.1f} %")
            mc2.metric("Résidu prédit", f"{residual:+.1f} pp")
            mc3.metric("Tendance station", f"{station['station_trend_avg']:.0f} %")

            st.caption(
                f"taux\_prédit = résidu\_prédit ({residual:+.2f}) + tendance\_station ({station['station_trend_avg']:.0f}) = {taux:.1f} %"
            )

            with st.expander("Payload envoyé à l'API", expanded=False):
                st.json(features)

        elif code == 422:
            st.error(f"❌ HTTP 422 — features invalides")
            st.json(body)
        elif code == 0:
            st.error("❌ API inaccessible — vérifiez `make health`")
        else:
            st.error(f"❌ HTTP {code}")
            st.write(body)

    else:
        # État initial
        st.markdown("""
<div style="background:#f0f8eb;border-radius:12px;padding:2rem;text-align:center;margin-top:2rem;">
  <div style="font-size:3rem;margin-bottom:1rem">🚲</div>
  <div style="font-weight:700;color:#2e7d32;font-size:1.1rem">Prêt pour la prédiction</div>
  <div style="color:#718096;margin-top:0.5rem;font-size:0.9rem">
    Configurez les paramètres à gauche<br>et cliquez sur <b>Prédire</b>
  </div>
</div>
""", unsafe_allow_html=True)

        st.divider()
        st.markdown("#### Niveaux d'alerte")
        for level, cfg in ALERT_CONFIG.items():
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.75rem;'
                f'padding:0.5rem 0.75rem;border-radius:8px;background:{cfg["bg"]};margin:0.3rem 0;">'
                f'<span style="font-size:1.3rem">{cfg["icon"]}</span>'
                f'<div><b style="color:{cfg["color"]}">{level.upper()}</b> — {cfg["label"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.caption("""
- 🟢 **green** : 30 % ≤ taux ≤ 70 %
- 🟡 **yellow** : 10–30 % ou 70–90 %
- 🔴 **red** : < 10 % ou > 90 %
""")
