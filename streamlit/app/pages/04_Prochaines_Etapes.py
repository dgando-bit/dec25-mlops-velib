import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import streamlit as st

st.set_page_config(
    page_title="Les prochaines étapes · Vélib' MLOps",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  section[data-testid="stSidebar"] { background: #f0f8eb; }
  .next-header {
    background: linear-gradient(135deg, #0d47a1 0%, #1565c0 60%, #1976d2 100%);
    border-radius: 12px;
    padding: 1.8rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
  }
  .next-header h1 { margin: 0 0 0.4rem 0; font-size: 1.8rem; }
  .next-header p  { margin: 0; opacity: 0.9; font-size: 0.95rem; }
  .card {
    background: #fff;
    border-radius: 10px;
    padding: 1.2rem 1.4rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 1rem;
    border-top: 3px solid #1976d2;
    height: 100%;
  }
  .card-green  { border-top-color: #2e7d32; }
  .card-orange { border-top-color: #e65100; }
  .card-purple { border-top-color: #6a1b9a; }
  .card-blue   { border-top-color: #0d47a1; }
  .card h4 { margin: 0 0 0.6rem 0; font-size: 1rem; color: #212121; }
  .card ul { margin: 0; padding-left: 1.2rem; color: #444; font-size: 0.9rem; line-height: 1.7; }
  .limit-card {
    background: #fff8e1;
    border-left: 4px solid #f9a825;
    border-radius: 8px;
    padding: 0.9rem 1.2rem;
    margin-bottom: 0.75rem;
    font-size: 0.9rem;
  }
  .limit-card strong { color: #5d4037; }
  .section-title {
    color: #1565c0;
    font-size: 1.05rem;
    font-weight: 700;
    border-bottom: 2px solid #bbdefb;
    padding-bottom: 0.3rem;
    margin: 1.6rem 0 1rem 0;
  }
  .horizon-badge {
    display: inline-block;
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.15rem 0.55rem;
    border-radius: 10px;
    margin-right: 0.4rem;
    vertical-align: middle;
  }
  .h-court  { background: #c8e6c9; color: #1b5e20; }
  .h-moyen  { background: #bbdefb; color: #0d47a1; }
  .h-long   { background: #e1bee7; color: #4a148c; }
</style>
""", unsafe_allow_html=True)

# ── En-tête ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="next-header">
  <h1>Les prochaines étapes</h1>
  <p>
    Limitations actuelles identifiées · Axes d'amélioration prioritaires ·
    Trajectoire vers un déploiement production
  </p>
</div>
""", unsafe_allow_html=True)

# ── Limitations actuelles ─────────────────────────────────────────────────────
st.markdown('<div class="section-title">Limitations actuelles</div>', unsafe_allow_html=True)

limitations = [
    (
        "Prédiction instantanée uniquement",
        "Le modèle prédit le taux au moment de la requête, pas en avance. "
        "Un usager qui planifie son trajet dans 30 min ne peut pas l'utiliser directement."
    ),
    (
        "Météo simulée dans la démo",
        "Les paramètres météo (température, sévérité) sont saisis manuellement. "
        "En production, ils devraient être injectés automatiquement depuis une API météo (OpenMeteo, Météo-France)."
    ),
    (
        "Modèle global (non par station)",
        "Un seul XGBoost couvre les 1 492 stations. Des stations atypiques (touristiques, multimodales) "
        "pourraient bénéficier de modèles dédiés ou de fine-tuning."
    ),
    (
        "Drift informatif uniquement",
        "La détection Evidently (train vs test) mesure un drift saisonnier structurel, pas la dérive en production. "
        "Elle n'est pas encore câblée pour déclencher un réentraînement d'urgence."
    ),
    (
        "Pas d'authentification sur l'API",
        "Les endpoints sont ouverts sur le réseau Docker interne. "
        "Un déploiement public nécessiterait au minimum une clé API ou des tokens JWT."
    ),
    (
        "Collecte dépendante de cron-job.org",
        "Le scraping des données Vélib' passe par un service tiers (cron-job.org → HuggingFace Space). "
        "Une panne ou un changement de plan suffit à interrompre la collecte."
    ),
]

for title, desc in limitations:
    st.markdown(
        f'<div class="limit-card"><strong>{title}</strong><br>{desc}</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Roadmap par horizon ───────────────────────────────────────────────────────
st.markdown('<div class="section-title">Roadmap par horizon</div>', unsafe_allow_html=True)

st.markdown("""
<span class="horizon-badge h-court">Court terme — 1 à 3 mois</span>
<span class="horizon-badge h-moyen">Moyen terme — 3 à 6 mois</span>
<span class="horizon-badge h-long">Long terme — 6 mois+</span>
""", unsafe_allow_html=True)

st.markdown("")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
<div class="card card-green">
<h4>Modèle & Pipeline</h4>
<ul>
  <li><span class="horizon-badge h-court">Court</span> Intégration météo temps réel (OpenMeteo API) dans la feature pipeline</li>
  <li><span class="horizon-badge h-court">Court</span> Forecast multi-horizon : prédiction à +15 min, +30 min, +1 h</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Modèle par cluster de stations (k-means sur profils fonctionnels)</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Features événementielles (concerts, matches, manifestations) via OpenAgenda API</li>
  <li><span class="horizon-badge h-long">Long</span> Online learning — mise à jour incrémentale du modèle sans réentraînement complet</li>
  <li><span class="horizon-badge h-long">Long</span> Ensemble XGBoost + LSTM pour capturer les dépendances temporelles longues</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card card-orange">
<h4>Qualité & Observabilité</h4>
<ul>
  <li><span class="horizon-badge h-court">Court</span> Drift production réel : comparaison prédictions vs données réelles J-1</li>
  <li><span class="horizon-badge h-court">Court</span> Alertes Grafana sur dérive R² ou MAE (pas seulement CPU/RAM)</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Logs centralisés avec Loki + Grafana (remplacement des logs fichiers)</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Traces distribuées OpenTelemetry (latence end-to-end par requête)</li>
  <li><span class="horizon-badge h-long">Long</span> A/B testing automatisé de modèles avec traffic splitting dans Nginx</li>
</ul>
</div>
""", unsafe_allow_html=True)

with col2:
    st.markdown("""
<div class="card card-blue">
<h4>Infrastructure & Déploiement</h4>
<ul>
  <li><span class="horizon-badge h-court">Court</span> CI/CD GitHub Actions : lint, tests, build image, push registry</li>
  <li><span class="horizon-badge h-court">Court</span> Authentification JWT sur l'API FastAPI (endpoint /token)</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Migration vers Kubernetes (GKE ou AKS) avec HPA sur le service api</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Collecte autonome : scraping Vélib' API officielle sans dépendance cron-job.org</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Object storage (GCS/S3) pour remplacer le bind mount DVC DagsHub</li>
  <li><span class="horizon-badge h-long">Long</span> Multi-tenancy : déploiement multi-villes (Lyon Vélo'v, Bordeaux Vcub)</li>
</ul>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div class="card card-purple">
<h4>Produit & UX</h4>
<ul>
  <li><span class="horizon-badge h-moyen">Moyen</span> Application mobile PWA avec carte interactive des stations (statut temps réel)</li>
  <li><span class="horizon-badge h-moyen">Moyen</span> Recommandation de station alternative la plus proche disponible</li>
  <li><span class="horizon-badge h-long">Long</span> Alertes push personnalisées (station favorite en tension dans 15 min)</li>
  <li><span class="horizon-badge h-long">Long</span> Intégration données RATP / IDFM pour itinéraire multimodal</li>
</ul>
</div>
""", unsafe_allow_html=True)

st.divider()

# ── Trajectoire cloud ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Trajectoire vers un déploiement production</div>', unsafe_allow_html=True)

st.markdown("""
L'architecture Docker Compose actuelle est pensée pour la démonstration et le développement local.
Un déploiement production suivrait cette trajectoire :
""")

steps = [
    ("1", "Containerisation CI/CD",
     "GitHub Actions → build images → push vers Artifact Registry (GCP) ou ACR (Azure). "
     "Tests automatisés (unit + intégration) bloquants avant merge.",
     "#e3f2fd"),
    ("2", "Kubernetes (staging)",
     "Déploiement sur GKE ou AKS. Services : api (HPA 2–10 pods), mlflow-server (StatefulSet), "
     "Airflow (helm chart officiel). Secrets via Kubernetes Secrets ou Vault.",
     "#e8f5e9"),
    ("3", "Stockage managé",
     "PostgreSQL → Cloud SQL (GCP) ou Azure Database. "
     "Artefacts MLflow → GCS/S3. Redis → Memorystore/Cache for Redis.",
     "#fff3e0"),
    ("4", "Observabilité cloud-native",
     "Prometheus → Google Managed Prometheus. Grafana → Grafana Cloud ou Grafana OSS sur GKE. "
     "Logs → Cloud Logging / Azure Monitor. Traces → Cloud Trace.",
     "#f3e5f5"),
    ("5", "Production (canary)",
     "Traffic splitting 5%/95% entre nouvelle version et version stable via Nginx Ingress Controller. "
     "Promotion automatique si métriques ML stables sur 24h (drift, MAE).",
     "#e1f5fe"),
]

for num, title, desc, bg in steps:
    st.markdown(
        f'<div style="display:flex;gap:1rem;align-items:flex-start;'
        f'background:{bg};border-radius:10px;padding:1rem 1.2rem;margin-bottom:0.75rem;">'
        f'<div style="font-size:1.5rem;font-weight:800;color:#1565c0;min-width:2rem">{num}</div>'
        f'<div><strong style="font-size:0.95rem">{title}</strong>'
        f'<div style="font-size:0.88rem;color:#444;margin-top:0.25rem">{desc}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.divider()

# ── Réflexions finales ────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Réflexions sur la scalabilité</div>', unsafe_allow_html=True)

col_a, col_b = st.columns(2, gap="large")

with col_a:
    st.markdown("**Points forts architecturaux**")
    st.markdown("""
- **Découplage fort** : chaque service est indépendant, scalable séparément
- **Pipeline reproductible** : DVC garantit la traçabilité des artefacts et la reproductibilité
- **Registry ML** : MLflow Registry + alias `staging` permettent des rollbacks instantanés
- **Modèle dégradé** : l'API répond `/health degraded` sans crasher si le modèle ne charge pas
- **Rate limiting** : Nginx protège l'API contre les abus sans modifier le code applicatif
- **Airflow quality gate** : un modèle sous-performant (R² < 0.75) n'est pas promu automatiquement
""")

with col_b:
    st.markdown("**Points de vigilance à l'échelle**")
    st.markdown("""
- **DVC bind mount** : le répertoire `data/` est partagé entre tous les conteneurs — à remplacer par un object store en multi-nœuds
- **MLflow monolithique** : le serveur MLflow est un SPOF ; en production, prévoir un cluster ou un service managé
- **Airflow LocalExecutor** → CeleryExecutor déjà en place, mais nécessite un Redis managé à l'échelle
- **Modèle en mémoire** : le cache LRU est par instance API — en multi-pods, chaque pod charge son propre modèle (coût réseau × N)
- **Nginx stateless** : les upstreams sont résolus au démarrage — à migrer vers Kubernetes Ingress pour la résilience DNS
""")

st.divider()
st.caption("Vélib' MLOps · DataScientest promotion décembre 2025")
