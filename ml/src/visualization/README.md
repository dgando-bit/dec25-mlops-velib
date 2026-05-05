# `ml/src/visualization` — Graphes d'exploration et de présentation

Ce dossier contient les modules qui produisent les visualisations du projet.

## Module `dataviz.py`

Refactoring de l'ancien `01_dataviz.py` en version Plotly interactive.

**Lancement** :

```bash
# Depuis la racine du repo, avec .venv activé
python -m ml.src.visualization.dataviz
```

**Prérequis** : avoir préalablement lancé `load_from_hf` puis `make_dataset` pour produire les deux parquets nécessaires.

**Sortie** : `data/outputs/plots/dataviz_report.html` — un fichier HTML autonome (~5-50 Mo selon volume de données) avec 7 graphes interactifs en navigation par onglets.

**Ouverture** : double-cliquer sur le fichier ou le glisser dans n'importe quel navigateur. Aucun serveur nécessaire.

## Les 7 graphes produits

| # | Titre | Source | Apport Plotly |
|---|---|---|---|
| 1 | Couverture temporelle | parquet brut | Hover détaillé par jour |
| 2 | Distribution du taux | cleaned | Histogramme zoomable + boxplot par station |
| 3 | Profils temporels | cleaned | Hover heure par heure |
| 4 | Impact météo | cleaned | Catégories WMO interactives |
| 5 | Anomalie thermique | cleaned | WebGL (Scattergl) pour scatter rapide |
| 6 | Profils stations | cleaned | Pie chart + histogramme combiné |
| 7 | Impact vacances | cleaned | Comparaison directe vacances / hors vacances |

## Réutilisation depuis Streamlit (à venir)

Les 7 fonctions `render_*()` sont exportables et utilisables depuis Streamlit :

```python
import streamlit as st
import pandas as pd
from ml.src.visualization.dataviz import render_temporal_patterns

df = pd.read_parquet("data/interim/velib_cleaned_latest.parquet")
fig = render_temporal_patterns(df)
st.plotly_chart(fig, use_container_width=True)
```

## Pipeline DVC

Voir `dvc.yaml` à la racine pour le DAG complet.
