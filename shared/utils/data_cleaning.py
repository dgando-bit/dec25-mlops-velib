"""
shared.utils.data_cleaning — Helpers réutilisables pour le nettoyage Vélib'.

Tous les helpers respectent deux principes :
    1. Pas de mutation in-place : ils retournent soit une nouvelle Series,
       soit un nouveau DataFrame, soit un dict de Series.
    2. Pas de copie inutile du DataFrame entier : pour les helpers qui
       calculent des features, on retourne uniquement les nouvelles colonnes
       (l'orchestrateur décide ensuite via df.assign(**result)).

Ces helpers sont consommés par :
    - ml/src/data/make_dataset.py  : nettoyage minimal (filtres, taux)
    - ml/src/features/build_features.py  : feature engineering météo (à venir)
    - ml/src/visualization/dataviz.py    : graphiques (à venir)
"""
from __future__ import annotations

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES — codes WMO et leurs catégories
# ─────────────────────────────────────────────────────────────────────────────
WMO_GROUPS: dict[str, list[int]] = {
    "Sans precipitation":      list(range(0, 20)),
    "Precipitations recentes": list(range(20, 30)),
    "Tempete":                 list(range(30, 40)),
    "Brouillard":              list(range(40, 50)),
    "Bruine":                  list(range(50, 60)),
    "Pluie":                   list(range(60, 70)),
    "Neige":                   list(range(70, 80)),
    "Averses":                 list(range(80, 90)),
    "Orages":                  list(range(90, 100)),
}

# Labels de l'échelle ordinale weather_severity 0-4
SEVERITY_LABELS: list[str] = [
    "0 Clair",
    "1 Nuageux",
    "2 Pluie legere",
    "3 Pluie forte",
    "4 Orage/neige",
]


# ─────────────────────────────────────────────────────────────────────────────
# CONVERSION DE TYPES
# ─────────────────────────────────────────────────────────────────────────────
def coerce_booleans(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Coerce les colonnes ``cols`` en booléens stricts.

    Gère les cas suivants :
        - bool natifs               → identité
        - chaînes 'true'/'false'    → True/False (case-insensitive)
        - chaînes '1'/'0'           → True/False
        - 1/0 numériques            → True/False
        - autres                    → False (comportement défensif)

    Args:
        df: DataFrame d'entrée (non muté).
        cols: liste des noms de colonnes à convertir.

    Returns:
        Nouveau DataFrame avec les colonnes coercées.
    """
    out = df.copy()
    for col in cols:
        if col not in out.columns:
            continue
        series = out[col]
        if series.dtype == bool:
            continue
        # Stratégie : convertir en string, lowercase, mapper
        as_str = series.astype(str).str.strip().str.lower()
        out[col] = as_str.isin(["true", "1", "1.0", "yes", "y"])
    return out


# ─────────────────────────────────────────────────────────────────────────────
# CALCULS DE CAPACITÉ ET TAUX
# ─────────────────────────────────────────────────────────────────────────────
def compute_total_capacity(df: pd.DataFrame) -> pd.Series:
    """Capacité physiquement disponible = vélos présents + docks libres.

    Cette grandeur peut différer de la colonne ``capacity`` (capacité statique
    annoncée par l'API) : ``total_capacity ≤ capacity`` quand des docks sont
    hors service. Sur ce projet, ~55% des relevés ont cette divergence —
    c'est l'opérationnel normal des stations Vélib', pas un bug.

    Args:
        df: DataFrame avec colonnes 'bikes_mechanical', 'bikes_ebike',
            'numdocksavailable'.

    Returns:
        Series d'entiers (nb de docks fonctionnels au moment du relevé).
    """
    return (
        df["bikes_mechanical"].fillna(0)
        + df["bikes_ebike"].fillna(0)
        + df["numdocksavailable"].fillna(0)
    ).astype(int)


def compute_fill_rate(
    df: pd.DataFrame,
    total_capacity: pd.Series | None = None,
) -> pd.Series:
    """Taux de remplissage = (vélos présents) / (capacité totale fonctionnelle) × 100.

    On utilise ``total_capacity`` (vélos+docks libres) au dénominateur, pas
    la colonne ``capacity`` statique. C'est le bon dénominateur quand des
    docks sont hors service (cas ~55% des relevés).

    Args:
        df: DataFrame avec colonnes 'bikes_mechanical', 'bikes_ebike',
            'numdocksavailable'.
        total_capacity: Series pré-calculée (économise un calcul si déjà fait).
            Si None, recalcule via ``compute_total_capacity(df)``.

    Returns:
        Series de floats dans [0, 100]. Lignes où total_capacity == 0 :
        retourne NaN (à filtrer en amont avec ``filter_valid_capacities``).
    """
    if total_capacity is None:
        total_capacity = compute_total_capacity(df)
    total_bikes = df["bikes_mechanical"].fillna(0) + df["bikes_ebike"].fillna(0)
    # Division protégée : NaN sur les capacités nulles (pas une fausse valeur 0)
    rate = pd.Series(index=df.index, dtype="float64")
    mask_valid = total_capacity > 0
    rate.loc[mask_valid] = (
        total_bikes.loc[mask_valid] / total_capacity.loc[mask_valid] * 100
    )
    return rate


# ─────────────────────────────────────────────────────────────────────────────
# FILTRES DE LIGNES
# ─────────────────────────────────────────────────────────────────────────────
def filter_active_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Garde uniquement les relevés où ``is_renting == True``.

    Les stations en maintenance, désactivées ou hors service apparaissent
    avec ``is_renting == False`` et ne sont pas exploitables pour la
    prédiction.

    Args:
        df: DataFrame avec colonne 'is_renting' (bool ou coercible).

    Returns:
        Nouveau DataFrame filtré (index réinitialisé).
    """
    if "is_renting" not in df.columns:
        return df.reset_index(drop=True)
    return df.loc[df["is_renting"].astype(bool)].reset_index(drop=True)


def filter_valid_capacities(
    df: pd.DataFrame,
    min_capacity: int = 1,
    total_capacity: pd.Series | None = None,
) -> pd.DataFrame:
    """Exclut les relevés avec capacité invalide.

    Deux types d'invalidités traitées :
        1. ``capacity`` statique == 0 (méta-donnée Vélib' cassée).
        2. ``total_capacity`` calculée < min_capacity (relevé en
           hors-service complet, division par zéro impossible).

    Args:
        df: DataFrame avec colonne 'capacity' et colonnes nécessaires
            au calcul de total_capacity.
        min_capacity: seuil minimum sur total_capacity (défaut 1).
        total_capacity: Series pré-calculée (sinon recalcule).

    Returns:
        Nouveau DataFrame filtré (index réinitialisé).
    """
    if total_capacity is None:
        total_capacity = compute_total_capacity(df)
    mask = (df["capacity"] > 0) & (total_capacity >= min_capacity)
    return df.loc[mask].reset_index(drop=True)


def filter_valid_fill_rate(df: pd.DataFrame, rate_col: str = "taux") -> pd.DataFrame:
    """Garde-fou défensif : taux ∈ [0, 100].

    Sur les données open-meteo + API Vélib' actuelles, ce filtre élimine
    0 ligne (vérifié sur 5M relevés). Conservé pour protéger d'un
    éventuel bug futur côté collector ou API source.

    Args:
        df: DataFrame avec une colonne de taux (défaut 'taux').
        rate_col: nom de la colonne de taux à valider.

    Returns:
        Nouveau DataFrame filtré (index réinitialisé).
    """
    if rate_col not in df.columns:
        return df.reset_index(drop=True)
    mask = df[rate_col].between(0, 100, inclusive="both")
    return df.loc[mask].reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# VARIANCE PAR STATION (étape 4 du preprocessing)
# ─────────────────────────────────────────────────────────────────────────────
def compute_station_variance(
    df: pd.DataFrame,
    rate_col: str = "taux",
    station_col: str = "station_id",
) -> pd.Series:
    """Variance du taux de remplissage par station.

    Sert à identifier les stations 'plates' (variance ~0) qui sont
    inentraînables : un modèle ne peut rien apprendre d'une station qui
    reste à 100% ou 0% en permanence.

    Args:
        df: DataFrame avec une colonne de taux et une colonne station_id.
        rate_col: nom de la colonne de taux (défaut 'taux').
        station_col: nom de la colonne identifiant la station.

    Returns:
        Series indexée par station_id, valeur = variance du taux.
    """
    return df.groupby(station_col)[rate_col].var()


def filter_stations_with_variance(
    df: pd.DataFrame,
    min_variance: float,
    rate_col: str = "taux",
    station_col: str = "station_id",
) -> tuple[pd.DataFrame, list[int]]:
    """Exclut les stations dont la variance du taux est < min_variance.

    Args:
        df: DataFrame d'entrée.
        min_variance: seuil minimum de variance (typiquement settings.min_variance).
        rate_col: colonne de taux.
        station_col: colonne station_id.

    Returns:
        Tuple (DataFrame filtré, liste des station_ids exclus).
    """
    variance_per_station = compute_station_variance(df, rate_col, station_col)
    excluded_ids = variance_per_station.loc[variance_per_station < min_variance].index.tolist()
    if excluded_ids:
        df = df.loc[~df[station_col].isin(excluded_ids)].reset_index(drop=True)
    return df, excluded_ids


# ─────────────────────────────────────────────────────────────────────────────
# MÉTÉO — catégorisation et features de sévérité
# ─────────────────────────────────────────────────────────────────────────────
def get_wmo_group(code: int) -> str:
    """Retourne la catégorie WMO descriptive d'un code météo.

    Utilisé principalement pour les graphiques (labels lisibles).

    Args:
        code: code WMO entier.

    Returns:
        Nom de catégorie (str). 'Autre' si code non répertorié.
    """
    for group, codes in WMO_GROUPS.items():
        if code in codes:
            return group
    return "Autre"


def apply_weather_severity(weather_code: pd.Series) -> dict[str, pd.Series]:
    """Calcule 3 features de sévérité météo à partir du code WMO.

    Logique préservée à l'identique de l'ancien 01_dataviz.py de l'équipe :

    weather_severity (échelle ordinale 0-4) :
        - 0 (Clair)        : codes 0-3, 4-19 implicites (initialisation)
                             [voir note ci-dessous]
        - 1 (Nuageux)      : codes 4-19 (codes WMO étendus, peu fréquents)
        - 2 (Pluie légère) : codes 20-29 ET 40-59 (bruine, brouillard)
        - 3 (Pluie forte)  : codes 60-69 ET 80-84 (pluie, averses)
        - 4 (Orage/neige)  : codes 30-39 ET 70-79 ET 85+ (tempête, neige, orage)

    is_frozen (binaire 0/1) :
        - 1 si code ∈ {56, 57, 66, 67, 70-79, 85, 86} (verglas, neige)

    is_stormy (binaire 0/1) :
        - 1 si code ∈ {17, 18, 19, 29, 90-99} (orages)

    Note importante : open-meteo (la source réelle) ne renvoie qu'un sous-ensemble
    des codes WMO. Sur les données réelles du projet, la severity = 1 sera
    rarement observée car les codes 4-19 ne sont pas standard open-meteo.

    Args:
        weather_code: Series de codes WMO (entiers).

    Returns:
        dict avec 3 Series ('weather_severity', 'is_frozen', 'is_stormy').
        L'orchestrateur appelle ``df.assign(**result)`` pour les ajouter.
    """
    code = weather_code.fillna(-1).astype(int)

    severity = pd.Series(0, index=code.index, dtype="int8")
    severity.loc[(code >= 4) & (code <= 19)] = 1
    severity.loc[
        ((code >= 20) & (code <= 29)) | ((code >= 40) & (code <= 59))
    ] = 2
    severity.loc[
        ((code >= 60) & (code <= 69)) | ((code >= 80) & (code <= 84))
    ] = 3
    severity.loc[
        ((code >= 30) & (code <= 39))
        | ((code >= 70) & (code <= 79))
        | (code >= 85)
    ] = 4

    frozen_codes = {56, 57, 66, 67, 85, 86, *range(70, 80)}
    is_frozen = code.isin(frozen_codes).astype("int8")

    stormy_codes = {17, 18, 19, 29, *range(90, 100)}
    is_stormy = code.isin(stormy_codes).astype("int8")

    return {
        "weather_severity": severity,
        "is_frozen": is_frozen,
        "is_stormy": is_stormy,
    }
