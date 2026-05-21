"""
ml.src.features._helpers — Briques de feature engineering du pipeline Vélib'.

Module privé au sous-package ``features`` (préfixe ``_`` par convention).
Regroupe toutes les transformations spécifiques au feature engineering ML
qui n'ont pas vocation à être réutilisées hors de ce module :

    - add_temporal_features       : hour, dow, month, encoding cyclique
    - add_capacity_group          : profil de taille des stations (0/1/2/3)
    - add_weather_features        : weather_severity, is_frozen, is_stormy
    - encode_calendar_booleans    : is_holiday, is_vacation → int 0/1
    - add_temporal_lags           : lag_60min, lag_240min via merge_asof
    - temporal_split              : split train/test par quantile temporel
    - add_post_split_features     : morning_evening_ratio, temp_anomalie
                                    (sur train uniquement, mappé sur test)
    - add_station_trend_avg       : moyenne historique avec fallback 2 niveaux
    - add_residual_target         : résidu et lag résiduel

Toutes les fonctions sont **pures** (pas de side-effect, pas d'I/O).
L'orchestrateur ``build_features.py`` les compose dans le bon ordre.

Note importante sur la prévention des fuites (data leakage) :
    Certaines features sont calculées sur le train uniquement, puis appliquées
    au test via mapping. C'est le cas de :
        - station_trend_avg (étape 9 originale)
        - morning_evening_ratio
        - temp_anomalie
    Si on les calculait sur le dataset entier avant le split, l'information
    du test set "fuiterait" dans le train, gonflant artificiellement
    les métriques d'évaluation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shared.config import settings
from shared.logger import get_logger
from shared.utils.data_cleaning import apply_weather_severity

logger = get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ─────────────────────────────────────────────────────────────────────────────

# Nom de la cible brute (taux de remplissage en %)
TARGET_RAW = "taux"
# Nom de la cible résiduelle (taux - station_trend_avg)
TARGET_RESIDUAL = "residual_target"

# Bornes des groupes de capacité (étape 4 originale)
CAPACITY_BINS = [0, 20, 35, 50, 999]
CAPACITY_LABELS = [0, 1, 2, 3]   # petite, moyenne, grande, très grande

# Heures de pointe (matin et soir)
PEAK_HOURS = [7, 8, 9, 17, 18, 19]
MORNING_HOURS = [7, 8, 9]
EVENING_HOURS = [17, 18, 19]

# Tolérance pour le merge_asof des lags (±30 minutes)
LAG_TOLERANCE = pd.Timedelta(minutes=30)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE A — FEATURES TEMPORELLES (sin/cos cycliques + flags binaires)
# ─────────────────────────────────────────────────────────────────────────────
def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les features temporelles brutes et leur encodage cyclique.

    Encodage cyclique — pourquoi ?
        Pour un modèle, hour=23 et hour=0 semblent distants de 23 unités.
        En projetant sur un cercle (sin/cos), 23h et 0h deviennent voisins.
        Même logique pour dimanche → lundi.

    Args:
        df: DataFrame avec colonne 'datetime' (datetime64).

    Returns:
        Nouveau DataFrame avec les colonnes :
            hour, day_of_week, month                    (brutes)
            hour_sin, hour_cos, dow_sin, dow_cos        (cycliques)
            is_peak_hour, is_friday_evening,
            is_monday_morning                            (flags binaires)
    """
    out = df.copy()

    # Brutes
    out["hour"] = out["datetime"].dt.hour
    out["day_of_week"] = out["datetime"].dt.dayofweek   # 0=lundi, 6=dimanche
    out["month"] = out["datetime"].dt.month

    # Cyclique (encodage sur le cercle unité)
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24).round(4)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24).round(4)
    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7).round(4)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7).round(4)

    # Flags d'interaction temporelle
    out["is_peak_hour"] = out["hour"].isin(PEAK_HOURS).astype("int8")
    out["is_friday_evening"] = (
        (out["day_of_week"] == 4) & (out["hour"] >= 17)
    ).astype("int8")
    out["is_monday_morning"] = (
        (out["day_of_week"] == 0) & (out["hour"] < 10)
    ).astype("int8")

    return out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE B — PROFIL DE TAILLE DES STATIONS
# ─────────────────────────────────────────────────────────────────────────────
def add_capacity_group(df: pd.DataFrame) -> pd.DataFrame:
    """Catégorise les stations par taille (capacity → groupe 0/1/2/3).

    Catégories :
        0 = petite      (≤ 20 docks)
        1 = moyenne     (21-35 docks)
        2 = grande      (36-50 docks)
        3 = très grande (> 50 docks)

    Args:
        df: DataFrame avec colonne 'capacity' (int).

    Returns:
        Nouveau DataFrame avec colonne 'capacity_group' (int 0-3).
    """
    out = df.copy()
    out["capacity_group"] = pd.cut(
        out["capacity"],
        bins=CAPACITY_BINS,
        labels=CAPACITY_LABELS,
    ).astype("int8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE C — FEATURES MÉTÉO (réutilise apply_weather_severity de shared/utils)
# ─────────────────────────────────────────────────────────────────────────────
def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """Ajoute les 3 features météo : weather_severity, is_frozen, is_stormy.

    Délègue le calcul au helper ``shared.utils.data_cleaning.apply_weather_severity``
    qui retourne un dict de 3 Series — l'orchestrateur les ajoute via assign().

    Args:
        df: DataFrame avec colonne 'weather_code' (int).

    Returns:
        Nouveau DataFrame avec les 3 colonnes ajoutées.
    """
    weather_features = apply_weather_severity(df["weather_code"])
    return df.assign(**weather_features)


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE D — ENCODAGE BOOLÉENS CALENDAIRES → INT
# ─────────────────────────────────────────────────────────────────────────────
def encode_calendar_booleans(df: pd.DataFrame) -> pd.DataFrame:
    """Convertit is_holiday et is_vacation en entier 0/1.

    Le DataFrame nettoyé par make_dataset.py a déjà coercé ces colonnes en
    bool stricts. On les passe ici en int8 pour minimiser la taille mémoire
    et garantir la compatibilité avec XGBoost / sklearn.

    Args:
        df: DataFrame avec colonnes 'is_holiday', 'is_vacation' (bool).

    Returns:
        Nouveau DataFrame avec ces colonnes en int8.
    """
    out = df.copy()
    for col in ("is_holiday", "is_vacation"):
        if col in out.columns:
            out[col] = out[col].astype("int8")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE E — LAGS TEMPORELS via merge_asof
# ─────────────────────────────────────────────────────────────────────────────
def add_temporal_lags(
    df: pd.DataFrame,
    lag_minutes: tuple[int, ...] = (60, 240),
) -> pd.DataFrame:
    """Ajoute des lags du taux à durée fixe (60min et 240min par défaut).

    Pourquoi merge_asof plutôt que shift(N) ?
        shift(N) décale de N LIGNES, pas de N minutes. Avec des intervalles
        de collecte variables (5 min nominal, mais des trous possibles),
        shift(8) peut représenter 40 min ou 12h selon les jours.
        merge_asof cherche le relevé le plus proche d'une DURÉE fixe dans
        le passé, garantissant un lag temporel réel.

    Tolérance ±30 min : si aucun relevé dans cette fenêtre, NaN.

    Implémentation : preserve l'index original via reset_index/set_index
    pour éviter les index dupliqués lors du concat des groupes par station.

    Args:
        df: DataFrame avec colonnes 'station_id', 'datetime', TARGET_RAW.
            Doit être déjà trié par (station_id, datetime).
        lag_minutes: tuple des durées de lag à calculer (en minutes).

    Returns:
        Nouveau DataFrame avec les colonnes 'lag_<N>min' ajoutées.
        Les lignes avec lag NaN (historique insuffisant) sont supprimées.
    """
    # Normaliser datetime64[us, UTC] → datetime64[ns, UTC] pour merge_asof
    df = df.copy()
    df["datetime"] = df["datetime"].dt.as_unit("ns")

    out = df.sort_values(["station_id", "datetime"]).reset_index(drop=True)

    def _compute_one_lag(minutes: int, col_name: str) -> pd.Series:
        """Pour chaque relevé, trouve le taux le plus proche `minutes` min avant."""
        lag_time = pd.Timedelta(minutes=minutes)
        results: list[pd.Series] = []

        for _, group in out.groupby("station_id", sort=False):
            group = group.sort_values("datetime")
            ref = group.copy()
            ref["datetime_lag"] = ref["datetime"] + lag_time

            # reset_index() : sauvegarde l'index original dans une colonne "_idx"
            # pour pouvoir le restaurer après merge_asof (qui le réinitialise)
            merged = pd.merge_asof(
                group[["datetime", TARGET_RAW]]
                .reset_index().rename(columns={"index": "_idx"}),
                ref[["datetime_lag", TARGET_RAW]].rename(
                    columns={"datetime_lag": "datetime", TARGET_RAW: col_name}
                ),
                on="datetime",
                direction="forward",
                tolerance=LAG_TOLERANCE,
            )
            merged = merged.set_index("_idx")
            results.append(merged[col_name])

        return pd.concat(results).reindex(out.index)

    n_before = len(out)
    for minutes in lag_minutes:
        col_name = f"lag_{minutes}min"
        out[col_name] = _compute_one_lag(minutes, col_name)

    # Supprime les lignes sans historique suffisant (NaN sur tous les lags)
    lag_cols = [f"lag_{m}min" for m in lag_minutes]
    out = out.dropna(subset=lag_cols).reset_index(drop=True)
    n_after = len(out)

    logger.info(
        "Lags temporels créés",
        extra={
            "lag_minutes": list(lag_minutes),
            "rows_before": n_before,
            "rows_after": n_after,
            "rows_dropped": n_before - n_after,
        },
    )
    return out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE F — SPLIT TEMPOREL TRAIN/TEST
# ─────────────────────────────────────────────────────────────────────────────
def temporal_split(
    df: pd.DataFrame,
    test_size: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split 80/20 par quantile temporel sur la colonne datetime.

    Pourquoi un split temporel et pas aléatoire ?
        Un split aléatoire permettrait au modèle de s'entraîner sur des
        données "du futur". En production, on prédit toujours à partir
        du passé — le split temporel simule cette condition réelle.

    Args:
        df: DataFrame avec colonne 'datetime'.
        test_size: proportion du test (défaut: settings.test_size).

    Returns:
        Tuple (train, test) — chacun avec son propre index réinitialisé.
    """
    if test_size is None:
        test_size = settings.test_size

    cutoff = df["datetime"].quantile(1.0 - test_size)
    train = df.loc[df["datetime"] <= cutoff].reset_index(drop=True)
    test = df.loc[df["datetime"] > cutoff].reset_index(drop=True)

    logger.info(
        "Split temporel effectué",
        extra={
            "cutoff": str(cutoff),
            "train_rows": len(train),
            "test_rows": len(test),
            "train_pct": round(len(train) / len(df) * 100, 1),
            "test_pct": round(len(test) / len(df) * 100, 1),
        },
    )
    return train, test


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE G — FEATURES POST-SPLIT (anti-fuite)
# ─────────────────────────────────────────────────────────────────────────────
def add_post_split_features(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calcule morning_evening_ratio et temp_anomalie sur le train UNIQUEMENT.

    Anti-fuite : si on calculait ces features sur le dataset complet avant
    le split, l'information du test "fuiterait" dans le train et fausserait
    les métriques.

    morning_evening_ratio :
        Ratio taux_moyen_matin (7-9h) / taux_moyen_soir (17-19h) par station.
        Calculé sur le train, mappé sur le test via map(station_id).
        Fallback : 1.0 (profil mixte) pour les stations absentes du train.

    temp_anomalie :
        Écart à la normale mensuelle de température.
        Normale calculée sur le train, appliquée au test via map(month).

    Args:
        train: DataFrame d'entraînement (avec colonnes hour, station_id,
               apparent_temperature, month).
        test: DataFrame de test (mêmes colonnes).

    Returns:
        Tuple (train, test) enrichis des deux nouvelles features.
    """
    train_out = train.copy()
    test_out = test.copy()

    # ── morning_evening_ratio (calculé sur train) ─────────────────────────
    morning_avg = (
        train_out.loc[train_out["hour"].isin(MORNING_HOURS)]
        .groupby("station_id")[TARGET_RAW].mean()
    )
    evening_avg = (
        train_out.loc[train_out["hour"].isin(EVENING_HOURS)]
        .groupby("station_id")[TARGET_RAW].mean()
    )
    station_ratio = (morning_avg / evening_avg.replace(0, np.nan)).dropna().round(3)

    for df_ in (train_out, test_out):
        df_["morning_evening_ratio"] = (
            df_["station_id"].map(station_ratio).fillna(1.0).round(3)
        )

    n_residential = int((station_ratio > 1.2).sum())
    n_office = int((station_ratio < 0.8).sum())
    n_mixed = int(((station_ratio >= 0.8) & (station_ratio <= 1.2)).sum())
    logger.info(
        "morning_evening_ratio calculé sur train",
        extra={
            "residential": n_residential,
            "office": n_office,
            "mixed": n_mixed,
        },
    )

    # ── temp_anomalie (normale mensuelle calculée sur train) ──────────────
    monthly_mean = train_out.groupby("month")["apparent_temperature"].mean()

    for df_ in (train_out, test_out):
        df_["temp_anomalie"] = (
            df_["apparent_temperature"] - df_["month"].map(monthly_mean)
        ).round(2)

    logger.info(
        "temp_anomalie calculée sur train",
        extra={"sigma_temp_anomalie": round(float(train_out["temp_anomalie"].std()), 2)},
    )

    return train_out, test_out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE H — STATION_TREND_AVG (cascade adaptative selon la taille du dataset)
# ─────────────────────────────────────────────────────────────────────────────
# Mode "clé courte" (datasets < seuil_jours, défaut 90j) :
#     L1 : station × dow × hour            ← clé principale
#     L2 : station × hour                  ← fallback si L1 vide
#     L3 : moyenne globale                 ← dernier recours
#
# Mode "clé fine" (datasets >= seuil_jours) :
#     L1 : station × dow × hour × month    ← clé principale (le mois apporte de la précision)
#     L2 : station × dow × hour            ← fallback si L1 vide
#     L3 : station × hour                  ← fallback si L2 vide
#     L4 : moyenne globale                 ← dernier recours
#
# Cohérence : chaque niveau retire une dimension du précédent.
# ─────────────────────────────────────────────────────────────────────────────
def add_station_trend_avg(
    train: pd.DataFrame,
    test: pd.DataFrame,
    min_obs: int = 3,
    min_days_for_month: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Calcule le comportement historique typique de chaque station.

    Pour chaque combinaison définie par la clé principale L1, calcule la
    moyenne du taux **uniquement sur le train**, puis applique sur train+test.

    La granularité de la clé L1 est adaptative selon la taille du dataset :
        - datasets courts (< min_days_for_month jours) : clé sans mois
          (sinon le mois discrimine artificiellement et casse les fallbacks)
        - datasets longs : clé avec mois (gain de précision)

    Cette feature est typiquement la plus importante du modèle XGBoost.
    Sa robustesse est cruciale : un fallback mal calibré → métriques cassées.

    Args:
        train: DataFrame d'entraînement (colonnes station_id, day_of_week,
               hour, month, datetime, TARGET_RAW).
        test: DataFrame de test (mêmes colonnes).
        min_obs: nombre minimum d'observations pour valider la clé L1.
        min_days_for_month: seuil de bascule clé courte → clé fine.
            Si None, lit settings.station_trend_min_days_for_month (défaut 90).

    Returns:
        Tuple (train, test, mode) avec :
            - train, test : DataFrames enrichis de la colonne 'station_trend_avg'
            - mode : "short_key" ou "fine_key" — utile pour tag MLflow
    """
    if min_days_for_month is None:
        min_days_for_month = settings.station_trend_min_days_for_month

    # ── Détection adaptative du mode ──────────────────────────────────────
    n_days = (train["datetime"].max() - train["datetime"].min()).days
    use_fine_key = n_days >= min_days_for_month
    mode = "fine_key" if use_fine_key else "short_key"

    if use_fine_key:
        groupby_l1 = ["station_id", "day_of_week", "hour", "month"]
        groupby_l2 = ["station_id", "day_of_week", "hour"]
        groupby_l3 = ["station_id", "hour"]
    else:
        groupby_l1 = ["station_id", "day_of_week", "hour"]
        groupby_l2 = ["station_id", "hour"]
        groupby_l3 = None  # cascade plus courte en mode short_key

    global_mean = float(train[TARGET_RAW].mean())

    logger.info(
        "Mode station_trend_avg sélectionné",
        extra={
            "mode": mode,
            "n_days_train": n_days,
            "threshold_days": min_days_for_month,
            "groupby_l1": "×".join(groupby_l1),
        },
    )

    # ── Niveau 1 : clé principale ─────────────────────────────────────────
    s_mean = train.groupby(groupby_l1)[TARGET_RAW].mean().round(2)
    s_count = train.groupby(groupby_l1)[TARGET_RAW].count()

    trend_raw = (
        s_mean.to_frame("trend_mean")
        .join(s_count.to_frame("trend_count"))
        .reset_index()
    )
    trend_raw["station_trend_avg"] = np.where(
        trend_raw["trend_count"] >= min_obs,
        trend_raw["trend_mean"],
        np.nan,
    )
    trend_l1 = trend_raw[groupby_l1 + ["station_trend_avg"]]

    n_l1_valid = int((trend_raw["trend_count"] >= min_obs).sum())
    n_l1_ignored = int((trend_raw["trend_count"] < min_obs).sum())

    # ── Niveau 2 : fallback ───────────────────────────────────────────────
    trend_l2 = (
        train.groupby(groupby_l2)[TARGET_RAW]
        .mean().round(2)
        .to_frame("station_trend_l2")
        .reset_index()
    )

    # ── Niveau 3 : fallback (uniquement en mode fine_key) ─────────────────
    trend_l3 = None
    if groupby_l3 is not None:
        trend_l3 = (
            train.groupby(groupby_l3)[TARGET_RAW]
            .mean().round(2)
            .to_frame("station_trend_l3")
            .reset_index()
        )

    # ── Application sur train et test (cascade) ───────────────────────────
    def _apply(df_in: pd.DataFrame) -> pd.DataFrame:
        df_out = df_in.merge(trend_l1, on=groupby_l1, how="left")
        df_out = df_out.merge(trend_l2, on=groupby_l2, how="left")
        if trend_l3 is not None:
            df_out = df_out.merge(trend_l3, on=groupby_l3, how="left")
            df_out["station_trend_avg"] = (
                df_out["station_trend_avg"]
                .fillna(df_out["station_trend_l2"])
                .fillna(df_out["station_trend_l3"])
                .fillna(global_mean)
            )
            df_out = df_out.drop(columns=["station_trend_l2", "station_trend_l3"])
        else:
            df_out["station_trend_avg"] = (
                df_out["station_trend_avg"]
                .fillna(df_out["station_trend_l2"])
                .fillna(global_mean)
            )
            df_out = df_out.drop(columns=["station_trend_l2"])
        return df_out

    train_out = _apply(train)
    test_out = _apply(test)

    # ── Diagnostic : combien de lignes test sont sur chaque niveau ────────
    # Utile pour valider la cascade en soutenance
    test_l1 = test_out.merge(
        trend_l1.rename(columns={"station_trend_avg": "_trend_l1_diag"}),
        on=groupby_l1,
        how="left",
    )
    n_test_on_l1 = int(test_l1["_trend_l1_diag"].notna().sum())
    n_test_total = len(test_out)

    correlation_train = float(
        train_out["station_trend_avg"].corr(train_out[TARGET_RAW])
    )
    correlation_test = float(
        test_out["station_trend_avg"].corr(test_out[TARGET_RAW])
    )

    logger.info(
        "station_trend_avg calculé",
        extra={
            "mode": mode,
            "level_1_valid_combinations": n_l1_valid,
            "level_1_ignored_combinations": n_l1_ignored,
            "test_rows_matched_on_L1": n_test_on_l1,
            "test_rows_total": n_test_total,
            "test_L1_coverage_pct": round(n_test_on_l1 / n_test_total * 100, 1),
            "global_mean": round(global_mean, 2),
            "correlation_train": round(correlation_train, 3),
            "correlation_test": round(correlation_test, 3),
        },
    )

    return train_out, test_out, mode


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE I — CIBLE RÉSIDUELLE + LAG RÉSIDUEL
# ─────────────────────────────────────────────────────────────────────────────
def add_residual_target(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajoute la cible résiduelle (pour le modèle) + lag résiduel à 240 min.

    Cible résiduelle = taux_réel - station_trend_avg
        Lecture : "Combien de points la station est au-dessus/en-dessous
        de sa normale historique ?"
        En production : taux_prédit = station_trend_avg + résidu_prédit

    Avantage : réduit la variance de la cible d'un facteur ~2.7
        σ(taux brut) ≈ 29% → σ(résidu) ≈ 11%
    Le problème devient plus ciblé, plus facile pour le modèle.

    lag_res_240min = lag_240min - station_trend_avg
        Lecture : "La station était X points au-dessus/en-dessous de sa
        normale 4h auparavant."

    Args:
        train: DataFrame avec colonnes TARGET_RAW, station_trend_avg,
               et lag_240min.
        test: DataFrame avec mêmes colonnes.

    Returns:
        Tuple (train, test) avec colonnes 'residual_target' et
        'lag_res_240min' ajoutées.
    """
    train_out = train.copy()
    test_out = test.copy()

    for df_ in (train_out, test_out):
        df_[TARGET_RESIDUAL] = (
            df_[TARGET_RAW] - df_["station_trend_avg"]
        ).round(2)
        df_["lag_res_240min"] = (
            df_["lag_240min"] - df_["station_trend_avg"]
        ).round(2)

    sigma_raw = float(train_out[TARGET_RAW].std())
    sigma_residual = float(train_out[TARGET_RESIDUAL].std())
    reduction_factor = sigma_raw / sigma_residual if sigma_residual > 0 else float("nan")

    logger.info(
        "Cible résiduelle créée",
        extra={
            "sigma_target_raw": round(sigma_raw, 2),
            "sigma_residual": round(sigma_residual, 2),
            "variance_reduction_factor": round(reduction_factor, 2),
        },
    )

    return train_out, test_out


# ─────────────────────────────────────────────────────────────────────────────
# ÉTAPE J — TABLE STATIONS_GEO (utilitaire pour Streamlit / API)
# ─────────────────────────────────────────────────────────────────────────────
def extract_stations_geo(df: pd.DataFrame) -> pd.DataFrame:
    """Extrait la table de correspondance station_id ↔ name + GPS.

    Utile pour Streamlit (carte des stations), pour l'API d'inférence
    (validation que station_id existe), et pour la documentation.

    Args:
        df: DataFrame avec colonnes 'station_id', 'name', 'lat', 'lon'.

    Returns:
        DataFrame dédupliqué (1 ligne par station), trié par station_id.
    """
    geo_cols = ["station_id", "name", "lat", "lon", "capacity"]
    available = [c for c in geo_cols if c in df.columns]
    return (
        df[available]
        .drop_duplicates(subset=["station_id"])
        .sort_values("station_id")
        .reset_index(drop=True)
    )
