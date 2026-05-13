"""
shared.config — Configuration centralisée du projet Vélib' MLOps.

Toute la configuration est lue depuis les variables d'environnement
(ou un fichier .env à la racine du projet), validée par Pydantic, et
exposée via l'objet singleton ``settings``.

Avantages :
    - validation au démarrage : erreur immédiate si une variable est mal typée
    - autocomplétion : ``settings.raw_data_dir`` plutôt que ``os.environ["RAW_DIR"]``
    - centralisation : tous les services partagent les mêmes définitions
    - testabilité : on peut surcharger n'importe quel champ dans les tests

Usage :
    from shared.config import settings
    raw_path = settings.raw_data_dir / settings.raw_snapshot_filename

Convention :
    - les chemins sont toujours des objets Path absolus
    - les secrets (HF_TOKEN) sont en SecretStr (n'apparaît jamais dans les logs)
    - les valeurs par défaut sont prévues pour un développement local sain
"""
from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr, computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os

# ─────────────────────────────────────────────────────────────────────────────
# RACINE PROJET
# ─────────────────────────────────────────────────────────────────────────────
# Path résolu une seule fois au chargement du module.
# Hypothèse : ce fichier vit dans <repo>/shared/shared/config.py
# Donc <repo> = parent du parent du parent.
_REPO_ROOT = BASE_DIR = Path(os.getenv("APP_DIR", "/app"))
# _REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """Configuration globale du projet Vélib'."""

    model_config = SettingsConfigDict(
        env_file=_REPO_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,        # HF_TOKEN ou hf_token : indifférent
        extra="ignore",              # ignore les variables d'env non déclarées
    )

    # ─────────────────────────────────────────────────────────────────────────
    # HUGGINGFACE — source de la collecte continue
    # ─────────────────────────────────────────────────────────────────────────
    hf_repo: str = Field(
        default="voroman/velib-ml-data",
        description="Identifiant du repo HF datasets (format 'user/repo').",
    )
    hf_token: SecretStr | None = Field(
        default=None,
        description="Token HF (Read suffit si le repo est public). "
                    "Lu via env HF_TOKEN, jamais hardcodé.",
    )
    hf_file_prefix: str = Field(
        default="dataset_velib_raw_",
        description="Préfixe des fichiers raw à concaténer.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # DOSSIERS DATA — versionnés DVC
    # ─────────────────────────────────────────────────────────────────────────
    raw_data_dir: Path = Field(
        default=_REPO_ROOT / "data" / "raw",
        description="Dossier des snapshots bruts produits par load_from_hf.",
    )
    interim_data_dir: Path = Field(
        default=_REPO_ROOT / "data" / "interim",
        description="Dossier des données nettoyées (sortie make_dataset).",
    )
    processed_data_dir: Path = Field(
        default=_REPO_ROOT / "data" / "processed",
        description="Dossier des features finales train/test (sortie build_features).",
    )
    plots_dir: Path = Field(
        default=_REPO_ROOT / "data" / "outputs" / "plots",
        description="Dossier des graphiques produits par dataviz.",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # NOMS DE FICHIERS — convention 'latest' (DVC versionne le contenu)
    # ─────────────────────────────────────────────────────────────────────────
    raw_snapshot_filename: str = Field(
        default="velib_snapshot_latest.parquet",
        description="Nom du parquet brut produit par load_from_hf.",
    )
    interim_filename: str = Field(
        default="velib_cleaned_latest.parquet",
        description="Nom du parquet nettoyé produit par make_dataset.",
    )
    train_filename: str = Field(
        default="train_preprocessed.parquet",
        description="Nom du parquet d'entraînement (sortie build_features).",
    )
    test_filename: str = Field(
        default="test_preprocessed.parquet",
        description="Nom du parquet de test (sortie build_features).",
    )
    snapshot_log_filename: str = Field(
        default=".snapshots.log",
        description="Log humain des snapshots produits (timestamp, taille, période).",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # ML — paramètres d'entraînement
    # ─────────────────────────────────────────────────────────────────────────
    random_state: int = Field(
        default=42,
        description="Seed reproductibilité (split, modèles, échantillonnage).",
    )
    test_size: float = Field(
        default=0.20,
        ge=0.0,
        le=1.0,
        description="Proportion du test set (split temporel par quantile).",
    )
    min_variance: float = Field(
        default=0.5,
        ge=0.0,
        description="Variance minimale du taux par station "
                    "(filtre les stations 'plates' inentraînables).",
    )
    station_trend_min_days_for_month: int = Field(
        default=90,
        ge=1,
        description=(
            "Seuil (en jours d'historique train) à partir duquel la dimension "
            "'month' est ajoutée à la clé d'agrégation de station_trend_avg. "
            "En dessous : clé courte (station × dow × hour). "
            "Au-dessus : clé fine (station × dow × hour × month). "
            "Justification : 90 jours = ~3 mois distincts × ~12 occurrences "
            "par combinaison, pour que l'agrégation par mois soit "
            "statistiquement défendable."
        ),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LOGGING
    # ─────────────────────────────────────────────────────────────────────────
    log_level: str = Field(
        default="INFO",
        description="Niveau de log (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    log_format: str = Field(
        default="text",
        description="Format des logs : 'text' (humain) ou 'json' (machine, "
                    "pour Loki/OpenTelemetry).",
    )

    # ─────────────────────────────────────────────────────────────────────────
    # COMPORTEMENT load_from_hf
    # ─────────────────────────────────────────────────────────────────────────
    hf_force_download: bool = Field(
        default=True,
        description="Si True, ignore le cache HF local et re-télécharge tout. "
                    "À passer à False quand le dataset deviendra trop gros (>2 Go).",
    )
    hf_max_retries: int = Field(
        default=3,
        ge=1,
        description="Nombre de tentatives en cas d'erreur réseau transitoire.",
    )
    hf_retry_initial_wait_s: float = Field(
        default=5.0,
        ge=0.0,
        description="Délai initial avant retry (secondes). Backoff exponentiel ensuite.",
    )

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow-server:5000"
    mlflow_artifact_uri: str = "file:///app/mlflow/artifacts"
    mlflow_experiment_name: str = "velib-metropole"
    mlflow_run_name: str = "velib-metropole-run"
    mlflow_model_name: str = "velib-metropole-model"

    # ─────────────────────────────────────────────────────────────────────────
    # VALIDATEURS
    # ─────────────────────────────────────────────────────────────────────────
    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        v_up = v.upper()
        if v_up not in valid:
            raise ValueError(f"log_level invalide : {v!r}. Valides : {sorted(valid)}")
        return v_up

    @field_validator("log_format")
    @classmethod
    def _validate_log_format(cls, v: str) -> str:
        valid = {"text", "json"}
        v_low = v.lower()
        if v_low not in valid:
            raise ValueError(f"log_format invalide : {v!r}. Valides : {sorted(valid)}")
        return v_low

    @field_validator("hf_repo")
    @classmethod
    def _validate_hf_repo(cls, v: str) -> str:
        # Format attendu : 'user/repo'
        # Si l'utilisateur passe une URL complète, on essaie d'extraire user/repo
        v = v.strip().rstrip("/")
        if v.startswith("http"):
            # https://huggingface.co/datasets/user/repo  ou  github.com/user/repo
            # On prend les deux derniers segments du path
            parts = v.split("/")
            if len(parts) >= 2:
                v = f"{parts[-2]}/{parts[-1]}"
        if "/" not in v or len(v.split("/")) != 2:
            raise ValueError(
                f"hf_repo doit être au format 'user/repo' (reçu : {v!r})"
            )
        return v

    # ─────────────────────────────────────────────────────────────────────────
    # CHEMINS DÉRIVÉS — calculés une fois, exposés en lecture seule
    # ─────────────────────────────────────────────────────────────────────────
    @computed_field
    @property
    def raw_snapshot_path(self) -> Path:
        """Chemin complet du parquet brut produit par load_from_hf."""
        return self.raw_data_dir / self.raw_snapshot_filename

    @computed_field
    @property
    def interim_path(self) -> Path:
        """Chemin complet du parquet nettoyé produit par make_dataset."""
        return self.interim_data_dir / self.interim_filename

    @computed_field
    @property
    def train_path(self) -> Path:
        return self.processed_data_dir / self.train_filename

    @computed_field
    @property
    def test_path(self) -> Path:
        return self.processed_data_dir / self.test_filename

    @computed_field
    @property
    def snapshot_log_path(self) -> Path:
        """Log humain consigné après chaque exécution de load_from_hf."""
        return self.raw_data_dir / self.snapshot_log_filename

    @computed_field
    @property
    def repo_root(self) -> Path:
        """Racine du repo, déduite de l'emplacement de ce fichier."""
        return _REPO_ROOT

    # ─────────────────────────────────────────────────────────────────────────
    # MÉTHODES UTILITAIRES
    # ─────────────────────────────────────────────────────────────────────────
    def hf_token_value(self) -> str | None:
        """Retourne le token HF en clair, ou None si non défini.

        À utiliser uniquement au moment de l'appel à l'API HF.
        Le SecretStr garantit que le token n'apparaît pas dans les ``__repr__``.
        """
        return self.hf_token.get_secret_value() if self.hf_token else None

    def ensure_directories(self) -> None:
        """Crée tous les dossiers data manquants (idempotent)."""
        for d in (
            self.raw_data_dir,
            self.interim_data_dir,
            self.processed_data_dir,
            self.plots_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)


# Instance singleton — importée par tous les services
settings = Settings()
