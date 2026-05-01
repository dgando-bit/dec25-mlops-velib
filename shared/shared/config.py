from pathlib import Path
from pydantic_settings import BaseSettings
from pydantic import ConfigDict
from functools import lru_cache
import os

BASE_DIR = Path(os.getenv("APP_DIR", "/app"))


class Settings(BaseSettings):
    # API
    api_url: str = "https://mon-api.example.com"
    critical_threshold: int = 3

    # MLflow
    mlflow_tracking_uri: str = "http://mlflow-server:5000"
    mlflow_artifact_uri: str = "file:///app/mlflow/artifacts"
    mlflow_experiment_name: str = "velib-metropole"
    mlflow_run_name: str = "velib-metropole-run"
    mlflow_model_name: str = "velib-metropole-model"

    # Paths
    data_dir: Path = BASE_DIR / "data"

    # Debug
    debug: bool = False

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings() -> Settings:
    return Settings()