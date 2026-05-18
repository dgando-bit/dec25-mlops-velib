import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.pipeline import Pipeline

from shared.config import settings
from shared.logger import get_logger
from data import build_dataset
from config import DatasetSplit, MODELS
from pipeline import build_training_pipeline
from evaluation import evaluate

logger = get_logger(__name__)


def train(dframe: pd.DataFrame, model_name: str = "lgbm", cutoff: float = 0.8) -> tuple[Pipeline, DatasetSplit]:
    dataset_split = build_dataset(dframe, cutoff)
    model = MODELS[model_name]

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("velib-capacity-prediction")

    with mlflow.start_run():
        model_pipeline = build_training_pipeline(model)
        model_pipeline.fit(dataset_split.X_train, dataset_split.y_train_delta)

        # Métriques et enregistrement modèle
        evaluate(model_pipeline, dataset_split)

        # Paramètres
        mlflow.log_param("model_type", model_name)
        model_params = model_pipeline.named_steps["model"].get_params()
        mlflow.log_params(model_params)

        # Modèle
        # mlflow.sklearn.log_model(
        #     model_pipeline,
        #     artifact_path="model",
        #     registered_model_name=settings.registered_model_name,
        # )

    return model_pipeline, dataset_split


if __name__ == '__main__':
    logger.info("Lancement du train...")
    df = pd.read_parquet(settings.raw_data_dir / settings.raw_snapshot_filename)
    pipeline, split = train(df, model_name="lgbm")
    logger.info("Fin du train.")