import mlflow
import mlflow.sklearn
import pandas as pd
from shared.config import settings
from shared.logger import get_logger
from sklearn.model_selection import GridSearchCV

from config import DatasetSplit, MODELS, PARAM_GRIDS
from data import build_dataset
from evaluation import evaluate
from pipeline import build_training_pipeline

logger = get_logger(__name__)

def tune(dframe: pd.DataFrame, model_name: str = "lgbm", cutoff: float = 0.8, cv: int = 5) -> tuple[GridSearchCV, DatasetSplit]:
    model      = MODELS[model_name]
    param_grid = PARAM_GRIDS[model_name]

    dataset_split = build_dataset(dframe, cutoff)

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment("velib-capacity-tuning")

    with mlflow.start_run():
        grid = GridSearchCV(
            build_training_pipeline(model),
            param_grid,
            cv=cv,
            scoring='neg_mean_absolute_error',
            n_jobs=-1,
        )
        grid.fit(dataset_split.X_train, dataset_split.y_train_delta)

        logger.info(f"Meilleurs paramètres : {grid.best_params_}")

        # Métriques et enregistrement du modèle
        evaluate(grid.best_estimator_, dataset_split)

        # Paramètres
        mlflow.log_params(grid.best_params_)
        mlflow.log_param("model_type", type(grid.best_estimator_.named_steps["model"]).__name__)

        # Modèle
        # mlflow.sklearn.log_model(
        #     grid.best_estimator_,
        #     artifact_path="model",
        #     registered_model_name=settings.registered_model_name,
        # )

    return grid, dataset_split


if __name__ == '__main__':
    logger.info("Lancement du tuning...")
    df = pd.read_parquet(settings.raw_data_dir / settings.raw_snapshot_filename)
    grid, split = tune(df, model_name="xgb")
    logger.info(f"Fin du tuning. Meilleur score MAE : {-grid.best_score_:.4f}")