from shared.logger import get_logger
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

import mlflow
from mlflow.models.signature import infer_signature
from config import DatasetSplit
from shared.config import settings

logger = get_logger(__name__)


def evaluate(pipeline: Pipeline, split: DatasetSplit) -> dict:
	# Métriques
	delta_preds = pipeline.predict(split.X_test)
	y_pred = split.lag_1h_test.values + delta_preds

	metrics = {
		"mae": mean_absolute_error(split.y_true, y_pred),
		"rmse": root_mean_squared_error(split.y_true, y_pred),
		"r2_score": r2_score(split.y_true, y_pred),
	}

	mlflow.log_metrics(metrics)
	logger.info(" | ".join(f"{k.upper()}: {v:.4f}" for k, v in metrics.items()))

	_log_model(pipeline, split)
	return metrics


def _log_model(pipeline: Pipeline, split: DatasetSplit) -> None:
	# Caster tout en float pour éviter les warnings MLflow sur les integers
	X_sample = split.X_train.astype({
		col: float
		for col in split.X_train.select_dtypes(include='integer').columns
	})

	signature = infer_signature(X_sample, pipeline.predict(X_sample))

	mlflow.sklearn.log_model(
		pipeline,
		artifact_path="model",
		registered_model_name=settings.registered_model_name,
		signature=signature,
		input_example=X_sample.iloc[:5],
	)
