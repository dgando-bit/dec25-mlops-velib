from sklearn.pipeline import Pipeline
from sklearn.base import RegressorMixin
import lightgbm as lgb
from ml.src.ml_src.preprocessing import build_preprocessor


def build_training_pipeline(model: RegressorMixin | None = None) -> Pipeline:
    return Pipeline(steps=[
        ("preprocessor", build_preprocessor()),
        ("model",        model or lgb.LGBMRegressor()),
    ])