from dataclasses import dataclass

import lightgbm as lgb
import pandas as pd
import xgboost as xgb

COLUMNS_TO_DROP = [
    "last_reported_h",
    "name",
    "capacity_status",
    "weather_code",
]

COLS_STANDARD = ['apparent_temperature', 'lat', 'lon', 'diff_1h_2h', 'diff_1h_24h']
COLS_BINARY   = ['is_holiday', 'is_vacation', 'weather_severity']

@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test:  pd.DataFrame
    y_train_delta: pd.Series
    y_test_delta:  pd.Series
    lag_1h_test: pd.Series
    lag_1h_train: pd.Series
    y_true: pd.Series
    y_train_true: pd.Series


MODELS = {
    "xgb":  xgb.XGBRegressor(),
    "lgbm": lgb.LGBMRegressor(),
}

PARAM_GRIDS = {
    "lgbm": {
        'model__n_estimators':  [100, 300],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth':     [5, 10],
    },
    "xgb": {
        'model__n_estimators':  [100, 300],
        'model__learning_rate': [0.05, 0.1],
        'model__max_depth':     [3, 6],
    },
}