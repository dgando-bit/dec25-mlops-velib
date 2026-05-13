import pandas as pd
from config import COLUMNS_TO_DROP, DatasetSplit
from ml.src.ml_src.features import build_feature_engineering


def split_by_date(df: pd.DataFrame, cutoff: float = 0.8):
    df = df.copy()
    df['last_reported_h'] = pd.to_datetime(df['last_reported_h'])
    seuil = df['last_reported_h'].quantile(cutoff)
    return df[df['last_reported_h'] < seuil], df[df['last_reported_h'] >= seuil]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=COLUMNS_TO_DROP, errors='ignore')


def build_target(df: pd.DataFrame) -> pd.Series:
    return df['capacity_status'] - df['capacity_status_lag_1h']


def prepare(train_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetSplit:
    y_true       = test_df['capacity_status'].copy()
    y_train_true = train_df['capacity_status'].copy()
    lag_1h_test  = test_df['capacity_status_lag_1h'].copy()
    lag_1h_train = train_df['capacity_status_lag_1h'].copy()

    return DatasetSplit(
        X_train=build_features(train_df),
        X_test=build_features(test_df),
        y_train_delta=build_target(train_df),
        y_test_delta=build_target(test_df),
        lag_1h_test=lag_1h_test,
        lag_1h_train=lag_1h_train,
        y_true=y_true,
        y_train_true=y_train_true,
    )

def build_dataset(df: pd.DataFrame, cutoff: float = 0.8) -> DatasetSplit:
    df = build_feature_engineering(df)
    train_df, test_df = split_by_date(df, cutoff)
    return prepare(train_df, test_df)