import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import TargetEncoder, StandardScaler, FunctionTransformer
from config import COLS_STANDARD, COLS_BINARY

sklearn.set_config(transform_output='pandas')


def _cyclic(data: pd.Series, period: int, names: list[str]) -> pd.DataFrame:
    d = data.astype(float)
    return pd.DataFrame(
        np.column_stack([np.sin(2 * np.pi * d / period),
                         np.cos(2 * np.pi * d / period)]),
        columns=names,
        index=data.index,
    )


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(transformers=[
        ('target_enc', TargetEncoder(),                                     ['station_id']),
        ('num',        StandardScaler(),                                    COLS_STANDARD),
        ('hour_cyc',   FunctionTransformer(lambda x: _cyclic(x, 24, ['hour_sin', 'hour_cos'])),   ['hour']),
        ('day_cyc',    FunctionTransformer(lambda x: _cyclic(x, 7,  ['day_sin',  'day_cos'])),    ['day_of_week']),
        ('bool',       'passthrough',                                       COLS_BINARY),
    ])