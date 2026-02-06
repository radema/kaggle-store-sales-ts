import pandas as pd
import numpy as np
from src.features.impute import TimeSeriesImputer


def test_imputer_ffill():
    df = pd.DataFrame({"val": [1, np.nan, 2]})
    imputer = TimeSeriesImputer(method="ffill", columns=["val"])
    res = imputer.fit_transform(df)
    assert res["val"].tolist() == [1.0, 1.0, 2.0]


def test_imputer_fill_value():
    df = pd.DataFrame({"val": [1, np.nan, 2]})
    imputer = TimeSeriesImputer(method="constant", value=0, columns=["val"])
    res = imputer.fit_transform(df)
    assert res["val"].tolist() == [1.0, 0.0, 2.0]


def test_imputer_interpolate():
    df = pd.DataFrame({"val": [1.0, np.nan, 3.0]})
    imputer = TimeSeriesImputer(method="interpolate", columns=["val"])
    res = imputer.fit_transform(df)
    assert res["val"].tolist() == [1.0, 2.0, 3.0]
