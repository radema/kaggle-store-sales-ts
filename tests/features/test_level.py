import pytest
import pandas as pd
import numpy as np
from src.features.level import LevelTransformer


def test_level_transformer_basic():
    # Sequence of sales: 0, 1, 2, 3, 4, ...
    # We want to check value at index 20 (day 20)
    # With shift=16 and window=3:
    # It should look at sales at indices [20-16, 20-17, 20-18] = [4, 3, 2]
    # Sales values: 4, 3, 2 -> mean = 3.0

    dates = pd.date_range("2021-01-01", periods=30)
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * 30,
            "family": ["A"] * 30,
            "sales": np.arange(30).astype(float),
        }
    )

    transformer = LevelTransformer(
        window=3, target_col="sales", groupby=["store_nbr", "family"]
    )

    X = transformer.fit_transform(df)

    col_name = "level_sales_3"
    assert col_name in X.columns

    # At index 20
    # Sales[20-16] = Sales[4] = 4.0
    # Sales[20-17] = Sales[3] = 3.0
    # Sales[20-18] = Sales[2] = 2.0
    # Mean = 3.0
    assert X.iloc[20][col_name] == 3.0

    # At index 15 (less than 16 days since start)
    # Shifted values are NaN
    assert np.isnan(X.iloc[15][col_name])


def test_level_transformer_global_mean():
    # If window is None, it should compute the global mean up to t-16
    dates = pd.date_range("2021-01-01", periods=30)
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * 30,
            "family": ["A"] * 30,
            "sales": [10.0] * 10 + [20.0] * 20,
        }
    )

    # At t=20, t-16 is t=4. Sales[0:5] are all 10.0. Mean is 10.0
    # At t=30? No, let's say t=25. t-16 is t=9. Sales[0:10] are 10.0. Mean is 10.0.
    # At t=28. t-16 is t=12. Sales[0:10] are 10.0, Sales[10:13] are 20.0.
    # Count: 10*10 + 3*20 = 100 + 60 = 160. Mean = 160/13 = 12.307

    transformer = LevelTransformer(
        window=None,  # Global
        target_col="sales",
        groupby=["store_nbr", "family"],
    )

    X = transformer.fit_transform(df)
    col_name = "level_sales_global"

    # At index 20 (Day 21)
    # Available data (shifted 16): indices 0, 1, 2, 3, 4
    # Sales[0...4] = 10.0. Mean = 10.0
    assert X.iloc[20][col_name] == 10.0


def test_level_transformer_fallback():
    # Check fallback value (0)
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2021-01-01", "2021-01-02"]),
            "store_nbr": [1, 1],
            "family": ["A", "A"],
            "sales": [10.0, 20.0],
        }
    )

    transformer = LevelTransformer(
        window=30, target_col="sales", groupby=["store_nbr", "family"], fallback=0.0
    )

    X = transformer.fit_transform(df)
    col_name = "level_sales_30"

    # No data available at t-16
    # If we use raw expansion, it's NaN.
    # But the spec says "fallback to a constant 0 if no data is available".
    # This usually means after computing rolling, fillna(0)
    assert X.iloc[0][col_name] == 0.0
