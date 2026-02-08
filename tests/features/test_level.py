import pandas as pd
import numpy as np
from src.features.level import LevelTransformer


def test_level_transformer_basic():
    # Sequence of sales: 0, 1, 2, 3, 4, ...
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
    # The default behavior of shift(16) followed by rolling(3, min_periods=1).mean()
    # will result in NaN for indices < 16.
    # We use fillna(0) in the transformer.
    assert X.iloc[15][col_name] == 0.0


def test_level_transformer_global_mean():
    dates = pd.date_range("2021-01-01", periods=30)
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * 30,
            "family": ["A"] * 30,
            "sales": [10.0] * 10 + [20.0] * 20,
        }
    )

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

    assert X.iloc[0][col_name] == 0.0
