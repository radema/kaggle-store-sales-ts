import pytest
import pandas as pd
import numpy as np
from src.features.rolling import RollingWindowTransformer


def test_rolling_basic():
    # Simple sequence
    df = pd.DataFrame({"val": [1, 2, 3, 4, 5]})

    # Rolling mean window=3
    # center=False (default): index 2 reflects mean([1,2,3]) = 2
    transformer = RollingWindowTransformer(window=3, columns=["val"], stats=["mean"])
    X = transformer.fit_transform(df)

    col_name = "rolling_val_mean_3"
    assert col_name in X.columns
    assert np.isnan(X.iloc[1][col_name])  # Not enough data
    assert X.iloc[2][col_name] == 2.0
    assert X.iloc[4][col_name] == 4.0  # mean(3,4,5) = 4


def test_rolling_lookahead_prevention():
    # Check center=False default
    df = pd.DataFrame({"val": [10, 20, 30]})
    transformer = RollingWindowTransformer(window=2, columns=["val"], stats=["mean"])
    X = transformer.fit_transform(df)

    # At index 0, result should be NaN (cannot see future)
    # At index 1, result is mean(10, 20) = 15
    assert np.isnan(X.iloc[0]["rolling_val_mean_2"])
    assert X.iloc[1]["rolling_val_mean_2"] == 15.0


def test_rolling_multiseries_boundary():
    # Store A: [100, 100, 100], Store B: [5, 5, 5]
    # If leakage occurs, early Store B values might average with late Store A values
    df = pd.DataFrame(
        {"store": ["A"] * 3 + ["B"] * 3, "sales": [100.0, 100.0, 100.0, 5.0, 5.0, 5.0]}
    )

    transformer = RollingWindowTransformer(
        window=3, groupby=["store"], columns=["sales"], stats=["mean"]
    )
    X = transformer.fit_transform(df)

    col = "rolling_sales_mean_3"

    # Store B first element (index 3) -> NaN (window=3)
    assert np.isnan(X.iloc[3][col])
    # Store B second element (index 4) -> NaN
    assert np.isnan(X.iloc[4][col])
    # Store B third element (index 5) -> 5.0 (mean of 5,5,5)
    # If boundary crossing happened: mean(100, 5, 5) -> ~36.6 or mean(100, 100, 5) depending on implementation
    assert X.iloc[5][col] == 5.0
