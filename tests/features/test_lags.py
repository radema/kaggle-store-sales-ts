import pandas as pd
import numpy as np
from src.features.lags import LagTransformer


def test_lag_transformer_basic():
    # Simple single series
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})

    transformer = LagTransformer(lags=[1, 2], column="value")
    X_transformed = transformer.fit_transform(df)

    assert "lag_1_value" in X_transformed.columns
    assert "lag_2_value" in X_transformed.columns

    # Check values
    assert np.isnan(X_transformed.iloc[0]["lag_1_value"])
    assert X_transformed.iloc[1]["lag_1_value"] == 1.0


def test_lag_transformer_multiseries_groupby():
    # Creating 2 groups: Store A and Store B
    # Store A: 3 rows, Store B: 3 rows
    data = {"store": ["A", "A", "A", "B", "B", "B"], "value": [10, 20, 30, 40, 50, 60]}
    df = pd.DataFrame(data)

    # We want to lag 'value' by 1, grouped by 'store'
    transformer = LagTransformer(lags=[1], groupby=["store"], column="value")
    X_transformed = transformer.fit_transform(df)

    # Verify Store A (first 3 rows) - standard lag behavior
    assert np.isnan(X_transformed.iloc[0]["lag_1_value"])
    assert X_transformed.iloc[1]["lag_1_value"] == 10.0

    # Verify Store B (next 3 rows) - CRITICAL CHECK
    # The first row of Store B (index 3) should be NaN, NOT the last value of Store A (30)
    assert np.isnan(X_transformed.iloc[3]["lag_1_value"])
    assert X_transformed.iloc[4]["lag_1_value"] == 40.0


def test_lag_transformer_multiple_lags():
    df = pd.DataFrame({"value": range(10)})
    transformer = LagTransformer(lags=[1, 7], column="value")
    X = transformer.fit_transform(df)

    assert "lag_1_value" in X.columns
    assert "lag_7_value" in X.columns

    assert np.isnan(X.iloc[6]["lag_7_value"])
    assert X.iloc[7]["lag_7_value"] == 0.0
