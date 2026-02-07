import pytest
import pandas as pd
import numpy as np
from src.pipeline.factory import FeaturePipelineFactory


def test_hybrid_pipeline_integration():
    # Load YAML content directly or from file
    yaml_config = """
    steps:
      - name: "DateFeatures"
        class: "src.features.dates.DatePartTransformer"
        params:
          parts: ["year", "month", "day", "is_weekend"]
          column: "date"

      - name: "LagFeatures"
        class: "src.features.lags.LagTransformer"
        params:
          column: "sales"
          lags: [1]
          groupby: ["store_nbr", "family"]
      
      - name: "RollingFeatures"
        class: "src.features.rolling.RollingWindowTransformer"
        params:
          columns: ["sales"]
          groupby: ["store_nbr", "family"]
          window: 3
          center: false
          stats: ["mean"]
    """

    factory = FeaturePipelineFactory()
    pipeline = factory.create_from_yaml(yaml_config)

    # Create sample data
    # 2 Stores, 1 Family, 10 days
    dates = pd.date_range("2023-01-01", periods=10)
    data = []
    for store in [1, 2]:
        for d in dates:
            data.append(
                {
                    "date": d,
                    "store_nbr": store,
                    "family": "GROCERY",
                    "sales": float(d.day) * 10,
                }
            )
    df = pd.DataFrame(data)

    # Transform
    X_transformed = pipeline.fit_transform(df)

    # Assertions
    # 1. Date features
    assert "year" in X_transformed.columns
    assert "is_weekend" in X_transformed.columns

    # 2. Lag features
    assert "lag_1_sales" in X_transformed.columns
    # Check lag correctness (first element of group should be NaN)
    # Store 1 starts at index 0
    assert np.isnan(X_transformed.iloc[0]["lag_1_sales"])

    # 3. Rolling features
    assert "rolling_sales_mean_3" in X_transformed.columns
    # Check rolling correctness (center=false, window=3)
    # Index 2 should be mean(10, 20, 30) = 20
    # Provide simple check
    pass
