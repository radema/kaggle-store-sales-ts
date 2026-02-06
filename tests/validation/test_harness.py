import pandas as pd
import numpy as np
import pytest
from src.validation.harness import EvaluationSuite


def test_evaluation_suite_metrics():
    # Mock data
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2017-01-01", "2017-01-01", "2017-01-02", "2017-01-02"]
            ),
            "store_nbr": [1, 2, 1, 2],
            "family": ["BREAD", "BREAD", "BREAD", "DAIRY"],
            "sales": [10, 20, 15, 25],
        }
    )
    # Perfect predictions for store 1, slightly off for store 2
    y_pred = np.array([10, 21, 15, 24])

    suite = EvaluationSuite()
    report = suite.evaluate(df, y_pred)

    assert "global_rmsle" in report
    assert report["global_rmsle"] > 0

    # Store 1 should have 0 error
    assert report["per_store"].loc[1] == 0.0
    # Store 2 should have non-zero error
    assert report["per_store"].loc[2] > 0.0

    # Check structure
    assert "per_family" in report
    assert "detailed" in report
    assert len(report["detailed"]) == len(df)
    assert "residual" in report["detailed"].columns
