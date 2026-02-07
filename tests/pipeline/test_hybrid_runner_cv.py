import pytest
import pandas as pd
import numpy as np
from src.pipeline.runners.hybrid import HybridRunner
from unittest.mock import patch, MagicMock


@pytest.fixture
def sample_data():
    dates = pd.date_range("2017-01-01", periods=100)
    df = pd.DataFrame(
        {
            "date": dates,
            "store_nbr": [1] * 100,
            "family": ["BREAD"] * 100,
            "sales": np.random.rand(100) * 100,
        }
    )
    return df


@patch("src.pipeline.runners.hybrid.load_config")
def test_hybrid_runner_run_cv(mock_load_config, sample_data):
    # Setup config
    mock_load_config.return_value = {
        "pipeline": {"steps": []},
        "model": {
            "trend_estimator": {"class": "sklearn.linear_model.Ridge", "params": {}},
            "residual_estimator": {"class": "sklearn.linear_model.Ridge", "params": {}},
            "feature_selector": ["store_nbr"],
        },
        "validation": {
            "train_days": 60,
            "val_days": 10,
            "n_folds": 2,
            "target_transform": "log1p",
        },
    }

    runner = HybridRunner("dummy_path")
    runner.data = {"train": sample_data}

    # Execute
    metrics = runner.run_cv()

    # Verify
    assert "global_rmsle" in metrics
    assert len(runner.oof_detailed) == 20  # 2 folds * 10 val_days
    assert "sales_pred" in runner.oof_detailed.columns
