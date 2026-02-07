import pytest
from unittest.mock import MagicMock, patch
from src.pipeline.runners.hybrid import HybridRunner
import pandas as pd


@patch("src.pipeline.runners.hybrid.load_config")
@patch("src.pipeline.runners.hybrid.DataLoader")
def test_hybrid_runner_init(mock_loader, mock_load_config):
    # Setup
    mock_load_config.return_value = {"run_name": "test_run"}

    # Execute
    runner = HybridRunner("dummy_path")

    # Verify
    assert runner.run_name == "test_run"
    mock_load_config.assert_called_once_with("dummy_path")


@patch("src.pipeline.runners.hybrid.load_config")
@patch("src.pipeline.runners.hybrid.DataLoader")
def test_hybrid_runner_load_data(mock_loader_cls, mock_load_config):
    # Setup
    mock_load_config.return_value = {
        "data": {"train_path": "train.csv", "test_path": "test.csv"}
    }
    mock_loader_instance = mock_loader_cls.return_value
    mock_load_result = pd.DataFrame({"a": [1]})
    mock_loader_instance.load.return_value = mock_load_result

    runner = HybridRunner("dummy_path")

    # Execute
    data = runner.load_data()

    # Verify
    assert "train" in data
    assert "test" in data
    assert mock_loader_instance.load.call_count == 2
