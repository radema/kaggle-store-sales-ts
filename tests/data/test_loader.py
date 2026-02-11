import pandas as pd
import pytest
from src.data.loader import DataLoader
from src.utils.logging import get_logger


def test_load_unified(tmp_path):
    # Setup dummy data
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"

    train_df = pd.DataFrame(
        {
            "id": [1, 2],
            "date": ["2013-01-01", "2013-01-02"],
            "store_nbr": [1, 1],
            "family": ["GROCERY", "GROCERY"],
            "sales": [10.0, 20.0],
            "onpromotion": [0, 1],
        }
    )

    test_df = pd.DataFrame(
        {
            "id": [3, 4],
            "date": ["2013-01-03", "2013-01-04"],
            "store_nbr": [1, 1],
            "family": ["GROCERY", "GROCERY"],
            "onpromotion": [0, 0],
        }
    )

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    config = {"data_sources": {"train": str(train_path), "test": str(test_path)}}

    loader = DataLoader()  # Assuming no logger injection needed in ctor yet
    unified_df = loader.load_unified(config)

    assert len(unified_df) == 4
    assert "split" in unified_df.columns
    assert unified_df[unified_df["split"] == "train"].shape[0] == 2
    assert unified_df[unified_df["split"] == "test"].shape[0] == 2
    assert unified_df["sales"].isna().sum() == 2  # test set has NaN sales naturally

    # Check date conversion
    assert pd.api.types.is_datetime64_any_dtype(unified_df["date"])


def test_load_metadata(tmp_path):
    # Setup dummy metadata
    oil_path = tmp_path / "oil.csv"
    pd.DataFrame({"date": ["2013-01-01"], "dcoilwtico": [90.0]}).to_csv(
        oil_path, index=False
    )

    config = {"data_sources": {"oil": str(oil_path)}}

    loader = DataLoader()
    dfs = loader.load_metadata(config)

    assert "oil" in dfs
    assert len(dfs["oil"]) == 1
