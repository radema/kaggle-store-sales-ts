import pytest
import pandas as pd
from src.data.loader import DataLoader


def test_dataloader_enforces_date_parsing(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("date,val\n2023-01-01,10")

    loader = DataLoader()
    df = loader.load(f)

    assert pd.api.types.is_datetime64_any_dtype(df["date"])


def test_dataloader_schema_override(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("id,val\n001,10")

    # Force id to be string
    loader = DataLoader(schema={"id": str})
    df = loader.load(f)

    assert pd.api.types.is_string_dtype(df["id"])
    assert df["id"].iloc[0] == "001"
