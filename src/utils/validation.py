import pandas as pd
from typing import List, Optional, Union


def validate_no_duplicates(
    df: pd.DataFrame, subset: Optional[Union[str, List[str]]] = None
) -> None:
    """
    Validates that there are no duplicate rows in the DataFrame,
    optionally checking only a subset of columns.
    Raises ValueError if duplicates are found.
    """
    if df.duplicated(subset=subset).any():
        if subset:
            msg = f"Data contains duplicates based on columns: {subset}"
        else:
            msg = "Data contains duplicate rows."
        raise ValueError(msg)


def validate_shape(df: pd.DataFrame, min_rows: int = 1) -> None:
    """
    Validates that the DataFrame has at least min_rows.
    """
    if df.empty or len(df) < min_rows:
        raise ValueError(f"DataFrame is empty or has fewer than {min_rows} rows.")


def validate_schema(df: pd.DataFrame, expected_cols: List[str]) -> None:
    """
    Validates that the DataFrame contains the expected columns.
    """
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def validate_no_nulls(df: pd.DataFrame, cols: Optional[List[str]] = None) -> None:
    """
    Validates that there are no null values in the specified columns (or all if None).
    """
    check_df = df[cols] if cols else df
    if check_df.isnull().any().any():
        null_counts = check_df.isnull().sum()
        detailed_msg = null_counts[null_counts > 0].to_dict()
        raise ValueError(f"Data contains null values: {detailed_msg}")
