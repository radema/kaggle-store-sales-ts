import pandas as pd
import numpy as np
import pytest
from src.validation.splitters import SlidingWindowTS


def test_sliding_window_splits_overlapping():
    df = pd.DataFrame(
        {"date": pd.date_range("2017-01-01", periods=100), "sales": range(100)}
    )
    # train 40 days, val 10 days, 2 folds, step 5 (partially overlap)
    splitter = SlidingWindowTS(train_days=40, val_days=10, n_folds=2, step_days=5)
    splits = list(splitter.split(df))

    assert len(splits) == 2

    # Fold 1 (last fold):
    # End is day 99. Val is 90-99. Train is 50-89.
    train_idx1, val_idx1 = splits[0]
    assert len(val_idx1) == 10
    assert len(train_idx1) == 40
    assert val_idx1[-1] == 99

    # Fold 2 (second to last):
    # End is 99 - step(5) = 94. Val is 85-94. Train is 45-84.
    train_idx2, val_idx2 = splits[1]
    assert val_idx2[-1] == 94
    assert len(val_idx2) == 10


def test_sliding_window_insufficient_data():
    df = pd.DataFrame(
        {"date": pd.date_range("2017-01-01", periods=30), "sales": range(30)}
    )
    # train 20, val 15 (total 35) > 30. Should return empty or fewer folds.
    splitter = SlidingWindowTS(train_days=20, val_days=15, n_folds=1)
    splits = list(splitter.split(df))
    assert len(splits) == 0
