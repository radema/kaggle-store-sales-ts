from src.features.base import BaseTimeSeriesTransformer
import pandas as pd
import numpy as np


class LevelTransformer(BaseTimeSeriesTransformer):
    """
    Computes 'Levels' (Target Encoding / Moving Averages) with a strict 16-day lag
    to respect the inference horizon.

    Parameters:
    -----------
    window : int, optional
        Window size for rolling mean. If None, computes global expanding mean.
    target_col : str
        Column to compute levels for (default: 'sales').
    groupby : list, optional
        Columns to group by (default: ['store_nbr', 'family']).
    fallback : float
        Value to use when no historical data is available (default: 0.0).
    """

    def __init__(self, window=None, target_col="sales", groupby=None, fallback=0.0):
        super().__init__()
        self.window = window
        self.target_col = target_col
        self.groupby = groupby or ["store_nbr", "family"]
        self.fallback = fallback

    def transform(self, X):
        X = X.copy()

        # Check if target column exists
        if self.target_col not in X.columns:
            # During inference for future dates, 'sales' might not be in X if not concatenated
            # However, the Runner usually handles train+test concatenation.
            # If still missing, we return X as is (or with fallback zeros)
            # For now, let's assume it's there.
            return X

        window_suffix = self.window if self.window else "global"
        column_name = f"level_{self.target_col}_{window_suffix}"

        if self.groupby:
            grouped = X.groupby(self.groupby)[self.target_col]
            if self.window:
                # Rolling window of size self.window, shifted by 16
                # min_periods=1 to provide a level as soon as we have 1 sample at t-16
                res = grouped.transform(
                    lambda x: (
                        x.shift(16).rolling(window=self.window, min_periods=1).mean()
                    )
                )
            else:
                # Global Mean up to t-16
                res = grouped.transform(
                    lambda x: x.shift(16).expanding(min_periods=1).mean()
                )
        else:
            if self.window:
                res = (
                    X[self.target_col]
                    .shift(16)
                    .rolling(window=self.window, min_periods=1)
                    .mean()
                )
            else:
                res = X[self.target_col].shift(16).expanding(min_periods=1).mean()

        X[column_name] = res.fillna(self.fallback)

        return X
