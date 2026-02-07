from src.features.base import BaseTimeSeriesTransformer
import pandas as pd
import numpy as np


class LagTransformer(BaseTimeSeriesTransformer):
    """
    Creates lagged features respecting time series boundaries (e.g., Store x Family).
    """

    def __init__(self, lags=None, groupby=None, column=None):
        """
        Args:
            lags (list[int]): List of lag periods (e.g., [1, 7]).
            groupby (list[str] or str): Columns to group by before shifting.
            column (str): The column to create lags for.
        """
        super().__init__()
        self.lags = lags or []
        self.groupby = groupby
        self.column = column

    def transform(self, X):
        X = X.copy()

        if not self.column:
            raise ValueError("LagTransformer requires a 'column' argument.")

        target_series = X[self.column]

        # If groupby is provided, we must shift within groups
        if self.groupby:
            # GroupBy object .shift() operates within groups efficiently
            grouped = X.groupby(self.groupby)[self.column]

            for lag in self.lags:
                lag_col_name = f"lag_{lag}_{self.column}"
                X[lag_col_name] = grouped.shift(lag)
        else:
            # Single series shift
            for lag in self.lags:
                lag_col_name = f"lag_{lag}_{self.column}"
                X[lag_col_name] = target_series.shift(lag)

        return X
