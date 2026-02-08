from src.features.base import BaseTimeSeriesTransformer


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
            return X

        window_suffix = self.window if self.window else "global"
        column_name = f"level_{self.target_col}_{window_suffix}"

        if self.groupby:
            grouped = X.groupby(self.groupby)[self.target_col]
            if self.window:
                res = grouped.transform(
                    lambda x: (
                        x.shift(16).rolling(window=self.window, min_periods=1).mean()
                    )
                )
            else:
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
