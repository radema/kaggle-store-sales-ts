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

    def __init__(
        self,
        window=None,
        target_col="sales",
        groupby=None,
        fallback=0.0,
        log_transform=False,
    ):
        super().__init__()
        self.window = window
        self.target_col = target_col
        self.groupby = groupby or ["store_nbr", "family"]
        self.fallback = fallback
        self.log_transform = log_transform

    def _transform(self, X):
        X = X.copy()

        # Check if target column exists
        if self.target_col not in X.columns:
            return X

        window_suffix = self.window if self.window else "global"
        column_name = f"level_{self.target_col}_{window_suffix}"

        if self.groupby:
            grouped = X.groupby(self.groupby)[self.target_col]

            # Helper to compute mean in the requested space
            def compute_level(s):
                target = s.shift(16)
                if self.log_transform:
                    import numpy as np

                    target = np.log1p(target)

                if self.window:
                    return target.rolling(window=self.window, min_periods=1).mean()
                return target.expanding(min_periods=1).mean()

            res = grouped.transform(compute_level)
        else:
            target = X[self.target_col].shift(16)
            if self.log_transform:
                import numpy as np

                target = np.log1p(target)

            if self.window:
                res = target.rolling(window=self.window, min_periods=1).mean()
            else:
                res = target.expanding(min_periods=1).mean()

        X[column_name] = res.fillna(self.fallback)

        # No extra transformation needed here, as it's now done before the mean
        return X
