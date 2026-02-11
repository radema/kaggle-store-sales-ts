from src.features.base import BaseTimeSeriesTransformer
import pandas as pd


class RollingWindowTransformer(BaseTimeSeriesTransformer):
    """
    Computes rolling window statistics (e.g., mean, std) respecting group boundaries.
    """

    def __init__(
        self,
        window=3,
        columns=None,
        groupby=None,
        stats=None,
        center=False,
        min_periods=None,
    ):
        super().__init__()
        self.window = window
        self.columns = columns or []
        self.groupby = groupby
        self.stats = stats or ["mean"]
        self.center = center
        self.min_periods = min_periods

    def _transform(self, X):
        X = X.copy()

        if not self.columns:
            return X

        for col in self.columns:
            if col not in X.columns:
                continue

            # Compute rolling object
            if self.groupby:
                # GroupBy -> Rolling
                grouped = X.groupby(self.groupby)[col]
                roller = grouped.rolling(
                    window=self.window, center=self.center, min_periods=self.min_periods
                )
            else:
                roller = X[col].rolling(
                    window=self.window, center=self.center, min_periods=self.min_periods
                )

            # Aggregate stats
            # For multiple stats, agg returns a DataFrame with columns as stat names
            # For 1 stat, it returns a Series if we pass 'mean', or DataFrame if ['mean']?
            # Let's be explicit and iterate applied functions or use agg list

            # If we pass list of stats to agg, we get DataFrame with those columns
            # But groupby rolling result index is MultiIndex.

            agg_res = roller.agg(self.stats)  # DataFrame or Series

            # If agg_res is Series (single stat not in list?), promote to DataFrame
            if isinstance(agg_res, pd.Series):
                agg_res = agg_res.to_frame(name=self.stats[0])

            # Iterate stats to assign columns
            for stat in self.stats:
                col_name = f"rolling_{col}_{stat}_{self.window}"

                if self.groupby:
                    # agg_res has MultiIndex (group keys..., original_index)
                    # We need to drop group keys levels to align with X by original index

                    # Ensure alignment:
                    # 1. Reset index levels corresponding to groups
                    # 2. Set index to the last level (original index)
                    # 3. Sort by index to match X (if X is not sorted by group) or just use index alignment

                    series = agg_res[stat]
                    # Drop the grouping levels (0 to len(groupby)-1)
                    # The last level is the original index
                    series_reset = series.droplevel(list(range(len(self.groupby))))

                    # Assign using index alignment
                    X[col_name] = series_reset
                else:
                    X[col_name] = agg_res[stat]

        return X
