import pandas as pd
from typing import List, Dict, Any
from src.features.base import BaseTimeSeriesTransformer
from src.utils.validation import validate_no_duplicates


class DateGridTransformer(BaseTimeSeriesTransformer):
    """
    Expands the dataframe to ensure a complete date grid for every group.
    Fills missing values based on an explicit fill_map.
    """

    def __init__(
        self,
        groupby: List[str] = ["store_nbr", "family"],
        date_col: str = "date",
        fill_map: Dict[str, Any] = {"sales": 0.0},
    ):
        super().__init__()
        self.groupby = groupby
        self.date_col = date_col
        self.fill_map = fill_map

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"Expanding date grid grouped by {self.groupby}")

        # 1. Identify all unique dates and unique groups
        all_dates = pd.date_range(
            X[self.date_col].min(), X[self.date_col].max(), freq="D"
        )

        # 2. Get unique combinations of groupby columns
        unique_groups = X[self.groupby].drop_duplicates()

        # 3. Create the cartesian product grid
        # Cross join unique_groups with all_dates
        # Newer pandas way:
        grid = (
            unique_groups.assign(key=1)
            .merge(pd.DataFrame({self.date_col: all_dates, "key": 1}), on="key")
            .drop("key", axis=1)
        )

        self.logger.info(f"Created full grid with {len(grid)} rows.")

        # 4. Merge original data into the grid
        # IMPORTANT: ensure X has no duplicates on keys before merge
        validate_no_duplicates(X, subset=[self.date_col] + self.groupby)

        X_full = grid.merge(X, on=[self.date_col] + self.groupby, how="left")

        # 5. Intentional Imputation
        for col, val in self.fill_map.items():
            if col in X_full.columns:
                null_count = X_full[col].isnull().sum()
                if null_count > 0:
                    self.logger.info(
                        f"Filling {null_count} missing values in '{col}' with {val}"
                    )
                    X_full[col] = X_full[col].fillna(val)

        # For other columns not in fill_map, they remain null, which might be correct
        # (e.g. metadata that will be joined later).
        # But if they were already there, we might want to ffill them (e.g. cluster)

        return X_full
