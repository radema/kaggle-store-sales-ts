import pandas as pd
from typing import Optional
from src.features.base import BaseTimeSeriesTransformer
from src.utils.validation import validate_no_duplicates


class OilMerger(BaseTimeSeriesTransformer):
    """
    Merges oil price data and performs intentional imputation (interpolation).
    """

    def __init__(
        self,
        oil_df: Optional[pd.DataFrame] = None,
        oil_path: Optional[str] = None,
        method: Optional[str] = None,  # Enforce via check
    ):
        super().__init__()
        self.oil_df = oil_df
        self.oil_path = oil_path
        self.method = method

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.method is None:
            raise ValueError(
                "OilMerger requires an explicit 'method' parameter (e.g., 'linear', 'ffill') for intentional imputation."
            )
        oil = self.oil_df
        if oil is None and self.oil_path:
            self.logger.info(f"Loading oil from {self.oil_path}")
            oil = pd.read_csv(self.oil_path)
            oil["date"] = pd.to_datetime(oil["date"])

        if oil is None:
            self.logger.warning("No oil data provided. Skipping.")
            return X

        # Oil might have missing dates or missing values on existing dates
        # Ensure we have a unique date index for interpolation
        validate_no_duplicates(oil, subset=["date"])
        oil = oil.set_index("date").sort_index()

        # Intentional Imputation on the oil series itself before merging
        nulls_before = oil["dcoilwtico"].isnull().sum()
        if nulls_before > 0 or True:  # Always attempt to fill gaps
            self.logger.info(f"Interpolating oil prices using method: {self.method}")
            # Resample oil to daily to fill date gaps if any, then interpolate
            oil = oil.resample("D").asfreq()
            if self.method == "linear":
                oil["dcoilwtico"] = oil["dcoilwtico"].interpolate(method="linear")
            else:
                oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()

        # Merge
        row_count_before = len(X)
        X = X.merge(oil, left_on="date", right_index=True, how="left")

        if len(X) != row_count_before:
            raise ValueError("Row count changed during Oil merge!")

        # Final check: if still null (e.g. before first oil record), ffill/bfill
        final_nulls = X["dcoilwtico"].isnull().sum()
        if final_nulls > 0:
            self.logger.info(
                f"Filling {final_nulls} remaining null oil prices with ffill/bfill"
            )
            X["dcoilwtico"] = X["dcoilwtico"].ffill().bfill()

        return X


class StoreMerger(BaseTimeSeriesTransformer):
    """
    Merges store metadata (city, state, type, cluster).
    """

    def __init__(
        self,
        stores_df: Optional[pd.DataFrame] = None,
        stores_path: Optional[str] = None,
    ):
        super().__init__()
        self.stores_df = stores_df
        self.stores_path = stores_path

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        stores = self.stores_df
        if stores is None and self.stores_path:
            self.logger.info(f"Loading stores from {self.stores_path}")
            stores = pd.read_csv(self.stores_path)

        if stores is None:
            return X

        validate_no_duplicates(stores, subset=["store_nbr"])

        row_count_before = len(X)
        X = X.merge(stores, on="store_nbr", how="left")

        if len(X) != row_count_before:
            raise ValueError("Row count changed during Store merge!")

        return X
