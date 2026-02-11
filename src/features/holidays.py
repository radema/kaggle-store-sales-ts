from src.features.base import BaseTimeSeriesTransformer
import pandas as pd
import numpy as np


class HolidayTransformer(BaseTimeSeriesTransformer):
    """
    Flags holidays based on date and store location (National, Regional, Local).
    Handles transferred holidays.
    """

    def __init__(
        self,
        holidays_df=None,
        stores_df=None,
        holidays_path=None,
        stores_path=None,
        date_col="date",
        store_col="store_nbr",
    ):
        super().__init__()
        self.holidays_df = holidays_df
        self.stores_df = stores_df
        self.holidays_path = holidays_path
        self.stores_path = stores_path
        self.date_col = date_col
        self.store_col = store_col

    def _transform(self, X):
        X = X.copy()

        # Load data if paths are provided instead of DataFrames
        holidays = self.holidays_df
        if holidays is None and self.holidays_path:
            holidays = pd.read_csv(self.holidays_path)

        stores = self.stores_df
        if stores is None and self.stores_path:
            stores = pd.read_csv(self.stores_path)

        if holidays is None or stores is None:
            # If no data provided, return X unchanged or with warning?
            # For now, return X
            return X

        holidays = holidays.copy()
        stores = stores.copy()

        # Ensure dates are datetime
        holidays["date"] = pd.to_datetime(holidays["date"])

        # Filter: Remove days that were transferred (they are workdays)
        # However, check handling: the original date has transferred=True.
        # The new date has type='Transfer'.
        holidays = holidays[~holidays["transferred"]]

        # National Holidays
        national_holidays = holidays[holidays["locale"] == "National"].copy()

        # Regional Holidays
        regional_holidays = holidays[holidays["locale"] == "Regional"].copy()

        # Local Holidays
        local_holidays = holidays[holidays["locale"] == "Local"].copy()

        # We need to enrich X with store metadata (City, State) to match
        # If X already has city/state, great. If not, merge with stores.
        must_merge_stores = "city" not in X.columns or "state" not in X.columns

        if must_merge_stores:
            X_enriched = X.merge(stores, on=self.store_col, how="left")
        else:
            X_enriched = X.copy()

        # Initialize features
        X["is_holiday"] = False
        X["holiday_type"] = "WorkDay"

        # deduplicate holidays to avoid row duplication in merge
        national_dates = set(national_holidays["date"].unique())
        regional_holidays = regional_holidays.drop_duplicates(["date", "locale_name"])
        local_holidays = local_holidays.drop_duplicates(["date", "locale_name"])

        # Update X
        X["is_holiday"] = X[self.date_col].isin(national_dates)
        X["holiday_type"] = np.where(X["is_holiday"], "National", "WorkDay")

        # Flag Regional
        regional_holidays = regional_holidays.rename(
            columns={"locale_name": "state", "type": "type_reg"}
        )
        merged_reg = X_enriched.merge(
            regional_holidays[["date", "state", "type_reg"]],
            on=["date", "state"],
            how="left",
        )
        is_reg = merged_reg["type_reg"].notna()
        X.loc[is_reg.values, "is_holiday"] = True
        X.loc[is_reg.values, "holiday_type"] = "Regional"

        # Flag Local
        local_holidays = local_holidays.rename(
            columns={"locale_name": "city", "type": "type_loc"}
        )
        merged_loc = X_enriched.merge(
            local_holidays[["date", "city", "type_loc"]],
            on=["date", "city"],
            how="left",
        )
        is_loc = merged_loc["type_loc"].notna()
        X.loc[is_loc.values, "is_holiday"] = True
        X.loc[is_loc.values, "holiday_type"] = "Local"

        return X
