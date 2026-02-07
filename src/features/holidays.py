from src.features.base import BaseTimeSeriesTransformer
import pandas as pd
import numpy as np


class HolidayTransformer(BaseTimeSeriesTransformer):
    """
    Flags holidays based on date and store location (National, Regional, Local).
    Handles transferred holidays.
    """

    def __init__(
        self, holidays_df=None, stores_df=None, date_col="date", store_col="store_nbr"
    ):
        super().__init__()
        self.holidays_df = holidays_df
        self.stores_df = stores_df
        self.date_col = date_col
        self.store_col = store_col

    def transform(self, X):
        X = X.copy()

        # Preprocessing
        # Filter out transferred holidays from the holidays table (they are workdays)
        # Note: The 'Transfer' type event is the NEW date.
        # So we keep type='Transfer' but remove rows where transferred=True

        # If holidays_df is a path, load it? For now assume DataFrame.
        holidays = self.holidays_df.copy()
        stores = self.stores_df.copy()

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

        # 1. Flag National
        # Merge X with national holidays on date
        # If match -> True

        # Vectorized approach:
        # Create a set of national dates
        national_dates = set(national_holidays["date"].unique())
        X.loc[X[self.date_col].isin(national_dates), "is_holiday"] = True
        X.loc[X[self.date_col].isin(national_dates), "holiday_type"] = "National"

        # 2. Flag Regional
        # Match on Date AND State
        # Rename columns for merge
        regional_holidays = regional_holidays.rename(
            columns={"locale_name": "state", "type": "type_reg"}
        )
        merged_reg = X_enriched.merge(
            regional_holidays[["date", "state", "type_reg"]],
            on=["date", "state"],
            how="left",
        )
        # If type_reg is not null, it's a regional holiday
        is_reg = merged_reg["type_reg"].notna()
        X.loc[is_reg, "is_holiday"] = True
        X.loc[is_reg, "holiday_type"] = "Regional"

        # 3. Flag Local
        # Match on Date AND City
        local_holidays = local_holidays.rename(
            columns={"locale_name": "city", "type": "type_loc"}
        )
        merged_loc = X_enriched.merge(
            local_holidays[["date", "city", "type_loc"]],
            on=["date", "city"],
            how="left",
        )
        is_loc = merged_loc["type_loc"].notna()
        X.loc[is_loc, "is_holiday"] = True
        X.loc[is_loc, "holiday_type"] = "Local"

        return X
