from src.features.base import BaseTimeSeriesTransformer
import pandas as pd


class DatePartTransformer(BaseTimeSeriesTransformer):
    """
    Extracts date parts from the index or a specific column for time series forecasting.
    Includes support for periodic features and boolean flags like 'is_weekend'.
    """

    def __init__(self, parts=None, column=None):
        """
        Args:
            parts (list): List of date parts to extract. Default: ['year', 'month', 'day'].
            column (str): Name of the column to extract dates from. If None, uses index.
        """
        super().__init__()
        self.parts = parts or ["year", "month", "day"]
        self.column = column

    def transform(self, X):
        X = X.copy()

        # Determine source
        if self.column:
            if self.column not in X.columns:
                raise ValueError(f"Column '{self.column}' not found in DataFrame.")
            dates = pd.to_datetime(X[self.column])
        else:
            if not isinstance(X.index, pd.DatetimeIndex):
                try:
                    dates = pd.to_datetime(X.index)
                except Exception as e:
                    # Fallback or strict error
                    # If index can be converted, great, else error.
                    raise ValueError(
                        f"Index is not DatetimeIndex and no column specified. Error: {e}"
                    )
            else:
                dates = X.index

        for part in self.parts:
            # When dates is a Series (from column), we need .dt accessor
            # When dates is DatetimeIndex, we can access properties directly
            # Normalize to work with both
            d = dates.dt if isinstance(dates, pd.Series) else dates

            if part == "year":
                X["year"] = d.year.astype("int16")
            elif part == "month":
                X["month"] = d.month.astype("int8")
            elif part == "day":
                X["day"] = d.day.astype("int8")
            elif part == "dayofweek":
                X["dayofweek"] = d.dayofweek.astype("int8")
            elif part == "dayofyear":
                X["dayofyear"] = d.dayofyear.astype("int16")
            elif part == "week":
                # .isocalendar().week returns UInt32, safe to cast to int8/int16
                X["week"] = d.isocalendar().week.astype("int8")
            elif part == "quarter":
                X["quarter"] = d.quarter.astype("int8")
            elif part == "is_weekend":
                X["is_weekend"] = (d.dayofweek >= 5).astype("bool")
            elif part == "is_month_start":
                X["is_month_start"] = d.is_month_start.astype("bool")
            elif part == "is_month_end":
                X["is_month_end"] = d.is_month_end.astype("bool")
            elif part == "is_year_start":
                X["is_year_start"] = d.is_year_start.astype("bool")
            elif part == "is_year_end":
                X["is_year_end"] = d.is_year_end.astype("bool")
            elif part == "is_wage_day":
                # Ecuadorian Wage days: 15th and last day of month
                X["is_wage_day"] = ((d.day == 15) | (d.is_month_end)).astype("bool")

        return X
