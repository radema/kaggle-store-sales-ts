import pandas as pd
import numpy as np


def compute_is_open_past(sales_df, transactions_df=None):
    """
    Identifies closed days in historical data.
    If sales is 0, we assume the store/item was closed or not traded.

    Args:
        sales_df: DataFrame with at least 'sales' column.
        transactions_df: Optional DataFrame with 'transactions'.

    Returns:
        pd.Series: Binary mask (1 for open, 0 for closed).
    """
    # Simple heuristic: if sales > 0, it was definitely open.
    # If sales == 0, we check transactions if available.
    is_open = (sales_df["sales"] > 0).astype(int)

    if transactions_df is not None:
        # If we have transactions but 0 sales, it's a weird data point,
        # but usually 0 sales & 0 transactions = closed.
        has_transactions = (transactions_df["transactions"] > 0).astype(int)
        is_open = (is_open | has_transactions).astype(int)

    return is_open


def compute_is_open_future(dates, holidays_df, store_info):
    """
    Predicts closure for future dates based on holiday schedule.

    Args:
        dates: pd.Series of datetime objects.
        holidays_df: DataFrame from holidays_events.csv.
        store_info: Series or dict with 'city' and 'state'.

    Returns:
        pd.Series: Binary mask (1 for open, 0 for closed).
    """
    # Initialize as open
    is_open = pd.Series(1, index=dates.index)

    # Pre-process dates to ensure they are datetime
    dates = pd.to_datetime(dates)
    holidays_df["date"] = pd.to_datetime(holidays_df["date"])

    # Filter valid holidays (not transferred)
    active_holidays = holidays_df[holidays_df["transferred"] == False]

    # 1. National Holidays
    national = active_holidays[active_holidays["locale"] == "National"]
    is_open[dates.isin(national["date"])] = 0

    # 2. Regional Holidays (State match)
    regional = active_holidays[
        (active_holidays["locale"] == "Regional")
        & (active_holidays["locale_name"] == store_info["state"])
    ]
    is_open[dates.isin(regional["date"])] = 0

    # 3. Local Holidays (City match)
    local = active_holidays[
        (active_holidays["locale"] == "Local")
        & (active_holidays["locale_name"] == store_info["city"])
    ]
    is_open[dates.isin(local["date"])] = 0

    # Special cases: In the Ecuador dataset, New Year (Jan 1) is always closed.
    # This is often captured in National, but we can be explicit if needed.
    is_open[dates.dt.month == 1] &= (dates.dt.day != 1).astype(int)

    return is_open
