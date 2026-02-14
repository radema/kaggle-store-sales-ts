import pandas as pd
import numpy as np
from src.models.gnn.features import (
    compute_is_open_past,
    compute_is_open_future,
    compute_calendar_features,
)


def test_is_open_past():
    # Mock data
    sales = pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
            "store_nbr": [1, 1, 1],
            "sales": [0.0, 100.0, 50.0],
        }
    )
    transactions = pd.DataFrame(
        {
            "date": ["2013-01-01", "2013-01-02", "2013-01-03"],
            "store_nbr": [1, 1, 1],
            "transactions": [0, 10, 5],
        }
    )

    # Merge for convenience as implementation might expect a merged DF or handle separately
    # Let's assume the function takes them and aligns them
    is_open = compute_is_open_past(sales, transactions)

    # 2013-01-01 should be closed (0 sales, 0 transactions)
    assert is_open.iloc[0] == 0
    assert is_open.iloc[1] == 1
    assert is_open.iloc[2] == 1


def test_is_open_future():
    dates = pd.DataFrame({"date": pd.to_datetime(["2013-01-01", "2013-01-02"])})
    holidays = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01"]),
            "type": ["Holiday"],
            "locale": ["National"],
            "locale_name": ["Ecuador"],
            "transferred": [False],
        }
    )
    store_info = pd.Series({"city": "Quito", "state": "Pichincha", "store_nbr": 1})

    is_open = compute_is_open_future(dates["date"], holidays, store_info)

    # 2013-01-01 is a National Holiday, should be closed
    assert is_open.iloc[0] == 0
    # 2013-01-02 no holiday, should be open
    assert is_open.iloc[1] == 1


def test_calendar_features():
    dates = pd.to_datetime(["2013-01-15", "2013-01-31", "2013-02-01"])
    df = compute_calendar_features(pd.Series(dates))

    # 15th and 31st are paydays
    assert df.loc[0, "is_payday"] == 1
    assert df.loc[1, "is_payday"] == 1
    assert df.loc[2, "is_payday"] == 0

    # Check cyclical (dow, dom, month)
    # Jan 15, 2013 was a Tuesday (dow=1)
    expected_dow_sin = np.sin(2 * np.pi * 1 / 7)
    assert np.isclose(df.loc[0, "dow_sin"], expected_dow_sin)

    # Month is Jan (month=1, encoding uses month-1=0)
    assert df.loc[0, "month_sin"] == 0
    assert df.loc[0, "month_cos"] == 1
