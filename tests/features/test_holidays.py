import pytest
import pandas as pd
from src.features.holidays import HolidayTransformer


@pytest.fixture
def mock_holidays():
    return pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2023-01-01", "2023-01-15", "2023-01-20", "2023-01-22"]
            ),
            "type": ["Holiday", "Holiday", "Transfer", "Holiday"],
            "locale": ["National", "Local", "Regional", "Local"],
            "locale_name": ["Ecuador", "Manta", "Manabi", "Quito"],
            "description": ["New Year", "Cantonal", "Provincial", "Capital"],
            "transferred": [False, False, False, True],  # 2023-01-22 was transferred
        }
    )


@pytest.fixture
def mock_stores():
    return pd.DataFrame(
        {
            "store_nbr": [1, 2],
            "city": ["Quito", "Manta"],
            "state": ["Pichincha", "Manabi"],
        }
    )


def test_holiday_national(mock_holidays, mock_stores):
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-01", "2023-01-02"]),
            "store_nbr": [1, 2],  # Both stores
        }
    )

    transformer = HolidayTransformer(holidays_df=mock_holidays, stores_df=mock_stores)
    X = transformer.fit_transform(df)

    # 2023-01-01 is National Holiday -> Should be holiday for both
    # Row 0: Store 1, Date 2023-01-01
    assert X.iloc[0]["is_holiday"] == True
    # But wait, date is 2023-01-01 for ONE row.
    # Let's adjust df to test properly

    # Better test data:
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2023-01-01", "2023-01-01"]), "store_nbr": [1, 2]}
    )
    X = transformer.fit_transform(df)
    assert X["is_holiday"].all()


def test_holiday_local(mock_holidays, mock_stores):
    # 2023-01-15 is Manta only
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-15", "2023-01-15"]),
            "store_nbr": [1, 2],  # 1=Quito, 2=Manta
        }
    )

    transformer = HolidayTransformer(holidays_df=mock_holidays, stores_df=mock_stores)
    X = transformer.fit_transform(df)

    # Store 1 (Quito) should NOT be holiday
    assert (
        X[(X["store_nbr"] == 1) & (X["date"] == "2023-01-15")].iloc[0]["is_holiday"]
        == False
    )

    # Store 2 (Manta) SHOULD be holiday
    assert (
        X[(X["store_nbr"] == 2) & (X["date"] == "2023-01-15")].iloc[0]["is_holiday"]
        == True
    )


def test_holiday_transferred(mock_holidays, mock_stores):
    # 2023-01-22 is transferred (so it is NOT a holiday itself)
    # The actual holiday happens on the transferred date (but here we just check the original date is NOT holiday)

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2023-01-22"]),
            "store_nbr": [1],  # Quito
        }
    )

    transformer = HolidayTransformer(holidays_df=mock_holidays, stores_df=mock_stores)
    X = transformer.fit_transform(df)

    # Transferred=True means it is a WORK day.
    assert X.iloc[0]["is_holiday"] == False
