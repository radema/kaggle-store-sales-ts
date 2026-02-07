import pandas as pd
from src.features.dates import DatePartTransformer


def test_date_part_transformer_basic():
    # Create sample data
    dates = pd.to_datetime(["2023-01-01", "2023-01-15", "2023-02-28"])
    df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

    # Initialize transformer
    transformer = DatePartTransformer(parts=["year", "month", "day", "dayofweek"])

    # Fit and transform
    X_transformed = transformer.fit_transform(df)

    # Assertions
    assert "year" in X_transformed.columns
    assert "month" in X_transformed.columns
    assert "day" in X_transformed.columns
    assert "dayofweek" in X_transformed.columns

    assert X_transformed.iloc[0]["year"] == 2023
    assert X_transformed.iloc[0]["month"] == 1
    assert X_transformed.iloc[0]["day"] == 1
    # 2023-01-01 was a Sunday (6 in pandas dayofweek 0-6 if Monday=0)
    assert X_transformed.iloc[0]["dayofweek"] == 6


def test_date_part_transformer_column_input():
    # Create sample data with date column
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2023-01-01", "2023-01-02"]), "value": [10, 20]}
    )

    transformer = DatePartTransformer(parts=["day"], column="date")
    X_transformed = transformer.fit_transform(df)

    assert "day" in X_transformed.columns
    assert X_transformed.iloc[0]["day"] == 1
    assert X_transformed.iloc[1]["day"] == 2


def test_date_part_transformer_wage_day():
    # Test strictly wage days (15th and last day of month)
    dates = pd.to_datetime(["2023-01-15", "2023-01-31", "2023-02-14"])
    df = pd.DataFrame({"value": [1, 2, 3]}, index=dates)

    transformer = DatePartTransformer(parts=["is_wage_day"])
    X_transformed = transformer.fit_transform(df)

    assert "is_wage_day" in X_transformed.columns
    assert bool(X_transformed.iloc[0]["is_wage_day"]) is True  # 15th
    assert bool(X_transformed.iloc[1]["is_wage_day"]) is True  # 31st (End of Month)
    assert bool(X_transformed.iloc[2]["is_wage_day"]) is False  # 14th
