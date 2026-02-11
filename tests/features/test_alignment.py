import pandas as pd
from src.features.alignment import DateGridTransformer


def test_date_grid_transformer():
    # Store 1 has 2013-01-01 and 2013-01-03, missing 2013-01-02
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-03", "2013-01-01"]),
            "store_nbr": [1, 1, 2],
            "family": ["A", "A", "A"],
            "sales": [1, 3, 10],
        }
    )

    # Expected:
    # Store 1, Family A: 3 rows (01, 02, 03)
    # Store 2, Family A: 3 rows (01, 02, 03)

    transformer = DateGridTransformer(
        fill_map={"sales": 0.0}, groupby=["store_nbr", "family"]
    )
    result = transformer.transform(df)

    assert len(result) == 6
    assert (
        result.loc[
            (result["date"] == "2013-01-02") & (result["store_nbr"] == 1), "sales"
        ].values[0]
        == 0.0
    )
    assert (
        result.loc[
            (result["date"] == "2013-01-02") & (result["store_nbr"] == 2), "sales"
        ].values[0]
        == 0.0
    )
    assert (
        result.isnull().sum().sum() == 0
    )  # Everything in fill_map or from previous should be there
