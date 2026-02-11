import pandas as pd

from src.features.meta import OilMerger, StoreMerger


def test_oil_merger_interpolation():
    main_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]),
            "store_nbr": [1, 1, 1],
        }
    )

    # Missing 2013-01-02 in oil
    oil_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-03"]),
            "dcoilwtico": [90.0, 92.0],
        }
    )

    # Test linear interpolation
    merger = OilMerger(oil_df=oil_df, method="linear")
    result = merger.transform(main_df)

    assert result.loc[result["date"] == "2013-01-02", "dcoilwtico"].values[0] == 91.0
    assert result["dcoilwtico"].isnull().sum() == 0


def test_store_merger():
    main_df = pd.DataFrame({"date": pd.to_datetime(["2013-01-01"]), "store_nbr": [1]})

    stores_df = pd.DataFrame(
        {"store_nbr": [1, 2], "city": ["Quito", "Guayaquil"], "cluster": [13, 10]}
    )

    merger = StoreMerger(stores_df=stores_df)
    result = merger.transform(main_df)

    assert result.iloc[0]["city"] == "Quito"
    assert result.iloc[0]["cluster"] == 13
    assert result.shape == (1, 4)
