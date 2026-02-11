import pandas as pd
import pytest
from src.features.transactions import TransactionMerger


def test_transaction_merger_basic():
    # Setup main df
    main_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-02", "2013-01-03"]),
            "store_nbr": [1, 1, 1],
            "sales": [10, 20, 0],
        }
    )

    # Setup transactions df
    trans_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-02"]),
            "store_nbr": [1, 1],
            "transactions": [100, 200],
        }
    )

    merger = TransactionMerger(transactions_df=trans_df, fill_value=0.0)
    result = merger.transform(main_df)

    # 2013-01-03 should have 0 transactions (intentional imputation)
    assert result.loc[result["date"] == "2013-01-03", "transactions"].values[0] == 0
    assert result["transactions"].isnull().sum() == 0
    assert "transactions" in result.columns
    assert len(result) == 3  # No duplication


def test_transaction_merger_duplicate_prevention():
    main_df = pd.DataFrame({"date": pd.to_datetime(["2013-01-01"]), "store_nbr": [1]})

    # Malformed transactions with duplicates
    trans_df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2013-01-01", "2013-01-01"]),
            "store_nbr": [1, 1],
            "transactions": [100, 101],
        }
    )

    merger = TransactionMerger(transactions_df=trans_df, fill_value=0.0)

    # Should raise ValueError due to explosion risk (duplicates in right table)
    with pytest.raises(ValueError, match="duplicates"):
        merger.transform(main_df)
