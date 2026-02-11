import pandas as pd
from typing import Optional
from src.features.base import BaseTimeSeriesTransformer
from src.utils.validation import validate_no_duplicates


class TransactionMerger(BaseTimeSeriesTransformer):
    """
    Merges transaction data into the main dataframe.
    Validates that no duplicates are created during the join and
    imputes missing values intentionally.
    """

    def __init__(
        self,
        transactions_df: Optional[pd.DataFrame] = None,
        transactions_path: Optional[str] = None,
        fill_value: float = None,  # No default, must be explicit
    ):
        super().__init__()
        if fill_value is None:
            # We will check this in _transform to allow lazy loading but enforce config
            pass
        self.transactions_df = transactions_df
        self.transactions_path = transactions_path
        self.fill_value = fill_value

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.fill_value is None:
            raise ValueError(
                "TransactionMerger requires an explicit 'fill_value' parameter for intentional imputation."
            )
        # Load transactions if path provided
        trans = self.transactions_df
        if trans is None and self.transactions_path:
            self.logger.info(f"Loading transactions from {self.transactions_path}")
            trans = pd.read_csv(self.transactions_path)
            trans["date"] = pd.to_datetime(trans["date"])

        if trans is None:
            self.logger.warning("No transactions data provided. Skipping merge.")
            return X

        # Validation: Check for duplicates in transaction data to prevent row explosion
        self.logger.info("Validating transaction data uniqueness.")
        validate_no_duplicates(trans, subset=["date", "store_nbr"])

        # Record pre-merge shape
        row_count_before = len(X)

        # Merge
        X = X.merge(trans, on=["date", "store_nbr"], how="left")

        # Validation: Ensure no row count explosion
        if len(X) != row_count_before:
            self.logger.error(
                f"Row count changed from {row_count_before} to {len(X)} during transaction merge!"
            )
            raise ValueError(
                "Row count changed during merge, check for duplicates in keys."
            )

        # Intentional Imputation
        null_count = X["transactions"].isnull().sum()
        if null_count > 0:
            self.logger.info(
                f"Imputing {null_count} missing transaction values with {self.fill_value}"
            )
            X["transactions"] = X["transactions"].fillna(self.fill_value)

        return X
