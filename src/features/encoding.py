from src.features.base import BaseTimeSeriesTransformer
from sklearn.preprocessing import OrdinalEncoder


class CategoricalEncoder(BaseTimeSeriesTransformer):
    """
    Encodes categorical features using Ordinal Encoding.
    Handles unknown categories by assigning a specific value.
    """

    def __init__(
        self,
        columns=None,
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        suffix=None,
    ):
        """
        Args:
            columns (list): Columns to encode.
            handle_unknown (str): Strategy for unknown categories.
            unknown_value (int): Value to use for unknowns.
            suffix (str): Optional suffix for encoded columns (e.g. '_enc').
        """
        super().__init__()
        self.columns = columns
        self.handle_unknown = handle_unknown
        self.unknown_value = unknown_value
        self.suffix = suffix
        self.encoder_ = None

    def fit(self, X, y=None):
        if not self.columns:
            # Auto-detect object/category columns if not specified
            self.columns = X.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

        if not self.columns:
            return self

        self.encoder_ = OrdinalEncoder(
            handle_unknown=self.handle_unknown,
            unknown_value=self.unknown_value,
        )
        self.encoder_.fit(X[self.columns].astype(str))
        return self

    def _transform(self, X):
        if self.encoder_ is None or not self.columns:
            return X

        X = X.copy()
        encoded = self.encoder_.transform(X[self.columns].astype(str))

        for i, col in enumerate(self.columns):
            out_col = f"{col}{self.suffix}" if self.suffix else col
            X[out_col] = encoded[:, i]

        return X
