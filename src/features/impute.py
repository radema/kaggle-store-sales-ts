from typing import List, Optional, Union
import pandas as pd
from src.features.base import BaseTimeSeriesTransformer


class TimeSeriesImputer(BaseTimeSeriesTransformer):
    def __init__(
        self,
        method: str = "ffill",
        value: Union[int, float] = 0,
        columns: Optional[List[str]] = None,
    ):
        super().__init__()
        self.method = method
        self.value = value
        self.columns = columns

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        cols = self.columns if self.columns else X.columns
        methods = [self.method] if isinstance(self.method, str) else self.method

        for method in methods:
            for col in cols:
                if method == "ffill":
                    X[col] = X[col].ffill()
                elif method == "bfill":
                    X[col] = X[col].bfill()
                elif method == "constant":
                    X[col] = X[col].fillna(self.value)
                elif method == "interpolate":
                    X[col] = X[col].interpolate()

        return X
