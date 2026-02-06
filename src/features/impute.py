from typing import List, Optional, Union
import pandas as pd
from .base import BaseTimeSeriesTransformer


class TimeSeriesImputer(BaseTimeSeriesTransformer):
    def __init__(
        self,
        method: str = "ffill",
        value: Union[int, float] = 0,
        columns: Optional[List[str]] = None,
    ):
        self.method = method
        self.value = value
        self.columns = columns

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        self.check_leakage(X)
        X = X.copy()
        cols = self.columns if self.columns else X.columns

        for col in cols:
            if self.method == "ffill":
                X[col] = X[col].ffill()
            elif self.method == "bfill":
                X[col] = X[col].bfill()
            elif self.method == "constant":
                X[col] = X[col].fillna(self.value)
            elif self.method == "interpolate":
                X[col] = X[col].interpolate()

        return X
