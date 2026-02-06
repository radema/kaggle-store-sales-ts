import pytest
import pandas as pd
from src.features.base import BaseTimeSeriesTransformer


class MockTransformer(BaseTimeSeriesTransformer):
    def transform(self, X):
        self.check_leakage(X)
        return X


def test_base_transformer_exists():
    assert issubclass(MockTransformer, BaseTimeSeriesTransformer)


def test_leakage_check_method_exists():
    t = MockTransformer()
    df = pd.DataFrame({"a": [1]}, index=pd.to_datetime(["2021-01-01"]))
    # Should not raise any error as it's a placeholder/hook in the base class
    res = t.transform(df)
    assert res.equals(df)
