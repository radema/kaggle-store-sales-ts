import pandas as pd
import pytest
from src.features.base import BaseTimeSeriesTransformer
from src.utils.validation import validate_no_duplicates


class MockTransformer(BaseTimeSeriesTransformer):
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["new_col"] = "test"
        return X


class FailingValidationTransformer(BaseTimeSeriesTransformer):
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        # Intentionally create duplicates
        X = pd.concat([X, X])
        return X

    def post_transform_validation(self, df: pd.DataFrame):
        validate_no_duplicates(df)


def test_base_transformer_logic(caplog):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    transformer = MockTransformer(name="TestMock")

    result = transformer.transform(df)

    assert "new_col" in result.columns
    assert result.shape == (2, 3)

    # Check logs
    assert "Starting transformation" in caplog.text
    assert "Finished transformation" in caplog.text
    assert "TestMock" in caplog.text


def test_validation_failure():
    df = pd.DataFrame({"a": [1], "b": [2]})
    transformer = FailingValidationTransformer(name="BadMock")

    with pytest.raises(ValueError, match="duplicate"):
        transformer.transform(df)
