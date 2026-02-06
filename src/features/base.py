from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd


class BaseTimeSeriesTransformer(BaseEstimator, TransformerMixin):
    """
    Base class for all time-series transformers in the pipeline.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.check_leakage(X)
        return X

    def check_leakage(self, X: pd.DataFrame):
        """
        Placeholder for leakage checks.
        Future concrete classes will implement specific logic here
        (e.g. checking if we are using t+1 info).
        """
        pass
