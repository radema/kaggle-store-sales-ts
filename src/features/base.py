from abc import ABC, abstractmethod
import pandas as pd
import time
from typing import Optional, Dict, Any, List
from sklearn.base import BaseEstimator, TransformerMixin
from src.utils.logging import get_logger
from src.utils.validation import validate_no_duplicates


class BaseTimeSeriesTransformer(BaseEstimator, TransformerMixin, ABC):
    """
    Base class for all feature transformers in the data processing pipeline.
    Provides standardized logging, timing, and validation hooks.
    Uses the Template Method pattern with _transform as the implementation hook.
    """

    def __init__(self, name: Optional[str] = None):
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)

    def fit(self, X: pd.DataFrame, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        start_time = time.time()
        self.logger.info(f"Starting transformation. Input shape: {X.shape}")

        try:
            X_transformed = self._transform(X)
        except Exception as e:
            self.logger.error(f"Transformation failed: {str(e)}")
            raise e

        elapsed = time.time() - start_time
        self.logger.info(
            f"Finished transformation in {elapsed:.4f}s. Output shape: {X_transformed.shape}"
        )

        self.post_transform_validation(X_transformed)
        return X_transformed

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Implement the actual transformation logic here.
        """
        pass

    def post_transform_validation(self, df: pd.DataFrame):
        """
        Hook for validation logic after transformation.
        Override in subclasses for stricter checks.
        """
        # Default: check for uniqueness if validate_unique_index is set
        if getattr(self, "validate_unique_index", False):
            validate_no_duplicates(df, subset=None)
        pass
