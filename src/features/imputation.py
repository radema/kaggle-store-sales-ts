import pandas as pd
from typing import Dict, Any, List
from src.features.base import BaseTimeSeriesTransformer


class ConfigurableImputer(BaseTimeSeriesTransformer):
    """
    General purpose imputer that applies specified strategies to specific columns.
    Supported strategies: 'fill' (constant), 'ffill', 'bfill', 'interpolate'.
    """

    def __init__(self, strategies: Dict[str, Dict[str, Any]]):
        """
        Example strategies:
        {
            "sales": {"method": "fill", "value": 0.0},
            "oil_price": {"method": "ffill", "limit": 7}
        }
        """
        super().__init__()
        self.strategies = strategies

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        for col, config in self.strategies.items():
            if col not in X.columns:
                self.logger.warning(f"Column '{col}' not found for imputation.")
                continue

            method = config.get("method", "fill")
            null_count = X[col].isnull().sum()

            if null_count == 0:
                continue

            self.logger.info(f"Imputing {null_count} values in '{col}' using {method}")

            if method == "fill":
                X[col] = X[col].fillna(config.get("value", 0))
            elif method == "ffill":
                X[col] = X[col].ffill(limit=config.get("limit"))
            elif method == "bfill":
                X[col] = X[col].bfill(limit=config.get("limit"))
            elif method == "interpolate":
                X[col] = X[col].interpolate(
                    method=config.get("interp_method", "linear")
                )
            else:
                self.logger.error(f"Unknown imputation method: {method}")

        return X
