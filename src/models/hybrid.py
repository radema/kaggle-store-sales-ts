from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor


class HybridRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self, trend_estimator=None, residual_estimator=None, feature_selector=None
    ):
        self.trend_estimator = trend_estimator
        self.residual_estimator = residual_estimator
        self.feature_selector = feature_selector

        # Internal state
        self.trend_estimator_ = None
        self.residual_estimator_ = None

    def _select_features(self, X, part_idx=None):
        """Helper to subset X based on feature_selector."""
        if self.feature_selector is None:
            return X

        # If feature_selector is a single list of strings
        if isinstance(self.feature_selector, list) and all(
            isinstance(i, str) for i in self.feature_selector
        ):
            return X[self.feature_selector]

        # If feature_selector is a list of lists
        if isinstance(self.feature_selector, list) and all(
            isinstance(i, list) for i in self.feature_selector
        ):
            if part_idx is not None and part_idx < len(self.feature_selector):
                return X[self.feature_selector[part_idx]]
            # Fallback for nested list with only 1 element or index overflow
            if len(self.feature_selector) > 0:
                return X[self.feature_selector[0]]

        return X

    def fit(self, X, y):
        # Initialize estimators if not provided
        self.trend_estimator_ = (
            clone(self.trend_estimator) if self.trend_estimator else Ridge()
        )
        self.residual_estimator_ = (
            clone(self.residual_estimator)
            if self.residual_estimator
            else LGBMRegressor()
        )

        # Prepare data for trend
        X_trend = self._select_features(X, 0)

        # Fit trend
        self.trend_estimator_.fit(X_trend, y)

        # Calculate residuals
        y_trend_pred = self.trend_estimator_.predict(X_trend)
        y_resid = y - y_trend_pred

        # Prepare data for residuals
        X_resid = self._select_features(X, 1)

        # Fit residuals
        self.residual_estimator_.fit(X_resid, y_resid)

        return self

    def predict(self, X):
        components = self.predict_components(X)
        return components["total"]

    def predict_components(self, X):
        X_trend = self._select_features(X, 0)
        X_resid = self._select_features(X, 1)

        # Note: We use the already fitted estimators in self.trend_estimator_ etc.
        y_trend_pred = self.trend_estimator_.predict(X_trend)
        y_resid_pred = self.residual_estimator_.predict(X_resid)

        return {
            "trend": y_trend_pred,
            "residual": y_resid_pred,
            "total": y_trend_pred + y_resid_pred,
        }
