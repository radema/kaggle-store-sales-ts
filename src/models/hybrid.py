from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor


class HybridRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        trend_estimator=None,
        residual_estimator=None,
        feature_selector=None,
        gate_feature=None,
        categorical_features=None,
        compose_mode="additive",
    ):
        self.trend_estimator = trend_estimator
        self.residual_estimator = residual_estimator
        self.feature_selector = feature_selector
        self.gate_feature = gate_feature
        self.categorical_features = categorical_features
        self.compose_mode = compose_mode  # "additive" or "multiplicative"

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

        # Filtering logic for Gated Baseline
        if self.gate_feature is not None:
            # Only train on rows where the store/family is NOT closed
            mask = X[self.gate_feature] == 0
            X_fit = X[mask].copy()
            y_fit = y[mask].copy()
        else:
            X_fit = X
            y_fit = y

        # Prepare data for trend
        X_trend = self._select_features(X_fit, 0)

        # Fit trend
        self.trend_estimator_.fit(X_trend, y_fit)

        # Calculate residuals
        y_trend_pred = self.trend_estimator_.predict(X_trend)

        if self.compose_mode == "multiplicative":
            # Prevent division by zero
            y_resid = y_fit / (y_trend_pred + 1e-9)
        else:
            y_resid = y_fit - y_trend_pred

        # Prepare data for residuals
        X_resid = self._select_features(X_fit, 1)

        # Fit residuals
        if (
            isinstance(self.residual_estimator_, LGBMRegressor)
            and self.categorical_features
        ):
            # Pass categoricals at fit time to avoid construction errors
            self.residual_estimator_.fit(
                X_resid, y_resid, categorical_feature=self.categorical_features
            )
        else:
            self.residual_estimator_.fit(X_resid, y_resid)

        return self

    def predict(self, X):
        components = self.predict_components(X)
        self.last_components_ = components
        return components["total"]

    def predict_components(self, X):
        X_trend = self._select_features(X, 0)
        X_resid = self._select_features(X, 1)

        # Note: We use the already fitted estimators in self.trend_estimator_ etc.
        y_trend_pred = self.trend_estimator_.predict(X_trend)
        y_resid_pred = self.residual_estimator_.predict(X_resid)

        if self.compose_mode == "multiplicative":
            total = y_trend_pred * y_resid_pred
        else:
            total = y_trend_pred + y_resid_pred

        # Apply gate if provided
        if self.gate_feature is not None:
            # total is a numpy array, X[self.gate_feature] needs to be aligned
            gate_mask = X[self.gate_feature].values
            total = total * (1 - gate_mask)

        return {
            "trend": y_trend_pred,
            "residual": y_resid_pred,
            "total": total,
        }
