import pytest
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from src.models.hybrid import HybridRegressor


class MockEstimator(BaseEstimator, RegressorMixin):
    """Helper to track what data was passed to fit/predict"""

    def __init__(self, return_value=0.0):
        self.return_value = return_value
        self.last_X = None
        self.last_y = None
        self.last_X_predict = None

    def fit(self, X, y):
        self.last_X = X
        self.last_y = y
        return self

    def predict(self, X):
        self.last_X_predict = X
        # Handle cases where X might be a dataframe or numpy array
        return np.full(len(X), self.return_value)

    def get_params(self, deep=True):
        return {"return_value": self.return_value}


@pytest.fixture
def sample_data():
    X = pd.DataFrame(
        {
            "trend_feat": range(10),
            "resid_feat": range(10, 20),
            "shared_feat": range(20, 30),
        }
    )
    y = np.array(range(100, 110))
    return X, y


def test_initialization():
    model = HybridRegressor()
    assert model.trend_estimator is None
    assert model.residual_estimator is None


def test_feature_splitting_none(sample_data):
    """Test that None selector passes all features to both models"""
    X, y = sample_data
    trend_mock = MockEstimator()
    resid_mock = MockEstimator()

    model = HybridRegressor(
        trend_estimator=trend_mock, residual_estimator=resid_mock, feature_selector=None
    )
    model.fit(X, y)

    pd.testing.assert_frame_equal(model.trend_estimator_.last_X, X)
    pd.testing.assert_frame_equal(model.residual_estimator_.last_X, X)


def test_feature_splitting_single_list(sample_data):
    """Test that single list selector passes subset to both models"""
    X, y = sample_data
    selector = ["trend_feat", "shared_feat"]

    trend_mock = MockEstimator()
    resid_mock = MockEstimator()

    model = HybridRegressor(
        trend_estimator=trend_mock,
        residual_estimator=resid_mock,
        feature_selector=selector,
    )
    model.fit(X, y)

    expected_X = X[selector]
    pd.testing.assert_frame_equal(model.trend_estimator_.last_X, expected_X)
    pd.testing.assert_frame_equal(model.residual_estimator_.last_X, expected_X)


def test_feature_splitting_nested_list(sample_data):
    """Test that nested list selector directs features correctly"""
    X, y = sample_data
    # [Trend Features, Residual Features]
    selector = [["trend_feat"], ["resid_feat", "shared_feat"]]

    trend_mock = MockEstimator()
    resid_mock = MockEstimator()

    model = HybridRegressor(
        trend_estimator=trend_mock,
        residual_estimator=resid_mock,
        feature_selector=selector,
    )
    model.fit(X, y)

    pd.testing.assert_frame_equal(model.trend_estimator_.last_X, X[["trend_feat"]])
    pd.testing.assert_frame_equal(
        model.residual_estimator_.last_X, X[["resid_feat", "shared_feat"]]
    )


def test_fit_predict_logic(sample_data):
    """Test that residuals are calculated correctly and predictions are summed"""
    X, y = sample_data

    # Trend model predicts 5.0
    # True y is ~100. Residuals should be ~95.0
    trend_model = MockEstimator(return_value=5.0)

    # Residual model predicts 2.0
    resid_model = MockEstimator(return_value=2.0)

    model = HybridRegressor(trend_estimator=trend_model, residual_estimator=resid_model)
    model.fit(X, y)

    # Check residuals passed to resid_model
    # y (100..109) - 5.0 = 95..104
    expected_residuals = y - 5.0
    np.testing.assert_array_almost_equal(
        model.residual_estimator_.last_y, expected_residuals
    )

    # Check predict
    preds = model.predict(X)
    # 5.0 + 2.0 = 7.0
    expected_preds = np.full(len(X), 7.0)
    np.testing.assert_array_almost_equal(preds, expected_preds)


def test_predict_components(sample_data):
    """Test that predict_components returns dictionary with parts"""
    X, y = sample_data
    trend_model = MockEstimator(return_value=10.0)
    resid_model = MockEstimator(return_value=5.0)

    model = HybridRegressor(trend_estimator=trend_model, residual_estimator=resid_model)
    model.fit(X, y)

    components = model.predict_components(X)
    assert isinstance(components, dict)
    assert "trend" in components
    assert "residual" in components
    assert "total" in components

    np.testing.assert_array_equal(components["trend"], np.full(len(X), 10.0))
    np.testing.assert_array_equal(components["residual"], np.full(len(X), 5.0))
    np.testing.assert_array_equal(components["total"], np.full(len(X), 15.0))


from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


def test_transformed_target_regressor_compatibility(sample_data):
    """Test that HybridRegressor works with TransformedTargetRegressor"""
    X, y = sample_data
    # Log transform target
    model = HybridRegressor(
        trend_estimator=LinearRegression(), residual_estimator=DecisionTreeRegressor()
    )

    wrapper = TransformedTargetRegressor(
        regressor=model, func=np.log1p, inverse_func=np.expm1
    )

    wrapper.fit(X, y)
    preds = wrapper.predict(X)

    assert len(preds) == len(X)
    assert np.all(preds >= 0)


def test_pipeline_compatibility(sample_data):
    """Test that HybridRegressor works inside a Pipeline"""
    X, y = sample_data

    pipe = Pipeline(
        [
            (
                "hybrid",
                HybridRegressor(
                    trend_estimator=LinearRegression(),
                    residual_estimator=DecisionTreeRegressor(),
                ),
            )
        ]
    )

    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert len(preds) == len(X)
