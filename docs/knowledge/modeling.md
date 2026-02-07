# Knowledge: Modeling Strategy & Lessons Learned

## Axioms
- **Axiom 1: Additive Decomposition**: Store sales time series are most robustly modeled as a sum of a deterministic trend (Linear/Ridge) and a stochastic residual component (LGBM/XGBoost).
- **Axiom 2: Explicit Feature Routing**: High-correlation time-features (like `date_idx`) should be isolated for the Trend model to prevent the Residual model from over-fitting to the global trend signal.

## Constraints
- **Constraint 1: Target Invariance**: The model components must operate on the same target space (Real OR Log-Space) to ensure residuals `y - y_pred` are mathematically meaningful for the second stage.
- **Constraint 2: Sklearn Interface**: All custom models MUST implement the `BaseEstimator` interface to allow wrapping in `TransformedTargetRegressor` or integration into `Pipeline`.

## Patterns (The Hybrid Pattern)
The `HybridRegressor` pattern implemented in Bolt #4 provides:
1. **Extrapolation Power**: Linear models handle "out-of-bounds" future dates better than tree-based models.
2. **Interaction Capture**: Boosting models handle complex seasonality and family-store interactions that linear models miss.
3. **Debuggability**: Component predictions (`predict_components`) allow visual sanity checks on "Is the trend still linear?" and "What is the scale of residuals?".

## Cross-Links
- **Internal**: [Feature Engineering](Bolt-#3), [Validation Framework](Bolt-#1).
- **External**: [Kaggle Hybrid Model Guide](https://www.kaggle.com/code/ryanholbrook/hybrid-models).
