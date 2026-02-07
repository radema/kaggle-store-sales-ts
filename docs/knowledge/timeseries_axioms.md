# Knowledge Base: Time-Series Forecasting Architecture

## 1. Core Axioms
- **Pipeline Idempotency**: A transformation pipeline must produce the same features regardless of the specific date range, provided the context (historical lags) is maintained.
- **Trend-Residual Separation**: In hybrid models, the trend estimator (Linear/Continuous) and residual estimator (Tree-based/Discrete) should consume partitioned feature spaces to avoid multicollinearity and capture distinct signal components.
- **Direct Strategy Constraint**: If no recursive loop is present, all features must be computable using only data available at time $t$ for a prediction $t+H$.

## 2. Technical Constraints
- **Lag Gaps**: Lag features create NaNs at the series start. Chained imputation (e.g., `ffill` -> `constant`) is mandatory for linear estimators (Ridge/SVR) that do not handle NaNs.
- **Categorical Handling**: While GBTs (LightGBM) handle categoricals, a standardized `OrdinalEncoder` in the pipeline ensures full traceability and serialization of the category-to-integer mapping.
- **Inference Context**: At test time, a window of training data (size $\ge \text{max\_lag}$) must be prepended to the test set to allow the feature pipeline to compute lagged values.

## 3. Knowledge Graph Nodes (Cross-Links)
- **External**: [Kaggle Store Sales Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Internal**: `src/models/hybrid.py` (Implementation of Axiom 2)
- **Internal**: `src/pipeline/ runners/hybrid.py` (Implementation of Constraint 3)
- **Strategy**: `strategy-map.md` (Updated in Bolt #5)
