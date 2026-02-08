# Knowledge Base: Time-Series Forecasting Architecture

## 1. Core Axioms
- **Pipeline Idempotency**: A transformation pipeline must produce the same features regardless of the specific date range, provided the context (historical lags) is maintained.
- **Trend-Residual Separation**: In hybrid models, the trend estimator (Linear/Continuous) and residual estimator (Tree-based/Discrete) should consume partitioned feature spaces to avoid multicollinearity and capture distinct signal components.
- **Direct Strategy Constraint**: If no recursive loop is present, all features must be computable using only data available at time $t$ for a prediction $t+H$.

## 2. Technical Constraints
- **Lag Gaps**: Lag features create NaNs at the series start. Chained imputation (e.g., `ffill` -> `constant`) is mandatory for linear estimators (Ridge/SVR) that do not handle NaNs.
- **Categorical Handling**: While GBTs (LightGBM) handle categoricals, a standardized `OrdinalEncoder` in the pipeline ensures full traceability and serialization of the category-to-integer mapping.
- **Inference Context (The Blind Validation Trap)**: Pipelines containing stateful transformers (Lags, Rolling Windows, Levels) MUST be provided with sufficient historical context during `predict()` calls. Predicting on a standalone validation/test slice without its historical "tail" results in feature-dropout (zeroed values).
- **Leveling Protocol**: Any historical average or "Level" must be computed using a strict lag of $\ge 16$ days (competition horizon) to prevent look-ahead bias during out-of-fold validation and final inference.

## 3. Knowledge Graph Nodes (Cross-Links)
- **External**: [Kaggle Store Sales Competition](https://www.kaggle.com/competitions/store-sales-time-series-forecasting)
- **Internal**: `src/models/hybrid.py` (Implementation of Axiom 2)
- **Internal**: `src/pipeline/ runners/hybrid.py` (Implementation of Constraint 3)
- **Strategy**: `strategy-map.md` (Updated in Bolt #5)
