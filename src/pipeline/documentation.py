from typing import Dict, Any
from datetime import datetime


class ModelCardGenerator:
    """
    Generates a Markdown Model Card for experiment documentation.
    """

    def generate(self, config: Dict[str, Any], pipeline_summary: str) -> str:
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

        # Extract metadata
        run_name = config.get("run_name", "unnamed")
        model_conf = config.get("model", {})
        val_conf = config.get("validation", {})

        card = rf"""# Model Card: {run_name}
Generated on: {date_str}

## 1. Description
This model belongs to the Hybrid Baseline family, combining a Linear Trend estimator with a Residual estimator (typically Gradient Boosting).

## 2. Architecture & Spaces
*   **Type**: Hybrid (Additive) Regressor.
*   **Transformation Space**: This model can be considered a function $f: \mathbb{{R}}^n \to \mathbb{{R}}$. 
    *   It maps an $n$-dimensional feature vector (tabular features + lags) to a single scalar representing the expected sales for a specific (store, family, date) tuple.
    *   Unlike a seq2seq model that maps $\mathbb{{R}}^{{T \times n}} \to \mathbb{{R}}^{{H \times 1}}$, this model handles the forecast horizon ($H=16$) through feature engineering (e.g., recursive forecasting or using lags that are available at the time of prediction).
*   **Transformation**: Target is transformed using `{val_conf.get("target_transform", "None")}`.
*   **Input Space**: {model_conf.get("feature_selector", "All features provided by pipeline")}
*   **Output Space**: Predicted Sales (Units).

## 3. Configuration
### Trend Estimator
`{model_conf.get("trend_estimator", {}).get("class", "Default")}`
Parameters: {model_conf.get("trend_estimator", {}).get("params", {})}

### Residual Estimator
`{model_conf.get("residual_estimator", {}).get("class", "Default")}`
Parameters: {model_conf.get("residual_estimator", {}).get("params", {})}

## 4. Pipeline Steps
{pipeline_summary}

## 5. Validation Strategy
*   **Method**: Sliding Window Time Series Split.
*   **Parameters**: {val_conf.get("n_folds")} folds, {val_conf.get("train_days")} train days, {val_conf.get("val_days")} val days.

## 6. Data Constraints & Strategy
*   **Forecasting Mode**: Independent (Direct).
*   **Lag Constraint**: This pipeline expects lags $\ge 15$ days to function correctly during inference without recursion.
*   **Handling of NaNs**: The current baseline model assumes features are fully populated. Lag-induced NaNs (due to the "cold start" or the slide) are handled by the estimators' internal defaults (or by the `impute` transformer if added to the pipeline).
"""
        return card
