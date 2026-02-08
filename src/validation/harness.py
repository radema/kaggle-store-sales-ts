from src.validation.metrics import squared_log_error
import numpy as np
import pandas as pd
from typing import Optional
import matplotlib.pyplot as plt
import seaborn as sns


class EvaluationSuite:
    """
    Evaluation suite for analyzing forecasting performance across
    different dimensions (store, family, time).
    """

    def __init__(self):
        # We can add plot styles here
        sns.set_theme(style="whitegrid")

    def evaluate(self, y_true_df, y_pred, extra_cols: Optional[pd.DataFrame] = None):
        """
        Computes metrics and prepares evaluation artifacts.

        Parameters:
        -----------
        y_true_df : pd.DataFrame
            DataFrame containing actual 'sales' and metadata columns
            ('date', 'store_nbr', 'family').
        y_pred : array-like
            Predicted sales values.
        extra_cols : pd.DataFrame, optional
            Additional columns to append to the detailed results (e.g., model components).

        Returns:
        --------
        dict :
            A dictionary containing:
            - global_rmsle: float
            - per_store: pd.Series
            - per_family: pd.Series
            - detailed: pd.DataFrame (actuals, preds, residuals, metadata)
        """
        results = y_true_df.copy()

        # Clip intentionally handled by metrics function but good to have explicit prediction column
        results["sales_pred"] = np.maximum(y_pred, 0)

        # Use shared metric logic for the error calculation
        results["sq_log_error"] = squared_log_error(
            results["sales"], results["sales_pred"]
        )
        results["residual"] = results["sales_pred"] - results["sales"]

        # Append extra information if provided
        if extra_cols is not None:
            # Reindex to match results or use reset_index if indices are assumed aligned
            extra_cols_copy = extra_cols.copy()
            if len(extra_cols_copy) == len(results):
                for col in extra_cols_copy.columns:
                    results[col] = extra_cols_copy[col].values

        # Global metric
        global_rmsle = np.sqrt(results["sq_log_error"].mean())

        # Grouped metrics
        per_store = results.groupby("store_nbr")["sq_log_error"].mean().apply(np.sqrt)
        per_family = results.groupby("family")["sq_log_error"].mean().apply(np.sqrt)

        # Rankings for the summary
        worst_5_stores = per_store.sort_values(ascending=False).head(5)
        worst_5_families = per_family.sort_values(ascending=False).head(5)

        report = {
            "global_rmsle": global_rmsle,
            "per_store": per_store,
            "per_family": per_family,
            "worst_5_stores": worst_5_stores,
            "worst_5_families": worst_5_families,
            "detailed": results,
        }

        return report

    def print_summary(self, report):
        """
        Prints a text summary of the evaluation results.
        """
        print("\n" + "=" * 40)
        print("📊 FORECASTING EVALUATION SUMMARY")
        print("=" * 40)
        print(f"Global RMSLE: {report['global_rmsle']:.4f}")
        print("-" * 40)

        print("\nTop 5 Worst Stores (RMSLE):")
        for store, error in report["worst_5_stores"].items():
            print(f"  Store {store:2}: {error:.4f}")

        print("\nTop 5 Worst Families (RMSLE):")
        for family, error in report["worst_5_families"].items():
            print(f"  {family:20}: {error:.4f}")
        print("=" * 40 + "\n")

    def plot_errors(self, report, title_suffix=""):
        """
        Generates analysis plots from the evaluation report.
        """
        detailed = report["detailed"]

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f"Forecasting Evaluation Performance {title_suffix}", fontsize=16)

        # 1. Top 10 Families by RMSLE
        top_families = report["per_family"].sort_values(ascending=False).head(10)
        sns.barplot(
            x=top_families.values,
            y=top_families.index,
            ax=axes[0, 0],
            palette="viridis",
        )
        axes[0, 0].set_title("Top 10 High-Error Families (RMSLE)")

        # 2. Top 10 Stores by RMSLE
        top_stores = report["per_store"].sort_values(ascending=False).head(10)
        sns.barplot(
            x=top_stores.values,
            y=top_stores.index.astype(str),
            ax=axes[0, 1],
            palette="magma",
        )
        axes[0, 1].set_title("Top 10 High-Error Stores (RMSLE)")

        # 3. Error over time (Daily Mean RMSLE)
        # Check if date is available
        if "date" in detailed.columns:
            error_over_time = (
                detailed.groupby("date")["sq_log_error"].mean().apply(np.sqrt)
            )
            axes[1, 0].plot(
                error_over_time.index, error_over_time.values, marker="o", linestyle="-"
            )
            axes[1, 0].set_title("RMSLE over Time")
            axes[1, 0].tick_params(axis="x", rotation=45)

        # 4. Residual Distribution
        sns.histplot(detailed["residual"], kde=True, ax=axes[1, 1])
        axes[1, 1].set_title("Residual Distribution (Pred - Actual)")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        return fig
