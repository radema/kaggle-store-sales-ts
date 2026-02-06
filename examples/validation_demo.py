import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.validation.splitters import SlidingWindowTS
from src.validation.harness import EvaluationSuite


def demo():
    # 1. Setup sample data
    # 2 years of daily data for 5 stores and 3 families
    dates = pd.date_range("2015-01-01", "2016-12-31")
    stores = [1, 2, 3, 4, 5]
    families = ["BREAD", "DAIRY", "MEATS"]

    data = []
    for d in dates:
        for s in stores:
            for f in families:
                # Add some trend and seasonality
                base = 100 + (d.dayofweek * 10) + (d.month * 5)
                sales = base + np.random.normal(0, 10)
                data.append([d, s, f, max(0, sales)])

    df = pd.DataFrame(data, columns=["date", "store_nbr", "family", "sales"])

    print(f"Data shape: {df.shape}")

    # 2. Setup Splitter
    # 365 days train, 16 days val, 3 folds, 16 days step (contiguous)
    splitter = SlidingWindowTS(train_days=365, val_days=16, n_folds=3)

    suite = EvaluationSuite()

    for i, (train_idx, val_idx) in enumerate(splitter.split(df)):
        print(f"\nProcessing Fold {i + 1}...")
        val_df = df.iloc[val_idx].copy()

        # Simple dummy prediction: Mean of train sales (very basic)
        train_mean = df.iloc[train_idx]["sales"].mean()
        y_pred = np.full(len(val_idx), train_mean)

        # 3. Evaluate
        report = suite.evaluate(val_df, y_pred)
        print(f"Fold {i + 1} Global RMSLE: {report['global_rmsle']:.4f}")

        # 4. Save visualization (smoke test)
        fig = suite.plot_errors(report, title_suffix=f"- Fold {i + 1}")
        fig.savefig(f"evaluation_fold_{i + 1}.png")
        plt.close(fig)

    print("\nDemo complete. Visualization artifacts saved to evaluation_fold_*.png")


if __name__ == "__main__":
    demo()
