import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.linear_model import Ridge
from lightgbm import LGBMRegressor
from sklearn.preprocessing import OrdinalEncoder
from src.models.hybrid import HybridRegressor
from src.validation.splitters import SlidingWindowTS
from src.validation.harness import EvaluationSuite
from src.utils.logging import get_logger

# Load logger
logger = get_logger("train_baseline")


class IterativeGatedRunner:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.logger = logger
        self.data_dir = Path("data/processed")
        self.output_dir = Path("artifacts/baseline")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_data(self):
        self.logger.info("Loading processed data...")
        self.train_df = pd.read_parquet(self.data_dir / "train.parquet")
        self.test_df = pd.read_parquet(self.data_dir / "test.parquet")
        self.catalog = yaml.safe_load(open(self.data_dir / "feature_catalog.yaml", "r"))

        # Pre-process categoricals (Ordinal Encoding)
        # We need a consistent encoding across train/test and all folds
        self.cat_cols = self.config.get("categoricals", [])
        self.encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )

        # Combine just to fit encoder on all possible labels (city, state, family, type)
        full_cats = pd.concat(
            [self.train_df[self.cat_cols], self.test_df[self.cat_cols]]
        )
        self.encoder.fit(full_cats)

        self.train_df[self.cat_cols] = self.encoder.transform(
            self.train_df[self.cat_cols]
        )
        self.test_df[self.cat_cols] = self.encoder.transform(
            self.test_df[self.cat_cols]
        )
        self.logger.info(f"Encoded {len(self.cat_cols)} categorical features.")

    def update_features_iterative(self, df, current_date, next_date):
        """
        Updates lags and rolling features for next_date based on
        the predictions/actuals available up to current_date.
        """
        # 1. Update Lags for next_date
        # In a real iterative loop, we'd shift.
        # But here we can recalculate easily since we have the full history in 'df'.
        lags = [7, 14, 21, 28]

        # Subset to speed up: only next_date rows
        next_mask = df["date"] == next_date

        # Note: In this simplified baseline, we assume 'preprocess.py' handled
        # the initial state. For iterative prediction, we ONLY update
        # the features that depend on the target.

        # For simplicity, we can use the fact that lag_k(t+1) == sales(t+1-k)
        # We need to find the sales at (next_date - lag days)
        for lag in lags:
            lag_date = next_date - pd.Timedelta(days=lag)
            # Find the sales for each (store, family) at lag_date
            # This can be slow if done row by row. Better to join or map.
            sales_map = df[df["date"] == lag_date].set_index(["store_nbr", "family"])[
                "log1p_sales"
            ]
            df.loc[next_mask, f"lag_{lag}"] = (
                df.loc[next_mask]
                .set_index(["store_nbr", "family"])
                .index.map(sales_map)
                .values
            )

        # 2. Update Rolling 30
        # rolling_30(t+1) = mean(sales in [t+1-30, t])
        roll_start = next_date - pd.Timedelta(days=30)
        roll_end = current_date
        roll_values = (
            df[(df["date"] >= roll_start) & (df["date"] <= roll_end)]
            .groupby(["store_nbr", "family"])["log1p_sales"]
            .mean()
        )

        df.loc[next_mask, "rolling_30_sales"] = (
            df.loc[next_mask]
            .set_index(["store_nbr", "family"])
            .index.map(roll_values)
            .values
        )
        df.loc[next_mask, "rolling_30_sales"] = df.loc[
            next_mask, "rolling_30_sales"
        ].fillna(0)

        # 2b. Update Rolling 30 for Target Encodings (cluster, city, type)
        # These are simple averages of the sales within the grouped attributes over 30 days
        rolling_scope = df[(df["date"] >= roll_start) & (df["date"] <= roll_end)]

        levels = ["cluster", "city", "type"]
        for level in levels:
            col_name = f"rolling_30_{level}_sales"
            if col_name in df.columns:
                # Average by level and family
                level_roll = rolling_scope.groupby([level, "family"])[
                    "log1p_sales"
                ].mean()

                # Map back using the next_mask store attributes
                df.loc[next_mask, col_name] = (
                    df.loc[next_mask]
                    .set_index([level, "family"])
                    .index.map(level_roll)
                    .values
                )
                df.loc[next_mask, col_name] = df.loc[next_mask, col_name].fillna(0)

        # 3. Update is_closed (Heuristic: Closed if rolling avg is 0 and NO ongoing promotion)
        # This allows a "warm-up" period where a closed store can reopen if promoted.
        df.loc[next_mask, "is_closed"] = (
            (df.loc[next_mask, "rolling_30_sales"] == 0)
            & (df.loc[next_mask, "onpromotion"] == 0)
        ).astype(np.int8)

    def iterative_predict(self, model, context_df, dates):
        """Predicts day-by-day and updates context_df."""
        preds = []
        trend_preds = []
        resid_preds = []
        unique_dates = sorted(dates)

        for i, curr_date in enumerate(unique_dates):
            mask = context_df["date"] == curr_date
            X_step = context_df[mask]

            # Predict components
            comp = model.predict_components(X_step)
            y_pred_step = comp["total"]

            context_df.loc[mask, "log1p_sales"] = y_pred_step
            preds.append(y_pred_step)
            trend_preds.append(comp["trend"])
            resid_preds.append(comp["residual"])

            # Update next date's features
            if i < len(unique_dates) - 1:
                next_date = unique_dates[i + 1]
                self.update_features_iterative(context_df, curr_date, next_date)

        return {
            "total": np.concatenate(preds),
            "trend": np.concatenate(trend_preds),
            "residual": np.concatenate(resid_preds),
        }

    def run_cv(self):
        val_conf = self.config["validation"]
        splitter = SlidingWindowTS(
            train_days=val_conf["train_days"],
            val_days=val_conf["val_days"],
            n_folds=val_conf["n_folds"],
        )

        harness = EvaluationSuite()
        model_conf = self.config["model"]
        feature_selector = [
            model_conf["features"]["trend"],
            model_conf["features"]["residual"],
        ]

        fold_metrics = []
        all_oof_results = []

        # We work on a copy to avoid corrupting data across folds
        df = self.train_df.sort_values(["date", "store_nbr", "family"]).copy()

        for fold, (train_idx, val_idx) in enumerate(splitter.split(df)):
            self.logger.info(f"--- Fold {fold + 1} ---")

            # Split data
            # Note: We need context for lags, but HybridRegressor.fit expects clean (X, y)
            train_fold = df.iloc[train_idx].copy()
            val_fold = df.iloc[val_idx].copy()

            # Initialize Model
            model = HybridRegressor(
                trend_estimator=Ridge(),
                residual_estimator=LGBMRegressor(n_estimators=100, random_state=42),
                feature_selector=feature_selector,
                gate_feature=model_conf["gate_feature"],
                categorical_features=self.cat_cols,  # Pass standard names here
            )

            # Fit
            # is_closed from the data is used for training
            model.fit(train_fold, train_fold[val_conf["target_col"]])

            # Log Trend Coefficients
            self.logger.info(f"Fold {fold + 1} Trend Model Params:")
            self.logger.info(f"  Intercept: {model.trend_estimator_.intercept_:.4f}")
            for feat, coef in zip(
                model_conf["features"]["trend"], model.trend_estimator_.coef_
            ):
                self.logger.info(f"  {feat:20}: {coef:.4f}")

            # Validation Window
            val_dates = sorted(val_fold["date"].unique())

            # Iterate
            # We need a context that includes train (for lags) and val (to fill with preds)
            # For simplicity, we use the whole 'df' and only work on the val slice
            results = self.iterative_predict(model, df, val_dates)
            y_pred_log = results["total"]

            # Invert transform for evaluation
            y_pred = np.expm1(y_pred_log)
            # Harness expects original sales and names for pretty-printing
            val_true = val_fold.copy()
            # Restore names for evaluation
            val_true[self.cat_cols] = self.encoder.inverse_transform(
                val_true[self.cat_cols]
            )

            # Prepare extra columns for analysis
            extra_cols = pd.DataFrame(
                {"trend_log": results["trend"], "residual_log": results["residual"]}
            )

            report = harness.evaluate(val_true, y_pred, extra_cols=extra_cols)
            harness.print_summary(report)
            fold_metrics.append(report["global_rmsle"])
            all_oof_results.append(report["detailed"])

            # Reset val sales in df for next fold if necessary (though folds move back in time)
            # Actually SlidingWindowTS moves backwards, so fold 0 is most recent.
            # So subsequent folds won't see fold 0's "polluted" predictions.

        self.logger.info(
            f"Mean CV RMSLE: {np.mean(fold_metrics):.4f} +/- {np.std(fold_metrics):.4f}"
        )

        # Save OOF Results
        oof_df = pd.concat(all_oof_results, ignore_index=True)
        oof_path = self.output_dir / "oof_predictions.csv"
        oof_df.to_csv(oof_path, index=False)
        self.logger.info(f"OOF predictions saved to {oof_path}")

    def produce_submission(self):
        self.logger.info("Training full model for submission...")
        model_conf = self.config["model"]
        feature_selector = [
            model_conf["features"]["trend"],
            model_conf["features"]["residual"],
        ]

        model = HybridRegressor(
            trend_estimator=Ridge(),
            residual_estimator=LGBMRegressor(n_estimators=200, random_state=42),
            feature_selector=feature_selector,
            gate_feature=model_conf["gate_feature"],
            categorical_features=self.cat_cols,
        )

        model.fit(self.train_df, self.train_df[self.config["validation"]["target_col"]])

        # Log Final Trend Coefficients
        self.logger.info("Final Trend Model Params:")
        self.logger.info(f"  Intercept: {model.trend_estimator_.intercept_:.4f}")
        for feat, coef in zip(
            model_conf["features"]["trend"], model.trend_estimator_.coef_
        ):
            self.logger.info(f"  {feat:20}: {coef:.4f}")

        # Inference context
        full_df = (
            pd.concat([self.train_df, self.test_df], axis=0)
            .sort_values(["date", "store_nbr", "family"])
            .reset_index(drop=True)
        )
        test_dates = sorted(self.test_df["date"].unique())

        self.logger.info("Iterative inference on test set...")
        # iterative_predict updates full_df in-place
        self.iterative_predict(model, full_df, test_dates)

        # Properly extract predictions by matching IDs (FIX ALIGNMENT)
        self.logger.info("Extracting submission results with correct alignment...")
        test_ids = self.test_df["id"].values
        results = full_df[full_df["id"].isin(test_ids)].copy()

        results["sales"] = np.maximum(np.expm1(results["log1p_sales"]), 0)
        submission = results[["id", "sales"]].sort_values("id")
        submission.to_csv(self.output_dir / "submission.csv", index=False)
        self.logger.info(f"Submission saved to {self.output_dir / 'submission.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Baseline Gated Model Trainer")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/baseline_gated.yaml",
        help="Path to the configuration file",
    )
    # Allows it to gracefully ignore other unknown flags like --smoke-test
    args, _ = parser.parse_known_args()

    runner = IterativeGatedRunner(args.config)
    runner.load_data()
    runner.run_cv()
    runner.produce_submission()
