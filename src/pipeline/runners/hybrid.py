from typing import Dict, Optional
from pathlib import Path
from src.pipeline.config import load_config
from src.data.loader import DataLoader
from src.pipeline.factory import FeaturePipelineFactory
from src.models.hybrid import HybridRegressor
from src.validation.splitters import SlidingWindowTS
from src.validation.harness import EvaluationSuite
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from src.pipeline.documentation import ModelCardGenerator
import pandas as pd
import numpy as np
import importlib
import joblib
import json


class HybridRunner:
    """
    Runner for the Hybrid Baseline experiment.
    Responsible for loading data, building the pipeline, and executing CV/Inference.
    """

    def __init__(self, config_path: str, run_name: Optional[str] = None):
        self.config = load_config(config_path)
        self.run_name = run_name or self.config.get("run_name", "unnamed_run")
        self.data_loader = DataLoader()
        self.data: Dict[str, pd.DataFrame] = {}

    def load_data(self):
        """Loads all data files specified in the configuration."""
        data_conf = self.config.get("data", {})

        # Mapping of config keys to loader internal keys
        mapping = {
            "train_path": "train",
            "test_path": "test",
            "stores_path": "stores",
            "oil_path": "oil",
            "holidays_path": "holidays",
        }

        for config_key, data_key in mapping.items():
            if config_key in data_conf:
                path = Path(data_conf[config_key])
                print(f"Loading {data_key} from {path}...")
                self.data[data_key] = self.data_loader.load(path)

        return self.data

    def _build_pipeline(self) -> TransformedTargetRegressor:
        """Assembles the full pipeline with target transformation."""
        # 1. Feature Engineering
        factory = FeaturePipelineFactory()
        feat_pipeline = factory.create_from_config(self.config.get("pipeline", {}))

        # 2. Hybrid Model
        model_conf = self.config.get("model", {})

        # Helper to instantiate estimators from config
        def _get_estimator(conf):
            if not conf:
                return None
            cls_path = conf["class"]
            params = conf.get("params", {})
            module_name, class_name = cls_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            return getattr(module, class_name)(**params)

        trend_est = _get_estimator(model_conf.get("trend_estimator"))
        resid_est = _get_estimator(model_conf.get("residual_estimator"))

        hybrid_model = HybridRegressor(
            trend_estimator=trend_est,
            residual_estimator=resid_est,
            feature_selector=model_conf.get("feature_selector"),
        )

        # 3. Full Sklearn Pipeline (Flattened)
        all_steps = list(feat_pipeline.steps)
        all_steps.append(("model", hybrid_model))
        full_pipe = Pipeline(all_steps)

        # 4. Wrap with Target Transformer
        target_transform = self.config.get("validation", {}).get("target_transform")
        if target_transform == "log1p":
            return TransformedTargetRegressor(
                regressor=full_pipe, func=np.log1p, inverse_func=np.expm1
            )

        return TransformedTargetRegressor(regressor=full_pipe)

    def run_cv(self):
        """Executes the cross-validation loop."""
        if "train" not in self.data:
            self.load_data()

        train_df = self.data["train"]
        val_conf = self.config.get("validation", {})

        splitter = SlidingWindowTS(
            train_days=val_conf.get("train_days", 365),
            val_days=val_conf.get("val_days", 15),
            n_folds=val_conf.get("n_folds", 5),
        )

        harness = EvaluationSuite()
        oof_results = []

        X = train_df.copy()
        y = train_df["sales"]

        for fold, (train_idx, val_idx) in enumerate(splitter.split(X)):
            print(f"Executing Fold {fold + 1}...")
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train = y.iloc[train_idx]

            pipeline = self._build_pipeline()
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_val)

            # Use harness to evaluate this fold
            # Harness expects a df with sales and metadata
            val_df = train_df.iloc[val_idx].copy()
            report = harness.evaluate(val_df, y_pred)
            oof_results.append(report["detailed"])

        # Combine all OOF results
        self.oof_detailed = pd.concat(oof_results).sort_index()
        self.metrics = harness.evaluate(
            self.oof_detailed[["sales", "store_nbr", "family", "date"]],
            self.oof_detailed["sales_pred"],
        )

        print(f"CV Global RMSLE: {self.metrics['global_rmsle']:.4f}")
        return self.metrics

    def train_full(self):
        """Fits the model on the entire training set."""
        if "train" not in self.data:
            self.load_data()

        print("Training final model on full dataset...")
        X = self.data["train"].copy()
        y = self.data["train"]["sales"]

        self.full_pipeline = self._build_pipeline()
        self.full_pipeline.fit(X, y)
        print("Final training complete.")
        return self.full_pipeline

    def predict_test(self):
        """Generates predictions for the test set."""
        if not hasattr(self, "full_pipeline"):
            self.train_full()

        if "test" not in self.data:
            self.load_data()

        print("Generating test predictions...")
        train_df = self.data["train"]
        test_df = self.data["test"].copy()

        # 1. Prepare Inference context (Concatenate Train + Test for Lags)
        test_df["sales"] = 0  # Dummy value for pipeline consistency
        X_inf = pd.concat([train_df, test_df], axis=0, sort=False)
        X_inf = X_inf.sort_values(["date", "store_nbr", "family"])

        # 2. Predict on the whole combined set
        preds_all = self.full_pipeline.predict(X_inf)

        # 3. Attach results to X_inf to easy extraction
        X_inf["sales_pred"] = preds_all

        # 4. Extract only the test rows (where id exists and belongs to test)
        # Kaggle test set has unique IDs that don't overlap with train
        test_ids = self.data["test"]["id"].values
        submission = X_inf[X_inf["id"].isin(test_ids)].copy()

        # Ensure order matches test_df if needed, though submission usually doesn't care
        submission = submission.sort_values("id")

        # Cleanup
        submission = submission[["id", "sales_pred"]].rename(
            columns={"sales_pred": "sales"}
        )
        submission["sales"] = np.maximum(submission["sales"], 0)

        self.test_predictions = submission
        return submission

    def save_artifacts(self, output_dir: str):
        """Saves all run artifacts."""
        path = Path(output_dir) / self.run_name
        path.mkdir(parents=True, exist_ok=True)

        print(f"Saving artifacts to {path}...")

        # 1. Submission
        if hasattr(self, "test_predictions"):
            self.test_predictions.to_csv(path / "submission.csv", index=False)

        # 2. Metrics
        if hasattr(self, "metrics"):
            # Serialize per_store and per_family which are series
            metrics_to_save = self.metrics.copy()
            metrics_to_save["per_store"] = metrics_to_save["per_store"].to_dict()
            metrics_to_save["per_family"] = metrics_to_save["per_family"].to_dict()
            del metrics_to_save["detailed"]  # Don't save full DF in JSON

            with open(path / "metrics.json", "w") as f:
                json.dump(metrics_to_save, f, indent=4)

        # 3. OOF Residuals
        if hasattr(self, "oof_detailed"):
            self.oof_detailed.to_csv(path / "oof_residuals.csv", index=False)

        # 4. Pipeline
        if hasattr(self, "full_pipeline"):
            joblib.dump(self.full_pipeline, path / "full_pipeline.pkl")

        # 5. Model Card
        doc_gen = ModelCardGenerator()
        # Simple pipeline summary
        pipe_summary = (
            str(self.full_pipeline.regressor_)
            if hasattr(self, "full_pipeline")
            else "Not trained on full data"
        )
        card = doc_gen.generate(self.config, pipe_summary)
        with open(path / "model_card.md", "w") as f:
            f.write(card)

        # 6. Config Copy
        with open(path / "config.yaml", "w") as f:
            import yaml

            yaml.dump(self.config, f)

        print("Artifacts saved successfully.")

    def run(self, output_dir: str = "artifacts/runs"):
        """Orchestrates the full experiment."""
        self.load_data()
        self.run_cv()
        self.train_full()
        self.predict_test()
        self.save_artifacts(output_dir)
