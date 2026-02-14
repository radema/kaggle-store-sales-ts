import sys
import yaml
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from src.models.gnn.model import SalesGNN
from src.models.gnn.trainer import GNNTrainer
from src.models.gnn.dataset import SalesGNNDataset
from src.models.gnn.graph import build_adjacency
from src.utils.logging import get_logger

logger = get_logger("train_gnn")


def main(config_path, dev_mode=False):
    # 1. Load Configuration
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Loading configuration...")

    # 2. Load Data
    data_dir = Path("data/processed")
    raw_dir = Path("data/raw")

    # We use the processed parquet files as our source of truth for features and log-scales
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    stores_df = pd.read_csv(raw_dir / "stores.csv")

    if dev_mode:
        logger.info("RUNNING IN DEV MODE (reduced data)")
        config["training"]["epochs"] = 1
        # Need at least window + horizon + buffer
        # 30 + 16 = 46 days. Let's take 60 days.
        config["data"]["train_start"] = "2017-06-15"

    # Filter by date
    train_df = train_df[train_df["date"] >= config["data"]["train_start"]]

    families = sorted(train_df["family"].unique())
    num_stores = len(stores_df)
    num_families = len(families)

    logger.info(f"Unique Families: {num_families}")
    logger.info(f"Unique Stores: {num_stores}")

    # 3. Build Adjacency
    adj_dicts = build_adjacency(stores_df, families)
    edge_index = torch.cat([adj_dicts["spatial"], adj_dicts["hierarchy"]], dim=1)

    # 4. Prepare Dataset
    window = config["model"]["window"]
    horizon = config["model"]["horizon"]
    val_days = config["training"]["val_days"]
    feature_cols = config["data"]["features"]

    cache_dir = data_dir / "gnn"
    if cache_dir.exists() and (cache_dir / "features.npy").exists():
        logger.info(f"Loading GNN Dataset from cache: {cache_dir}")
        full_dataset = SalesGNNDataset.load_from_cache(
            cache_dir, window, horizon, mmap=True
        )
    else:
        logger.info(
            "Cache not found. Building GNN Dataset from related scripts (Slow)..."
        )
        # Filters are applied inside from_df if we use train_df as input
        full_dataset = SalesGNNDataset.from_df(
            df=train_df,
            stores_df=stores_df,
            families=families,
            feature_cols=feature_cols,
            window=window,
            horizon=horizon,
        )

    # 4.1 Data Integrity Check (on final tensors)
    if (
        torch.isnan(full_dataset.features).any()
        or torch.isinf(full_dataset.features).any()
    ):
        logger.warning("NaNs/Infs found in cache. Attempting local zero-imputation...")
        full_dataset.features = torch.nan_to_num(
            full_dataset.features, nan=0.0, posinf=0.0, neginf=0.0
        )

    if torch.isnan(full_dataset.labels).any():
        full_dataset.labels = torch.nan_to_num(full_dataset.labels, nan=0.0)

    logger.info("Data integrity verified (Internal imputation applied).")

    # Time split: Robust slicing relative to the ACTUAL cache size
    # New T-major layout: (Time, Nodes, Features)
    num_cache_steps = full_dataset.features.shape[0]
    train_limit_idx = num_cache_steps - val_days

    logger.info(f"Cache time steps: {num_cache_steps}. Val days: {val_days}")
    logger.info(
        f"Slicing cache: Train until {train_limit_idx}, Val from {train_limit_idx - window}"
    )

    # Manual slicing for train/val - T-major indexing
    train_dataset = SalesGNNDataset(
        features=full_dataset.features[:train_limit_idx, :, :],
        labels=full_dataset.labels[:train_limit_idx, :],
        is_open=full_dataset.is_open[:train_limit_idx, :],
        window=window,
        horizon=horizon,
    )

    val_dataset = SalesGNNDataset(
        features=full_dataset.features[train_limit_idx - window :, :, :],
        labels=full_dataset.labels[train_limit_idx - window :, :],
        is_open=full_dataset.is_open[train_limit_idx - window :, :],
        window=window,
        horizon=horizon,
    )

    # Performance: Use num_workers and pin_memory (if not on MPS/CPU only)
    # Note: On Mac MPS, pin_memory can sometimes be problematic, but usually fine
    num_workers = 4 if not dev_mode else 0

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    logger.info(f"Train batches: {len(train_loader)} (Samples: {len(train_dataset)})")
    logger.info(f"Val batches: {len(val_loader)} (Samples: {len(val_dataset)})")

    # 5. Initialize Model
    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=len(feature_cols),
        hidden_dim=config["model"]["hidden_dim"],
        edge_index=edge_index,
        horizon=horizon,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    # 6. Training with Logging & Callbacks (Early Stopping)
    trainer = GNNTrainer(
        model=model,
        optimizer=optimizer,
        alpha=config["model"]["alpha"],
        patience=config["training"]["patience"],
        logger=logger,
    )

    logger.info("Starting training loop...")
    history = trainer.fit(train_loader, val_loader, epochs=config["training"]["epochs"])

    # 7. Final Report
    logger.info("--- Training Summary ---")
    logger.info(f"Total Epochs: {len(history['train_loss'])}")
    logger.info(f"Best Val Loss: {trainer.best_val_loss:.6f}")

    if history["train_loss"]:
        final_train_loss = history["train_loss"][-1]
        logger.info(f"Final Train Loss: {final_train_loss:.6f}")
    else:
        logger.warning("No epochs completed. Summary skipped.")

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model Complexity: {num_params} trainable parameters")

    # 8. Save Artifacts
    output_dir = Path("artifacts/gnn")
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_dir / "model_best.pt")

    # Save training history for analysis
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "training_history.csv", index=False)

    logger.info(f"Artifacts saved to {output_dir}")

    # 9. Inference & Submission
    logger.info("Starting inference for submission...")

    # We need the last 'window' days of training data to provide history for the test set
    last_train_date = train_df["date"].max()
    start_history_date = last_train_date - pd.Timedelta(days=window - 1)

    history_df = train_df[train_df["date"] >= start_history_date].copy()

    # Prepare test_df: must have the same columns and order as train_df
    # test.parquet is already processed, but we need dummy sales
    inference_test_df = test_df.copy()
    inference_test_df["log1p_sales"] = 0.0
    inference_test_df["sales"] = 0.0

    # Combine history and test
    combined_inf_df = pd.concat([history_df, inference_test_df], axis=0).sort_values(
        ["date", "store_nbr", "family"]
    )

    # Build a single-sample dataset for inference
    logger.info("Building inference dataset...")
    test_dataset = SalesGNNDataset.from_df(
        df=combined_inf_df,
        stores_df=stores_df,
        families=families,
        feature_cols=feature_cols,
        window=window,
        horizon=horizon,
    )

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Predict
    logger.info("Running model forward pass...")
    preds_log = trainer.predict(test_loader)  # (1, Nodes, Horizon)

    # Invert log transform: exp(x) - 1
    # We only care about the Series nodes (Store-Family pairs)
    # Recall node indexing: [Stores, Families, Series]
    series_offset = num_stores + num_families
    series_preds_log = preds_log[0, series_offset:, :]  # (NumSeries, Horizon)

    # Flatten to match test_df order (Date then Store then Family)
    # The dataset was built using groupby(["store_nbr", "family"]) which matches our node order
    # test_df is sorted by date, then store, then family.
    # Our series_preds_log is (Series, Time).
    # We need to transpose to (Time, Series) and then flatten to match test_df.
    # Actually, test_df has Dates as outer loop, then Stores, then Families.
    # our tensor is [Node, Time]. If we flatten it as is (C-order), it's Node0_T0, Node0_T1...
    # We want T0_Node0, T0_Node1, ..., T0_NodeN, T1_Node0...

    # Correct reshaping:
    # 1. Transpose (Series, Horizon) -> (Horizon, Series)
    # 2. Flatten -> (Horizon * Series,)
    preds_final = torch.expm1(series_preds_log.t()).flatten().numpy()
    preds_final = np.maximum(preds_final, 0)  # Clip negative sales

    # Final Alignment Check
    if len(preds_final) != len(test_df):
        logger.error(
            f"Prediction count mismatch! Preds: {len(preds_final)}, Test: {len(test_df)}"
        )
    else:
        # Create submission
        submission = test_df[["id"]].copy()
        submission["sales"] = preds_final
        submission = submission.sort_values("id")

        sub_path = output_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gnn.yaml")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    main(args.config, args.dev)
