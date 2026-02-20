import torch
import yaml
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from torch.utils.data import DataLoader
from src.models.gnn.model import SalesGNN
from src.models.gnn.dataset import SalesGNNDataset
from src.models.gnn.trainer import GNNTrainer
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

    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    stores_df = pd.read_csv(raw_dir / "stores.csv")

    if dev_mode:
        logger.info("RUNNING IN DEV MODE (reduced data)")
        config["training"]["epochs"] = 1
        config["data"]["train_start"] = "2017-06-15"

    # Filter by date
    train_df = train_df[train_df["date"] >= config["data"]["train_start"]]

    families = sorted(train_df["family"].unique())
    num_stores = len(stores_df)
    num_families = len(families)

    logger.info(f"Unique Families: {num_families}")
    logger.info(f"Unique Stores: {num_stores}")

    # 4. Prepare Dataset
    window = config["model"]["window"]
    horizon = config["model"]["horizon"]
    val_days = config["training"]["val_days"]
    feature_cols = config["data"]["features"]
    logger.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    cache_dir = data_dir / "gnn"
    cache_valid = False
    
    if cache_dir.exists() and (cache_dir / "features.npy").exists():
        # Quick check for feature dimensionality
        cached_features_shape = np.load(cache_dir / "features.npy", mmap_mode="r").shape
        if len(cached_features_shape) == 3 and cached_features_shape[2] == len(feature_cols):
            logger.info(f"Loading GNN Dataset from cache: {cache_dir}")
            full_dataset = SalesGNNDataset.load_from_cache(
                cache_dir, window, horizon, mmap=True
            )
            cache_valid = True
        else:
            logger.warning(f"Cache feature mismatch: expected {len(feature_cols)}, found {cached_features_shape[2]}. Rebuilding...")

    if not cache_valid:
        logger.info("Cache not found or invalid. Building GNN Dataset from related scripts (Slow)...")
        full_dataset = SalesGNNDataset.from_df(
            df=train_df,
            stores_df=stores_df,
            families=families,
            feature_cols=feature_cols,
            window=window,
            horizon=horizon,
        )
        logger.info(f"Saving GNN Dataset to cache: {cache_dir}")
        full_dataset.save_to_cache(cache_dir)

    # 4.1 Data Integrity Check
    if torch.isnan(full_dataset.features).any() or torch.isinf(full_dataset.features).any():
        logger.warning("NaNs/Infs found in cache. Attempting local zero-imputation...")
        full_dataset.features = torch.nan_to_num(full_dataset.features, nan=0.0, posinf=0.0, neginf=0.0)

    if torch.isnan(full_dataset.labels).any():
        full_dataset.labels = torch.nan_to_num(full_dataset.labels, nan=0.0)

    logger.info("Data integrity verified (Internal imputation applied).")

    num_cache_steps = full_dataset.features.shape[0]
    if dev_mode:
        dev_start_idx = max(0, num_cache_steps - 100)
        full_dataset.features = full_dataset.features[dev_start_idx:, :, :]
        full_dataset.labels = full_dataset.labels[dev_start_idx:, :]
        full_dataset.is_open = full_dataset.is_open[dev_start_idx:, :]
        num_cache_steps = full_dataset.features.shape[0]
        logger.info(f"DEV MODE: Reduced cache to last {num_cache_steps} steps.")

    train_limit_idx = num_cache_steps - val_days
    logger.info(f"Cache time steps: {num_cache_steps}. Val days: {val_days}")

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

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=False,
    )
    val_loader = DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=False)

    logger.info(f"Train batches: {len(train_loader)} (Samples: {len(train_dataset)})")
    logger.info(f"Val batches: {len(val_loader)} (Samples: {len(val_dataset)})")

    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=len(feature_cols),
        hidden_dim=config["model"]["hidden_dim"],
        horizon=horizon,
        num_layers=config["model"].get("num_layers", 4),
        embedding_dim=config["model"].get("embedding_dim", 16),
        dist_matrix=full_dataset.dist_matrix,
        static_feat_dim=full_dataset.static_node_features.shape[1] if full_dataset.static_node_features is not None else 0,
        use_checkpointing=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["lr"])

    trainer = GNNTrainer(
        model=model,
        optimizer=optimizer,
        alpha=config["model"]["alpha"],
        patience=config["training"]["patience"],
        logger=logger,
        static_node_features=full_dataset.static_node_features
    )

    logger.info("Starting training loop...")
    history = trainer.fit(train_loader, val_loader, epochs=config["training"]["epochs"])

    logger.info("--- Training Summary ---")
    logger.info(f"Best Val Loss: {trainer.best_val_loss:.6f}")

    output_dir = Path("artifacts/gnn")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model_best.pt")

    # 9. Inference & Submission
    logger.info("Starting inference for submission...")
    last_train_date = train_df["date"].max()
    start_history_date = last_train_date - pd.Timedelta(days=window - 1)
    history_df = train_df[train_df["date"] >= start_history_date].copy()

    inference_test_df = test_df.copy()
    inference_test_df["log1p_sales"] = 0.0
    inference_test_df["sales"] = 0.0

    combined_inf_df = pd.concat([history_df, inference_test_df], axis=0).sort_values(["date", "store_nbr", "family"])

    test_dataset = SalesGNNDataset.from_df(
        df=combined_inf_df,
        stores_df=stores_df,
        families=families,
        feature_cols=feature_cols,
        window=window,
        horizon=horizon,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    logger.info("Running model forward pass...")
    preds_log = trainer.predict(test_loader)  # (1, Nodes, Horizon)

    # The SalesGNN model returns ONLY series nodes by default
    series_preds_log = preds_log[0, :, :] 
    preds_final = torch.expm1(series_preds_log.t()).flatten().numpy()
    preds_final = np.maximum(preds_final, 0)

    if len(preds_final) != len(test_df):
        logger.error(f"Prediction count mismatch! Preds: {len(preds_final)}, Test: {len(test_df)}")
    else:
        submission = test_df[["id"]].copy()
        submission["sales"] = preds_final
        submission = submission.sort_values("id")
        sub_path = output_dir / "submission.csv"
        submission.to_csv(sub_path, index=False)
        logger.info(f"Submission saved to {sub_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/gnn.yaml")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    main(args.config, args.dev)
