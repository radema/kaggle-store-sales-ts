import sys
import yaml
import torch
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

    # Time split: We need a buffer to ensure val_dataset has enough history (window)
    all_dates = sorted(train_df["date"].unique())
    num_time_steps = len(all_dates)
    train_limit_idx = num_time_steps - val_days
    train_limit_date = all_dates[train_limit_idx]

    logger.info(f"Splitting data at {train_limit_date} for validation.")

    # Create complete dataset and then slice
    # This is more efficient than pivoting twice
    logger.info("Building GNN Dataset from related scripts...")
    full_dataset = SalesGNNDataset.from_df(
        df=train_df,
        stores_df=stores_df,
        families=families,
        feature_cols=feature_cols,
        window=window,
        horizon=horizon,
    )

    # Manual slicing for train/val
    # Note: SalesGNNDataset handles the windowing, we just need to provide the right time slices
    # However, our SalesGNNDataset currently stores full tensors.
    # To split by time without re-pivoting, we can just slice the tensors.

    # train_data: [:, :train_limit_idx, :]
    train_dataset = SalesGNNDataset(
        features=full_dataset.features[:, :train_limit_idx, :].numpy(),
        labels=full_dataset.labels[:, :train_limit_idx].numpy(),
        is_open=full_dataset.is_open[:, :train_limit_idx].numpy(),
        window=window,
        horizon=horizon,
    )

    # val_data: [:, train_limit_idx - window:, :]
    val_dataset = SalesGNNDataset(
        features=full_dataset.features[:, train_limit_idx - window :, :].numpy(),
        labels=full_dataset.labels[:, train_limit_idx - window :].numpy(),
        is_open=full_dataset.is_open[:, train_limit_idx - window :].numpy(),
        window=window,
        horizon=horizon,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config["training"]["batch_size"], shuffle=True
    )
    val_loader = DataLoader(val_dataset, batch_size=1)

    logger.info(f"Train batches: {len(train_loader)} (Samples: {len(train_dataset)})")
    logger.info(f"Val batches: {len(val_loader)} (Samples: {len(val_dataset)})")

    # 5. Initialize Model
    total_nodes = full_dataset.features.size(0)
    model = SalesGNN(
        num_nodes=total_nodes,
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

    final_train_loss = history["train_loss"][-1]
    logger.info(f"Final Train Loss: {final_train_loss:.6f}")

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gnn.yaml")
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    main(args.config, args.dev)
