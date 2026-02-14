import sys
import yaml
import pandas as pd
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parents[1]))

from src.models.gnn.dataset import SalesGNNDataset
from src.utils.logging import get_logger

logger = get_logger("build_gnn_dataset")


def main(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    logger.info("Building GNN binary cache for full history (from 2013)...")

    data_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    output_dir = data_dir / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Data
    logger.info("Loading parquet files...")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    stores_df = pd.read_csv(raw_dir / "stores.csv")

    families = sorted(train_df["family"].unique())
    feature_cols = config["data"]["features"]

    # Build Dataset
    logger.info("Pivoting data to 3D. This might take a minute...")
    dataset = SalesGNNDataset.from_df(
        df=train_df,
        stores_df=stores_df,
        families=families,
        feature_cols=feature_cols,
        window=config["model"]["window"],
        horizon=config["model"]["horizon"],
    )

    # Save to Cache
    logger.info(f"Saving binary tensors to {output_dir}...")
    dataset.save_to_cache(output_dir)

    logger.info("Successfully built GNN dataset cache.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/gnn.yaml")
    args = parser.parse_args()
    main(args.config)
