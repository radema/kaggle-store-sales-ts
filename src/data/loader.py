import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any
from src.utils.logging import get_logger


class DataLoader:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema or {}
        # Avoid creating multiple logger instances if not needed, but here simple is fine
        self.logger = get_logger("DataLoader")

    def load(self, path: Path, **kwargs) -> pd.DataFrame:
        self.logger.info(f"Loading data from {path}...")
        try:
            # We load the data first, then ensure date is converted if it exists.
            df = pd.read_csv(
                path,
                dtype=self.schema,
                low_memory=False,
                **kwargs,
            )
        except FileNotFoundError:
            self.logger.error(f"File not found: {path}")
            raise

        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["date"]
        ):
            df["date"] = pd.to_datetime(df["date"])

        self.logger.info(f"Loaded dataframe with shape: {df.shape}")
        return df

    def load_unified(self, config: Dict[str, Any]) -> pd.DataFrame:
        self.logger.info("Loading and unifying Train and Test datasets.")
        sources = config.get("data_sources", {})

        train_path_str = sources.get("train")
        test_path_str = sources.get("test")

        if not train_path_str or not test_path_str:
            self.logger.error("Config missing 'train' or 'test' paths in data_sources.")
            raise ValueError(
                "Config must specify 'train' and 'test' paths in 'data_sources'."
            )

        train_path = Path(train_path_str)
        test_path = Path(test_path_str)

        train_df = self.load(train_path)
        test_df = self.load(test_path)

        train_df["split"] = "train"
        test_df["split"] = "test"

        # Concatenate
        unified_df = pd.concat([train_df, test_df], ignore_index=True)
        self.logger.info(f"Unified dataframe shape: {unified_df.shape}")
        return unified_df

    def load_metadata(self, config: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Loads other specified sources as a dictionary of DataFrames."""
        self.logger.info("Loading metadata sources.")
        sources = config.get("data_sources", {})
        metadata = {}

        # We skip train/test as they are handled by load_unified
        keys_to_skip = ["train", "test"]

        for key, path_str in sources.items():
            if key in keys_to_skip:
                continue

            self.logger.info(f"Loading metadata: {key}")
            metadata[key] = self.load(Path(path_str))

        return metadata
