import pandas as pd
from pathlib import Path
from typing import Optional, Dict, Any


class DataLoader:
    def __init__(self, schema: Optional[Dict[str, Any]] = None):
        self.schema = schema or {}

    def load(self, path: Path, **kwargs) -> pd.DataFrame:
        # We load the data first, then ensure date is converted if it exists.
        # This avoids ValueError in read_csv if 'date' is not present in the file.
        df = pd.read_csv(
            path,
            dtype=self.schema,
            **kwargs,
        )

        if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(
            df["date"]
        ):
            df["date"] = pd.to_datetime(df["date"])

        return df
