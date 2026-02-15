import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from typing import Tuple


class SalesGNNDataset(Dataset):
    """
    Spatio-Temporal Dataset for GNN.
    Slices history and future windows across all nodes simultaneously.
    """

    def __init__(self, features, labels, is_open, window, horizon):
        """
        Args:
            features: np.array or torch.Tensor of shape (Time, Nodes, Features)
            labels: np.array or torch.Tensor of shape (Time, Nodes)
            is_open: np.array or torch.Tensor of shape (Time, Nodes)
            window: int, input sequence length (T_in)
            horizon: int, forecast horizon (T_out)
        """
        # Ensure we have tensors
        if isinstance(features, np.ndarray):
            self.features = torch.from_numpy(features).float()
        else:
            self.features = features.float()

        if isinstance(labels, np.ndarray):
            self.labels = torch.from_numpy(labels).float()
        else:
            self.labels = labels.float()

        if isinstance(is_open, np.ndarray):
            self.is_open = torch.from_numpy(is_open).float()
        else:
            self.is_open = is_open.float()

        self.window = window
        self.horizon = horizon

        # Calculate valid starting indices for windows
        # Current layout: (Time, Nodes, ...)
        self.num_time_steps = self.features.shape[0]
        self.num_samples = self.num_time_steps - window - horizon + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Insufficient time steps ({self.num_time_steps}) for window {window} and horizon {horizon}. "
                f"Need at least {window + horizon} steps, got {self.num_time_steps}."
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        enc_end = idx + self.window
        dec_end = enc_end + self.horizon

        # History (Encoder): (Window, Nodes, Features) -> (Nodes, Window, Features)
        x_enc = self.features[idx:enc_end, :, :].transpose(0, 1)

        # Target: (Horizon, Nodes) -> (Nodes, Horizon)
        y = self.labels[enc_end:dec_end, :].transpose(0, 1)

        # Hard Gating Mask: (Horizon, Nodes) -> (Nodes, Horizon)
        mask = self.is_open[enc_end:dec_end, :].transpose(0, 1)

        return x_enc, y, mask

    def save_to_cache(self, directory):
        """Saves current tensors to a directory for later loading."""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "features.npy", self.features.numpy())
        np.save(path / "labels.npy", self.labels.numpy())
        np.save(path / "is_open.npy", self.is_open.numpy())

    @classmethod
    def load_from_cache(cls, directory, window, horizon, mmap=False):
        """Loads tensors from a directory, optionally using memory mapping."""
        path = Path(directory)
        mmap_mode = "r" if mmap else None

        features = np.load(path / "features.npy", mmap_mode=mmap_mode)
        labels = np.load(path / "labels.npy", mmap_mode=mmap_mode)
        is_open = np.load(path / "is_open.npy", mmap_mode=mmap_mode)

        return cls(features, labels, is_open, window, horizon)

    @staticmethod
    def from_df(df, stores_df, families, feature_cols, window, horizon, use_hub_nodes=False):
        """
        Helper to construct the 3D tensors from a flat DataFrame.
        Ordering: Lexical Sort of Stores -> Families.
        Layout: (Time, Nodes, Features)
        """
        from sklearn.preprocessing import StandardScaler

        # 1. Lexical Sorting for alignment with submission logic
        stores_sorted = stores_df.sort_values("store_nbr")
        families_sorted = sorted(families)
        
        num_stores = len(stores_sorted)
        num_families = len(families_sorted)
        num_series = num_stores * num_families
        
        # Hub nodes (summary nodes) are now optional to eliminate MPS bottlenecks
        series_offset = (num_stores + num_families) if use_hub_nodes else 0
        total_nodes = series_offset + num_series

        all_dates = sorted(df["date"].unique())
        num_time_steps = len(all_dates)
        date_to_idx = {d: i for i, d in enumerate(all_dates)}

        num_features = len(feature_cols)

        # 2. Allocate Tensors
        features_arr = np.zeros(
            (num_time_steps, total_nodes, num_features), dtype=np.float32
        )
        labels_arr = np.zeros((num_time_steps, total_nodes), dtype=np.float32)
        is_open_arr = np.ones(
            (num_time_steps, total_nodes), dtype=np.float32
        )

        # 3. Build Maps
        store_map = {nbr: i for i, nbr in enumerate(stores_sorted["store_nbr"])}
        family_map = {name: i for i, name in enumerate(families_sorted)}

        # 4. Fill Series Data (Lexical Order Guaranteed by loop and node_idx)
        for (s_nbr, f_name), group in df.groupby(["store_nbr", "family"]):
            if s_nbr not in store_map or f_name not in family_map:
                continue

            s_idx = store_map[s_nbr]
            f_idx = family_map[f_name]
            # Lexical index: Store index first, then Family index
            node_idx = series_offset + (s_idx * num_families) + f_idx

            t_indices = [date_to_idx[d] for d in group["date"]]
            labels_arr[t_indices, node_idx] = group["log1p_sales"].fillna(0).values
            is_open_arr[t_indices, node_idx] = (group["is_closed"] == 0).astype(int)
            features_arr[t_indices, node_idx, :] = group[feature_cols].fillna(0).values

        # 5. Feature Scaling (Exogenous Features)
        # We scale features across the temporal dimension for each node if needed, 
        # but standard way is across all observations per feature.
        orig_shape = features_arr.shape
        flat_features = features_arr.reshape(-1, num_features)
        scaler = StandardScaler()
        # Only fit on non-zero values if needed, but here we just do global scaling
        features_arr = scaler.fit_transform(flat_features).reshape(orig_shape)

        # 6. Spatial Distance Matrix (Optional Initialization for Model)
        # Based on cluster similarity: dist=0 if same cluster, 1 otherwise
        clusters = stores_sorted["cluster"].values
        dist_matrix = (clusters[:, None] != clusters[None, :]).astype(float)
        
        dataset = SalesGNNDataset(features_arr, labels_arr, is_open_arr, window, horizon)
        dataset.scaler = scaler
        dataset.dist_matrix = torch.from_numpy(dist_matrix).float()
        dataset.num_stores = num_stores
        dataset.num_families = num_families
        dataset.series_offset = series_offset
        
        return dataset
