import torch
from torch.utils.data import Dataset


class SalesGNNDataset(Dataset):
    """
    Spatio-Temporal Dataset for GNN.
    Slices history and future windows across all nodes simultaneously.
    """

    def __init__(self, features, labels, is_open, window, horizon):
        """
        Args:
            features: np.array of shape (Nodes, Time, Features)
            labels: np.array of shape (Nodes, Time)
            is_open: np.array of shape (Nodes, Time)
            window: int, input sequence length (T_in)
            horizon: int, forecast horizon (T_out)
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()
        self.is_open = torch.from_numpy(is_open).float()
        self.window = window
        self.horizon = horizon

        # Calculate valid starting indices for windows
        self.num_time_steps = features.shape[1]
        self.num_samples = self.num_time_steps - window - horizon + 1

        if self.num_samples <= 0:
            raise ValueError(
                f"Insufficient time steps ({self.num_time_steps}) for window {window} and horizon {horizon}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # enc_start = idx
        enc_end = idx + self.window
        # dec_start = enc_end
        dec_end = enc_end + self.horizon

        # History (Encoder)
        x_enc = self.features[:, idx:enc_end, :]

        # Future (Decoder) - contains only features known in advance
        x_dec = self.features[:, enc_end:dec_end, :]

        # Target
        y = self.labels[:, enc_end:dec_end]

        # Hard Gating Mask
        mask = self.is_open[:, enc_end:dec_end]

        return x_enc, x_dec, y, mask
