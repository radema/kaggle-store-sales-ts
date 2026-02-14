import torch
import numpy as np
from src.models.gnn.dataset import SalesGNNDataset


def test_gnn_dataset():
    # 2 nodes, 50 time steps, 3 features
    num_nodes = 2
    num_timesteps = 50
    num_features = 3

    # Mock data: (Time, Node, Feature) -> reshape to (Node, Time, Feature)
    # Actually, the dataset might expect it in a specific format.
    # Let's assume (Nodes, Time, Features)
    data = np.random.rand(num_nodes, num_timesteps, num_features).astype(np.float32)
    labels = np.random.rand(num_nodes, num_timesteps).astype(np.float32)
    is_open = np.ones((num_nodes, num_timesteps)).astype(np.float32)

    window = 30
    horizon = 16

    dataset = SalesGNNDataset(
        features=data, labels=labels, is_open=is_open, window=window, horizon=horizon
    )

    # num_samples should be num_timesteps - window - horizon + 1
    # 50 - 30 - 16 + 1 = 5
    assert len(dataset) == 5

    x_enc, x_dec, y, mask = dataset[0]

    # x_enc: (Nodes, Window, Features)
    assert x_enc.shape == (num_nodes, window, num_features)
    # x_dec: (Nodes, Horizon, Features)
    assert x_dec.shape == (num_nodes, horizon, num_features)
    # y: (Nodes, Horizon)
    assert y.shape == (num_nodes, horizon)
    # mask: (Nodes, Horizon)
    assert mask.shape == (num_nodes, horizon)

    assert isinstance(x_enc, torch.Tensor)
