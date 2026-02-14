import torch
import time
import numpy as np
from torch.utils.data import DataLoader
from src.models.gnn.model import SalesGNN
from src.models.gnn.dataset import SalesGNNDataset


def profile():
    device = (
        torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    print(f"Profiling on {device}")

    # Mock Data
    num_stores = 54
    num_families = 33
    num_nodes = num_stores + num_families + (num_stores * num_families)
    num_time = 100
    num_features = 20
    window = 30
    horizon = 16
    batch_size = 32

    features = np.random.randn(num_nodes, num_time, num_features).astype(np.float32)
    labels = np.random.randn(num_nodes, num_time).astype(np.float32)
    is_open = np.ones((num_nodes, num_time)).astype(np.float32)

    dataset = SalesGNNDataset(features, labels, is_open, window, horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Edge index for hierarchy
    edge_index = torch.randint(0, num_nodes, (2, 5000))

    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=num_features,
        hidden_dim=64,
        edge_index=edge_index,
        horizon=horizon,
    ).to(device)

    # Profiling Data Loading
    start = time.time()
    for i, batch in enumerate(loader):
        if i >= 5:
            break
        pass
    print(f"Data Loading (5 batches): {time.time() - start:.4f}s")

    # Profiling Forward Pass
    batch = next(iter(loader))
    x_enc, x_dec, y, mask = [t.to(device) for t in batch]
    x_enc_combined = torch.randn(
        batch_size,
        num_nodes,
        x_enc.size(2),
        model.feature_dim + model.total_embedding_dim,
        device=device,
    )

    # Warmup
    for _ in range(3):
        _ = model(x_enc, x_dec, mask)

    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        _ = model(x_enc, x_dec, mask)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Forward Pass (10 runs): {time.time() - start:.4f}s")

    # Component Profiling: Encoder
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        _ = model.input_norm(x_enc_combined)
        x_enc_flat = x_enc_combined.view(-1, x_enc.size(2), x_enc_combined.size(-1))
        _, h_temporal = model.encoder_gru(x_enc_flat)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Encoder GRU (10 runs): {time.time() - start:.4f}s")

    # Component Profiling: Spatial Mixer
    h_temporal_flat = h_temporal.squeeze(0)
    edge_index_batch = model._get_batch_edge_index(batch_size, device)
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        _ = model.spatial_mixer(h_temporal_flat, edge_index_batch)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Spatial Mixer GATv2 (10 runs): {time.time() - start:.4f}s")

    # Component Profiling: Decoder Rollout
    h_spatial = h_temporal_flat
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        hidden = h_spatial
        node_embed_dec = torch.randn(
            batch_size, num_nodes, model.total_embedding_dim, device=device
        )
        for t in range(horizon):
            x_t = x_dec[:, :, t, :]
            x_t_combined = torch.cat([x_t, node_embed_dec], dim=-1)
            x_t_combined = model.input_norm(x_t_combined)
            x_t_flat = x_t_combined.reshape(-1, x_t_combined.size(-1))
            hidden = model.decoder_gru(x_t_flat, hidden)
            _ = model.fc_out(hidden).view(batch_size, num_nodes)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Decoder Rollout (10 runs): {time.time() - start:.4f}s")

    # Component Profiling: Batch Edge Index
    start = time.time()
    for _ in range(100):
        _ = model._get_batch_edge_index(batch_size, device)
    print(f"Batch Edge Index (100 runs): {time.time() - start:.4f}s")


if __name__ == "__main__":
    profile()
