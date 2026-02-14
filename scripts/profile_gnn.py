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
    # num_nodes = 54 + 33 + 1782 = 1869
    num_nodes = num_stores + num_families + (num_stores * num_families)
    num_time = 200  # Longer to ensure enough samples
    num_features = 20
    window = 30
    horizon = 16
    batch_size = 32

    # IMPORTANT: New T-major layout (Time, Nodes, Features)
    features = np.random.randn(num_time, num_nodes, num_features).astype(np.float32)
    labels = np.random.randn(num_time, num_nodes).astype(np.float32)
    is_open = np.ones((num_time, num_nodes)).astype(np.float32)

    dataset = SalesGNNDataset(features, labels, is_open, window, horizon)
    # Use persistent workers to simulate real training overhead
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

    # x_enc should be (B, N, T, F) after dataset.transpose
    print(f"Input shapes: x_enc {x_enc.shape}, x_dec {x_dec.shape}")

    # Warmup
    for _ in range(3):
        _ = model(x_enc, x_dec, mask)

    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        _ = model(x_enc, x_dec, mask)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Forward Pass (10 runs): {time.time() - start:.4f}s")

    seq_len = x_enc.size(2)
    node_embed = model._get_node_embeddings(device)
    embedding_dim = model.total_embedding_dim

    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        x_enc_flat = x_enc.reshape(-1, seq_len, num_features)
        node_embed_expanded = (
            node_embed.view(1, num_nodes, 1, embedding_dim)
            .expand(batch_size, num_nodes, seq_len, embedding_dim)
            .reshape(batch_size * num_nodes, seq_len, embedding_dim)
        )
        x_enc_combined = torch.cat([x_enc_flat, node_embed_expanded], dim=-1)
        x_enc_combined = model.input_norm(x_enc_combined)
        _, h_temporal = model.encoder_gru(x_enc_combined)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Encoder GRU Optimized (10 runs): {time.time() - start:.4f}s")

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
        node_embed_dec_flat = node_embed.repeat(batch_size, 1)
        for t in range(horizon):
            x_t = x_dec[:, :, t, :].reshape(-1, num_features)
            x_t_combined = torch.cat([x_t, node_embed_dec_flat], dim=-1)
            x_t_combined = model.input_norm(x_t_combined)
            hidden = model.decoder_gru(x_t_combined, hidden)
            _ = model.fc_out(hidden)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Decoder Rollout Optimized (10 runs): {time.time() - start:.4f}s")


if __name__ == "__main__":
    profile()
