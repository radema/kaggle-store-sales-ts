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
        hidden_dim=32,  # Matching config
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

    # Warmup
    for _ in range(3):
        _ = model(x_enc, x_dec, mask)

    # Mixed Precision Info
    device_type = (
        "cuda" if "cuda" in str(device) else "mps" if "mps" in str(device) else "cpu"
    )
    autocast_enabled = device_type != "cpu"

    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        with torch.autocast(device_type=device_type, enabled=autocast_enabled):
            _ = model(x_enc, x_dec, mask)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Forward Pass (10 runs): {time.time() - start:.4f}s")

    seq_len = x_enc.size(2)

    # Component Profiling: Light Encoder
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        with torch.autocast(device_type=device_type, enabled=autocast_enabled):
            x_enc_flat = x_enc.reshape(-1, seq_len, num_features)
            x_enc_flat = model.input_norm(x_enc_flat)
            _, h_temporal = model.encoder_gru_features(x_enc_flat)
            h_temporal = h_temporal.squeeze(0)

            # Node projection logic
            node_embed_temp = model._get_node_embeddings(device)
            node_proj = model.node_projector(node_embed_temp[:num_nodes])
            node_proj_expanded = (
                node_proj.view(1, num_nodes, -1)
                .expand(batch_size, num_nodes, -1)
                .reshape(batch_size * num_nodes, -1)
            )
            h_context = h_temporal + node_proj_expanded
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Light Encoder + Context (10 runs): {time.time() - start:.4f}s")

    # Component Profiling: Spatial Mixer
    edge_index_batch = model._get_batch_edge_index(batch_size, num_nodes, device)
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        with torch.autocast(device_type=device_type, enabled=autocast_enabled):
            _ = model.spatial_mixer(h_context, edge_index_batch)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Spatial Mixer GATv2 (10 runs): {time.time() - start:.4f}s")

    # Component Profiling: Vectorized Decoder
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(10):
        with torch.autocast(device_type=device_type, enabled=autocast_enabled):
            h_decoder_init = h_context.unsqueeze(0)
            x_dec_flat = x_dec.reshape(-1, horizon, num_features)
            x_dec_flat = model.input_norm(x_dec_flat)
            decoder_out, _ = model.decoder_gru(x_dec_flat, h_decoder_init)
            node_proj_dec = node_proj_expanded.unsqueeze(1)
            final_context = decoder_out + node_proj_dec
            _ = model.fc_out(final_context)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Vectorized Decoder (10 runs): {time.time() - start:.4f}s")


if __name__ == "__main__":
    profile()
