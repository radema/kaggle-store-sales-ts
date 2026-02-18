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
    num_series = num_stores * num_families
    num_time = 200
    num_features = 20
    window = 30
    horizon = 16
    batch_size = 32

    # (Time, Nodes, Features)
    features = np.random.randn(num_time, num_series, num_features).astype(np.float32)
    labels = np.random.randn(num_time, num_series).astype(np.float32)
    is_open = np.ones((num_time, num_series)).astype(np.float32)

    dataset = SalesGNNDataset(features, labels, is_open, window, horizon)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=num_features,
        hidden_dim=32,
        horizon=horizon,
        num_layers=4
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
    x_enc, y, mask = [t.to(device) for t in batch]

    print(f"Input shapes: x_enc={x_enc.shape}, y={y.shape}, mask={mask.shape}")

    # Warmup
    for _ in range(3):
        _ = model(x_enc, mask)

    # Mixed Precision Info
    device_type = (
        "cuda" if "cuda" in str(device) else "mps" if "mps" in str(device) else "cpu"
    )
    # MPS supports autocast only in very recent versions, often safer to use manual scaling or skip if not Nvidia
    # But we'll try it if mps/cuda
    autocast_enabled = device_type != "cpu"

    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    for _ in range(20):
        with torch.autocast(device_type=device_type, enabled=autocast_enabled):
            _ = model(x_enc, mask)
    torch.mps.synchronize() if device.type == "mps" else None
    print(f"Full Forward Pass (20 runs): {time.time() - start:.4f}s")

    # Component Profiling: STGNN Blocks
    torch.mps.synchronize() if device.type == "mps" else None
    start = time.time()
    
    # Manually run the initial projection
    with torch.no_grad():
        x = x_enc.permute(0, 3, 1, 2).contiguous()
        x = model.feature_norm(x)
        x = model.input_proj(x)
        B, C, N, T = x.shape
        x = x.view(B, C, num_stores, num_families, T)
        
        block = model.layers[0]
        
        start = time.time()
        for _ in range(20):
            with torch.autocast(device_type=device_type, enabled=autocast_enabled):
                _ = block(x)
        torch.mps.synchronize() if device.type == "mps" else None
        print(f"Single STGNNBlock (20 runs): {time.time() - start:.4f}s")

        # Sub-component: GCN
        gcn = block.gcn
        x_t = block.tcn(x)
        start = time.time()
        for _ in range(20):
            with torch.autocast(device_type=device_type, enabled=autocast_enabled):
                _ = gcn(x_t)
        torch.mps.synchronize() if device.type == "mps" else None
        print(f"  - FactoredGCN only (20 runs): {time.time() - start:.4f}s")

        # Sub-component: TCN
        tcn = block.tcn
        start = time.time()
        for _ in range(20):
            with torch.autocast(device_type=device_type, enabled=autocast_enabled):
                _ = tcn(x)
        torch.mps.synchronize() if device.type == "mps" else None
        print(f"  - TemporalBlock only (20 runs): {time.time() - start:.4f}s")


if __name__ == "__main__":
    with torch.no_grad():
        profile()
