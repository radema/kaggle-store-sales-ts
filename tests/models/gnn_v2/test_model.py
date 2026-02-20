import pytest
import torch
from torch_geometric.data import Data
from src.models.gnn_v2.loader import TemporalGraphDataset, TemporalNeighborLoader
from src.models.gnn_v2.model import SalesGNNv2

@pytest.fixture
def dummy_data():
    num_nodes = 100
    num_time_steps = 10
    window_size = 30
    num_features = 5
    horizon = 16
    
    # 1. Static Graph: Connect nodes in a ring
    edges = []
    for i in range(num_nodes):
        edges.append([i, (i + 1) % num_nodes])
        edges.append([(i + 1) % num_nodes, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t()
    static_graph = Data(edge_index=edge_index, num_nodes=num_nodes)
    
    # 2. Temporal Data
    temporal_x = torch.randn(num_time_steps, num_nodes, window_size, num_features)
    temporal_y = torch.randn(num_time_steps, num_nodes, horizon)
    
    dataset = TemporalGraphDataset(static_graph, temporal_x, temporal_y)
    return dataset, num_features, horizon

def test_gnn_v2_forward_pass(dummy_data):
    dataset, num_features, horizon = dummy_data
    batch_size = 8
    hidden_channels = 32
    
    # Loader
    time_indices = [5] * 10
    node_indices = list(range(10))
    loader = TemporalNeighborLoader(
        dataset, time_indices, node_indices, batch_size=batch_size, num_neighbors=[5, 5]
    )
    
    # Model
    model = SalesGNNv2(
        in_channels=num_features,
        hidden_channels=hidden_channels,
        out_channels=horizon,
        num_layers=1,
        gat_heads=2
    )
    
    # Sample batch and run forward
    batch = next(iter(loader))
    output = model(batch)
    
    # Assertions
    assert output.shape == (batch_size, horizon)
    assert not torch.isnan(output).any(), "Model output contains NaNs"
    
def test_gnn_v2_gradient_flow(dummy_data):
    dataset, num_features, horizon = dummy_data
    batch_size = 4
    
    loader = TemporalNeighborLoader(
        dataset, [1, 2, 3, 4], [0, 1, 2, 3], batch_size=batch_size, num_neighbors=[2, 2]
    )
    
    model = SalesGNNv2(
        in_channels=num_features,
        hidden_channels=16,
        out_channels=horizon,
        dropout=0.1
    )
    
    batch = next(iter(loader))
    output = model(batch)
    
    loss = output.pow(2).mean()
    loss.backward()
    
    # Check if gradients exist in key layers
    assert model.encoder.weight.grad is not None
    assert model.temporal_block.conv.weight.grad is not None
    # Check if any parameter in the first spatial layer has a gradient
    spatial_params = list(model.spatial_layers[0].parameters())
    assert len(spatial_params) > 0
    assert any(p.grad is not None for p in spatial_params)
    assert model.predictor[0].weight.grad is not None

@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_gnn_v2_mps_compatibility(dummy_data):
    dataset, num_features, horizon = dummy_data
    device = torch.device("mps")
    
    batch_size = 4
    loader = TemporalNeighborLoader(
        dataset, [1], [0, 1, 2, 3], batch_size=batch_size, num_neighbors=[2]
    )
    
    model = SalesGNNv2(
        in_channels=num_features,
        hidden_channels=16,
        out_channels=horizon
    ).to(device)
    
    batch = next(iter(loader)).to(device)
    output = model(batch)
    
    assert output.device.type == "mps"
    assert output.shape == (batch_size, horizon)
