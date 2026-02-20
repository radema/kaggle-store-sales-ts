import pytest
import torch
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from src.models.gnn_v2.loader import TemporalGraphDataset, make_gnn_v2_loader

def test_temporal_graph_dataset_init():
    # Mock data
    num_nodes = 10
    num_time_steps = 5
    window_size = 7
    num_features = 3
    horizon = 16
    
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    static_x = torch.randn((num_nodes, 5))
    static_graph = Data(x=static_x, edge_index=edge_index)
    
    temporal_x = torch.randn((num_time_steps, num_nodes, window_size, num_features))
    temporal_y = torch.randn((num_time_steps, num_nodes, horizon))
    
    dataset = TemporalGraphDataset(static_graph, temporal_x, temporal_y)
    
    assert dataset.static_graph.num_nodes == num_nodes
    assert dataset.temporal_x.shape == (num_time_steps, num_nodes, window_size, num_features)
    assert dataset.temporal_y.shape == (num_time_steps, num_nodes, horizon)

def test_make_gnn_v2_loader():
    # Mock data
    num_nodes = 50
    num_time_steps = 20
    window_size = 14
    num_features = 10
    horizon = 16
    
    # Fully connected graph for simplicity in sampling
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j:
                edges.append([i, j])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    static_graph = Data(x=torch.randn((num_nodes, 5)), edge_index=edge_index)
    
    temporal_x = torch.randn((num_time_steps, num_nodes, window_size, num_features))
    temporal_y = torch.randn((num_time_steps, num_nodes, horizon))
    
    dataset = TemporalGraphDataset(static_graph, temporal_x, temporal_y)
    
    # Input nodes: (time_idx, node_idx)
    # We want to sample a batch of target nodes
    time_indices = [0, 0, 1, 1]
    node_indices = [5, 10, 15, 20]
    
    batch_size = 2
    loader = make_gnn_v2_loader(
        dataset, 
        time_indices=time_indices, 
        node_indices=node_indices, 
        batch_size=batch_size,
        num_neighbors=[10, 10], 
        shuffle=False
    )
    
    # PULL ONE BATCH
    batch = next(iter(loader))
    
    # Assertions
    assert hasattr(batch, 'n_id'), "Batch must have n_id (original indices)"
    assert batch.batch_size == batch_size, f"Expected {batch_size} target nodes, got {batch.batch_size}"
    
    # Check temporal features retrieval
    # batch.x should contain the temporal features sliced by n_id
    # Shape: (Total_Sampled_Nodes, Window, Features)
    total_nodes = batch.n_id.size(0)
    assert batch.x.shape == (total_nodes, window_size, num_features)
    
    # Check target retrieval
    # batch.y should contain targets for the TARGET nodes only (first batch_size nodes)
    assert batch.y.shape == (batch_size, horizon)
    
    # Verify values match expectations for target nodes
    # The first batch_size elements in n_id are the input target nodes
    target_node_indices_in_batch = batch.n_id[:batch_size]
    # In our case, time_idx matches for the first batch
    t_idx = time_indices[0] 
    
    expected_y = temporal_y[t_idx, target_node_indices_in_batch]
    assert torch.allclose(batch.y, expected_y)
    
    expected_x0 = temporal_x[t_idx, batch.n_id[0]]
    assert torch.allclose(batch.x[0], expected_x0)
