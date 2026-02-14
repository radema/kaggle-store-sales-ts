import torch
from src.models.gnn.model import SalesGNN


def test_gnn_forward():
    num_nodes = 10
    hidden_dim = 16
    feature_dim = 8
    window = 30
    horizon = 16
    batch_size = 4

    # Mock Adjacency
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long)

    model = SalesGNN(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        edge_index=edge_index,
        horizon=horizon,
    )

    # x_enc: (Batch, Nodes, Window, Features)
    x_enc = torch.randn(batch_size, num_nodes, window, feature_dim)
    # x_dec: (Batch, Nodes, Horizon, Features)
    x_dec = torch.randn(batch_size, num_nodes, horizon, feature_dim)
    # is_open: (Batch, Nodes, Horizon)
    is_open = torch.ones(batch_size, num_nodes, horizon)

    output = model(x_enc, x_dec, is_open)

    # Assert output shape: (Batch, Nodes, Horizon)
    assert output.shape == (batch_size, num_nodes, horizon)


def test_hard_gating():
    num_nodes = 5
    hidden_dim = 8
    feature_dim = 4
    window = 10
    horizon = 5
    batch_size = 2

    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    model = SalesGNN(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        edge_index=edge_index,
        horizon=horizon,
    )

    x_enc = torch.randn(batch_size, num_nodes, window, feature_dim)
    x_dec = torch.randn(batch_size, num_nodes, horizon, feature_dim)

    # High-intensity "is_open" mask - all zeros
    is_open = torch.zeros(batch_size, num_nodes, horizon)

    output = model(x_enc, x_dec, is_open)

    # Verify that all outputs are 0 due to hard gating
    assert torch.all(output == 0)


def test_learnable_adjacency_shape():
    num_nodes = 5
    hidden_dim = 8
    feature_dim = 4
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

    model = SalesGNN(
        num_nodes=num_nodes,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        edge_index=edge_index,
        horizon=5,
    )

    # Adjacency should be (num_nodes, num_nodes)
    adj = model.get_adjacency()
    assert adj.shape == (num_nodes, num_nodes)
