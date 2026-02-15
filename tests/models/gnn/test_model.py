import torch
from src.models.gnn.model import SalesGNN


def test_gnn_forward():
    num_stores = 5
    num_families = 3
    num_nodes = num_stores + num_families + (num_stores * num_families)

    hidden_dim = 16
    feature_dim = 8
    window = 30
    horizon = 16
    batch_size = 4

    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        num_layers=2,
    )

    # x_enc: (Batch, Nodes, Window, Features)
    x_enc = torch.randn(batch_size, num_nodes, window, feature_dim)
    # is_open: (Batch, Nodes, Horizon)
    is_open = torch.ones(batch_size, num_nodes, horizon)

    output = model(x_enc, is_open=is_open)

    # Assert output shape: (Batch, Nodes, Horizon)
    assert output.shape == (batch_size, num_nodes, horizon)


def test_hard_gating():
    num_stores = 2
    num_families = 2
    num_nodes = num_stores + num_families + (num_stores * num_families)

    hidden_dim = 8
    feature_dim = 4
    window = 10
    horizon = 5
    batch_size = 2

    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        num_layers=1,
    )

    x_enc = torch.randn(batch_size, num_nodes, window, feature_dim)

    # High-intensity "is_open" mask - all zeros
    is_open = torch.zeros(batch_size, num_nodes, horizon)

    output = model(x_enc, is_open=is_open)

    # Verify that all outputs are 0 due to hard gating
    assert torch.all(output == 0)


def test_learnable_adjacency_shape():
    num_stores = 3
    num_families = 2
    num_nodes = num_stores + num_families + (num_stores * num_families)

    hidden_dim = 8
    feature_dim = 4

    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        horizon=5,
    )

    # Adjacency should be (num_series, num_series) for Factored GNN
    adj = model.get_adjacency()
    num_series = num_stores * num_families
    assert adj.shape == (num_series, num_series)
