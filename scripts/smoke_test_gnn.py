import torch
from src.models.gnn.model import SalesGNN


def smoke_test_model():
    print("Starting GNN Model Smoke Test...")

    # Mock parameters
    num_stores = 54
    num_families = 33
    feature_dim = 15
    hidden_dim = 32
    horizon = 16
    batch_size = 4
    seq_len = 60

    num_nodes = num_stores + num_families + (num_stores * num_families)

    # Instantiate model
    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        num_layers=3,
    )

    # Mock input: (Batch, Nodes, Time, Features)
    x_enc = torch.randn(batch_size, num_nodes, seq_len, feature_dim)
    mask = torch.ones(batch_size, num_nodes, horizon)

    print(f"Input shape: {x_enc.shape}")

    # Forward pass
    output = model(x_enc, is_open=mask)

    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, num_nodes, horizon), (
        f"Wrong output shape: {output.shape}"
    )

    # Check adjacency
    adjs = model.get_adjacency()
    print(f"Factored Adjacencies: {[a.shape for a in adjs]}")
    assert len(adjs) == 2
    assert adjs[0].shape == (num_stores, num_stores)
    assert adjs[1].shape == (num_families, num_families)

    print("Smoke Test Passed!")


if __name__ == "__main__":
    smoke_test_model()
