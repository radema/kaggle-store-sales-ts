import torch
from src.models.gnn.trainer import GNNTrainer, CompositeLoss
from src.models.gnn.model import SalesGNN


def test_composite_loss():
    # log_y is typically used in the competition for RMSLE
    y_pred = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y_true = torch.tensor([[1.1, 1.9], [3.2, 3.8]])

    # Large adj matrix to see L1 impact
    adj = torch.ones((10, 10))
    alpha = 0.1

    loss_fn = CompositeLoss(alpha=alpha)
    loss = loss_fn(y_pred, y_true, adj)

    # MSE part: mean((y_pred-y_true)^2) = 0.025
    # L1 part: alpha * (norm(adj, p=1) / (N*N)) = 0.1 * (100 / 100) = 0.1
    # Total: 0.125
    assert torch.isclose(loss, torch.tensor(0.125))


def test_trainer_fit_step():
    # Simple smoke test for trainer.train_step
    num_stores = 2
    num_families = 2
    num_nodes = num_stores + num_families + (num_stores * num_families)

    hidden_dim = 8
    feature_dim = 4
    horizon = 3

    model = SalesGNN(
        num_stores=num_stores,
        num_families=num_families,
        feature_dim=feature_dim,
        hidden_dim=hidden_dim,
        horizon=horizon,
        num_layers=1,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    trainer = GNNTrainer(model, optimizer, alpha=0.001)

    # Dummy batch: (Batch, Nodes, Time, Features)
    x_enc = torch.randn(2, num_nodes, 10, feature_dim)
    y = torch.randn(2, num_nodes, horizon)
    mask = torch.ones(2, num_nodes, horizon)

    loss = trainer.train_step(x_enc, y, mask)
    assert loss > 0
    import math

    assert not math.isnan(loss)
