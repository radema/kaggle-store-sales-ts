import torch
import torch.nn as nn


class CompositeLoss(nn.Module):
    """
    Combined Loss: MSE + L1 Regularization on Adjacency matrix.
    MSE is calculated on log-scaled targets (consistent with RMSLE).
    """

    def __init__(self, alpha=0.01):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, y_pred, y_true, adj):
        # mse_loss: (Batch, Nodes, Horizon)
        mse_loss = self.mse(y_pred, y_true)

        # l1_loss: sparsity constraint on learned graph
        l1_loss = torch.norm(adj, p=1) / (adj.shape[0] * adj.shape[1])

        return mse_loss + self.alpha * l1_loss


class GNNTrainer:
    """
    Trainer for SalesGNN model.
    Handles device placement, training loops, and predictions.
    """

    def __init__(self, model, optimizer, alpha=0.01, device=None):
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = CompositeLoss(alpha=alpha)

    def train_step(self, x_enc, x_dec, y, mask):
        """Single optimization step."""
        self.model.train()
        self.optimizer.zero_grad()

        x_enc = x_enc.to(self.device)
        x_dec = x_dec.to(self.device)
        y = y.to(self.device)
        mask = mask.to(self.device)

        y_pred = self.model(x_enc, x_dec, mask)

        loss = self.loss_fn(y_pred, y, self.model.get_adjacency())

        loss.backward()
        self.optimizer.step()

        return loss.item()

    def val_step(self, x_enc, x_dec, y, mask):
        """Single validation step."""
        self.model.eval()
        with torch.no_grad():
            x_enc = x_enc.to(self.device)
            x_dec = x_dec.to(self.device)
            y = y.to(self.device)
            mask = mask.to(self.device)

            y_pred = self.model(x_enc, x_dec, mask)
            loss = self.loss_fn(y_pred, y, self.model.get_adjacency())

        return loss.item()

    def fit(self, train_loader, val_loader=None, epochs=20):
        """Standard training loop."""
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(epochs):
            train_loss = 0
            # Use tqdm if interactive, otherwise standard loop
            for batch in train_loader:
                x_enc, x_dec, y, mask = batch
                train_loss += self.train_step(x_enc, x_dec, y, mask)

            avg_train_loss = train_loss / len(train_loader)
            history["train_loss"].append(avg_train_loss)

            val_info = ""
            if val_loader:
                val_loss = 0
                for batch in val_loader:
                    x_enc, x_dec, y, mask = batch
                    val_loss += self.val_step(x_enc, x_dec, y, mask)
                avg_val_loss = val_loss / len(val_loader)
                history["val_loss"].append(avg_val_loss)
                val_info = f" | Val Loss: {avg_val_loss:.6f}"

            print(
                f"Epoch {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.6f}{val_info}"
            )

        return history

    def predict(self, loader):
        """Generates predictions for the entire loader."""
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                x_enc, x_dec, _, mask = batch
                x_enc = x_enc.to(self.device)
                x_dec = x_dec.to(self.device)
                mask = mask.to(self.device)

                y_pred = self.model(x_enc, x_dec, mask)
                all_preds.append(y_pred.cpu())

        return torch.cat(all_preds, dim=0)
