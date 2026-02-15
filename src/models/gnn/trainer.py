import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional
from src.utils.logging import get_logger


class CompositeLoss(nn.Module):
    """
    Combined Loss: MSE + L1 Regularization on Adjacency matrix.
    MSE is calculated on log-scaled targets (consistent with RMSLE).
    """

    def __init__(self, alpha: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(
        self, y_pred: torch.Tensor, y_true: torch.Tensor, adjs: List[torch.Tensor]
    ) -> torch.Tensor:
        # mse_loss: (Batch, Nodes, Horizon)
        mse_loss = self.mse(y_pred, y_true)

        # l1_loss: sparsity constraint on all factorized graphs
        l1_sum = 0
        for adj in adjs:
            l1_sum += torch.norm(adj, p=1) / adj.numel()
        
        return mse_loss + self.alpha * l1_sum


class GNNTrainer:
    """
    Trainer for SalesGNN model.
    Handles device placement, training loops, and predictions with robust logging.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        alpha: float = 0.01,
        device: Optional[torch.device] = None,
        patience: int = 5,
        logger=None,
        static_node_features: Optional[torch.Tensor] = None,
    ):
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = CompositeLoss(alpha=alpha)
        self.patience = patience
        self.logger = logger or get_logger("gnn_trainer")
        self.best_model_state = None
        self.best_val_loss = float("inf")
        self.history = {"train_loss": [], "val_loss": []}
        
        # Move static features to device once
        if static_node_features is not None:
            self.static_node_features = static_node_features.to(self.device).contiguous()
        else:
            self.static_node_features = None

    def train_step(
        self, x_enc: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> float:
        """Single optimization step."""
        self.model.train()
        self.optimizer.zero_grad()

        x_enc = x_enc.to(self.device).contiguous()
        y = y.to(self.device)
        mask = mask.to(self.device)

        # Ensure targets and masks only contain series nodes to match model output
        # (This handles datasets that still include hub nodes)
        N_series = self.model.num_stores * self.model.num_families
        if y.shape[1] > N_series:
            offset = y.shape[1] - N_series
            y = y[:, offset:, :].contiguous()
            mask = mask[:, offset:, :].contiguous()

        # Autocast strategy: CUDA is stable with FP16, MPS is often unstable
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        use_autocast = device_type == "cuda"

        with torch.autocast(device_type=device_type, enabled=use_autocast):
            y_pred = self.model(x_enc, is_open=mask, static_feats=self.static_node_features)
            loss = self.loss_fn(y_pred, y, self.model.get_adjacency())

        if torch.isnan(loss):
            self.logger.error("Loss is NaN during train step!")
            return float("nan")

        loss.backward()

        # Gradient clipping to prevent NaN on MPS/GPU
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

        return loss.item()

    def val_step(
        self, x_enc: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> float:
        """Single validation step."""
        self.model.eval()
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        use_autocast = device_type == "cuda"

        with torch.no_grad():
            x_enc = x_enc.to(self.device).contiguous()
            y = y.to(self.device)
            mask = mask.to(self.device)

            # Ensure targets and masks only contain series nodes
            N_series = self.model.num_stores * self.model.num_families
            if y.shape[1] > N_series:
                offset = y.shape[1] - N_series
                y = y[:, offset:, :].contiguous()
                mask = mask[:, offset:, :].contiguous()

            with torch.autocast(device_type=device_type, enabled=use_autocast):
                y_pred = self.model(x_enc, is_open=mask, static_feats=self.static_node_features)
                loss = self.loss_fn(y_pred, y, self.model.get_adjacency())

        return loss.item()

    def fit(
        self, train_loader, val_loader=None, epochs: int = 20
    ) -> Dict[str, List[float]]:
        """Standard training loop with Early Stopping."""
        self.logger.info(f"Starting training for {epochs} epochs on {self.device}")

        early_stop_counter = 0

        for epoch in range(epochs):
            train_loss = 0
            for batch in train_loader:
                x_enc, y, mask = batch
                loss = self.train_step(x_enc, y, mask)
                if np.isnan(loss):
                    self.logger.error(
                        f"NaN loss detected at epoch {epoch + 1}. Halting."
                    )
                    return self.history
                train_loss += loss

            avg_train_loss = train_loss / len(train_loader)
            self.history["train_loss"].append(avg_train_loss)

            val_info = ""
            if val_loader:
                val_loss = 0
                for batch in val_loader:
                    x_enc, y, mask = batch
                    val_loss += self.val_step(x_enc, y, mask)
                avg_val_loss = val_loss / len(val_loader)
                self.history["val_loss"].append(avg_val_loss)
                val_info = f" | Val Loss: {avg_val_loss:.6f}"

                # Early Stopping Logic
                if avg_val_loss < self.best_val_loss:
                    self.best_val_loss = avg_val_loss
                    self.best_model_state = {
                        k: v.cpu() for k, v in self.model.state_dict().items()
                    }
                    early_stop_counter = 0
                else:
                    early_stop_counter += 1

            self.logger.info(
                f"Epoch {epoch + 1:03d}/{epochs} | Train Loss: {avg_train_loss:.6f}{val_info}"
            )

            if early_stop_counter >= self.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        if self.best_model_state:
            self.logger.info("Restoring best model state...")
            self.model.load_state_dict(self.best_model_state)
            self.model.to(self.device)

        return self.history

    def predict(self, loader) -> torch.Tensor:
        """Generates predictions for the entire loader."""
        self.model.eval()
        all_preds = []
        device_type = "cuda" if "cuda" in str(self.device) else "cpu"
        use_autocast = device_type == "cuda"

        with torch.no_grad():
            for batch in loader:
                x_enc, _, mask = batch
                x_enc = x_enc.to(self.device).contiguous()
                mask = mask.to(self.device)

                with torch.autocast(device_type=device_type, enabled=use_autocast):
                    y_pred = self.model(x_enc, is_open=mask, static_feats=self.static_node_features)
                all_preds.append(y_pred.cpu())

        return torch.cat(all_preds, dim=0)
