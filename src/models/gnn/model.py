import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class CausalConv2d(nn.Module):
    """2D Convolution with causal padding on the time dimension."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(1, kernel_size),
            padding=0,
            dilation=(1, dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Gated Temporal Block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.filter_conv = CausalConv2d(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv2d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))


class FactoredGCN(nn.Module):
    """
    Factored Spatial Mixer.
    Processes series data by decomposing interactions into Store-Store and Family-Family.
    Complexity: O(S^2 + F^2) instead of O((S*F)^2)
    """

    def __init__(self, channels: int, num_stores: int, num_families: int, embedding_dim: int):
        super().__init__()
        self.S = num_stores
        self.F = num_families
        
        # Small learnable embeddings for the two graphs
        self.e1_s = nn.Parameter(torch.randn(num_stores, embedding_dim))
        self.e2_s = nn.Parameter(torch.randn(num_stores, embedding_dim))
        
        self.e1_f = nn.Parameter(torch.randn(num_families, embedding_dim))
        self.e2_f = nn.Parameter(torch.randn(num_families, embedding_dim))

        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def get_adj_s(self) -> torch.Tensor:
        return F.relu(torch.tanh(torch.matmul(self.e1_s, self.e2_s.t())))

    def get_adj_f(self) -> torch.Tensor:
        return F.relu(torch.tanh(torch.matmul(self.e1_f, self.e2_f.t())))

    def forward(self, x_series: torch.Tensor) -> torch.Tensor:
        """
        x_series: (B, C, S*F, T) 
        We treat the S*F nodes as a 2D grid of Store x Family
        """
        B, C, _, T = x_series.shape
        
        # 1. Reshape to Grid: (B, C, S, F, T)
        x = x_series.view(B, C, self.S, self.F, T)
        
        adj_s = self.get_adj_s() # (S, S)
        adj_f = self.get_adj_f() # (F, F)

        # 2. Store-wise Mixing (Inter-store correlations for each family)
        # We want to multiply adj_s @ x on the S dimension (dim 2)
        # torch.matmul handles multiple batch dims (B, C) and trailing dims (F, T)
        # We need to move S to the second-to-last pos to use matmul easily
        # Current: (B, C, S, F, T) -> (B, C, F, T, S)
        x = x.permute(0, 1, 3, 4, 2)
        x = torch.matmul(x, adj_s.t()) 
        
        # 3. Family-wise Mixing (Inter-family correlations for each store)
        # Current: (B, C, F, T, S) -> (B, C, S, T, F)
        x = x.permute(0, 1, 4, 3, 2)
        x = torch.matmul(x, adj_f.t())
        
        # Back to (B, C, S, F, T) -> (B, C, S*F, T)
        x = x.permute(0, 1, 2, 4, 3).reshape(B, C, -1, T)
        
        return self.proj(x)


class STGNNBlock(nn.Module):
    """Factored Spatio-Temporal block."""

    def __init__(self, hidden_dim: int, num_stores: int, num_families: int, 
                 kernel_size: int, dilation: int, embedding_dim: int):
        super().__init__()
        self.tcn = TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation)
        self.gcn = FactoredGCN(hidden_dim, num_stores, num_families, embedding_dim)
        
        self.norm = nn.BatchNorm2d(hidden_dim)
        self.skip_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x_all: torch.Tensor, series_offset: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x_all: (B, C, N, T) where N = S + F + S*F
        """
        # 1. Temporal Update for everyone
        x_t = self.tcn(x_all)
        
        # 2. Spatial Update ONLY for series nodes (The 1782 leaf nodes)
        x_series = x_t[:, :, series_offset:, :]
        x_spatial = self.gcn(x_series)
        
        # Update original leaf nodes in the tensor
        x_new = x_t.clone()
        x_new[:, :, series_offset:, :] = x_spatial
        
        # 3. Residual & Norm
        out = self.norm(x_new + x_all)
        
        return out, self.skip_proj(x_new)


class SalesGNN(nn.Module):
    """
    Factored SalesGNN: High-speed hierarchical forecasting.
    Separates Store-level and Family-level spatial mixing.
    """

    def __init__(
        self,
        num_stores: int,
        num_families: int,
        feature_dim: int,
        hidden_dim: int,
        horizon: int,
        embedding_dim: int = 8,
        kernel_size: int = 2,
        num_layers: int = 4,
        use_checkpointing: bool = True,
    ):
        super().__init__()
        self.num_stores = num_stores
        self.num_families = num_families
        self.series_offset = num_stores + num_families
        self.num_nodes = self.series_offset + (num_stores * num_families)
        self.horizon = horizon
        self.use_checkpointing = use_checkpointing

        # Project input features to channels
        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                STGNNBlock(hidden_dim, num_stores, num_families, kernel_size, 2**i, embedding_dim)
            )

        self.fc_head = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim * 2, horizon, kernel_size=1),
        )

    def forward(self, x_enc: torch.Tensor, is_open: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_enc: (Batch, Nodes, Time, Features)"""
        # (B, N, T, F) -> (B, F, N, T)
        x = x_enc.permute(0, 3, 1, 2).contiguous()
        x = self.input_proj(x)

        skip_total = 0
        for layer in self.layers:
            if self.training and self.use_checkpointing:
                # Wrap the layer call in checkpoint
                def create_custom_forward(module):
                    def custom_forward(*args):
                        return module(*args)

                    return custom_forward

                x, skip = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    x,
                    self.series_offset,
                    use_reentrant=False,
                )
            else:
                x, skip = layer(x, self.series_offset)
            skip_total = skip_total + skip

        # Pool temporal dimensions (take last step)
        final_feat = skip_total[..., -1:] # (B, C, N, 1)
        
        # Output: (B, Horizon, N, 1)
        out = self.fc_head(final_feat)
        out = out.squeeze(-1).permute(0, 2, 1).contiguous()

        if is_open is not None:
            out = out * is_open

        return out

    def get_adjacency(self) -> torch.Tensor:
        """Returns the total Kronecker-style adjacency for visualization."""
        # This is high-memory, only used for debugging/visualization
        adj_s = self.layers[0].gcn.get_adj_s()
        adj_f = self.layers[0].gcn.get_adj_f()
        
        # Identity matrices
        I_s = torch.eye(self.num_stores, device=adj_s.device)
        I_f = torch.eye(self.num_families, device=adj_f.device)
        
        # Kronecker sum approximation A = (As @ If) + (Is @ Af)
        # This represents the internal spatial mixing logic
        term1 = torch.kron(adj_s, I_f)
        term2 = torch.kron(I_s, adj_f)
        return term1 + term2
