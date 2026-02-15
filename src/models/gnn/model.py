import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CausalConv3d(nn.Module):
    """3D Convolution with causal padding on the time dimension."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        # (B, C, S, F, T) -> kernel treats S, F as spatial, T as time
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, kernel_size),
            padding=0,
            dilation=(1, 1, dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, F, T)
        x = F.pad(x, (self.padding, 0)) # Pad last dim (Time)
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Gated Temporal Block in 5D."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.filter_conv = CausalConv3d(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv3d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))


class FactoredGCN(nn.Module):
    """
    Factored Spatial Mixer.
    Complexity: O(S^2 + F^2)
    """

    def __init__(self, channels: int, num_stores: int, num_families: int, embedding_dim: int):
        super().__init__()
        self.S = num_stores
        self.F = num_families
        
        self.e1_s = nn.Parameter(torch.randn(num_stores, embedding_dim))
        self.e2_s = nn.Parameter(torch.randn(num_stores, embedding_dim))
        
        self.e1_f = nn.Parameter(torch.randn(num_families, embedding_dim))
        self.e2_f = nn.Parameter(torch.randn(num_families, embedding_dim))

        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def get_adjs(self) -> List[torch.Tensor]:
        adj_s = F.relu(torch.tanh(torch.matmul(self.e1_s, self.e2_s.t())))
        adj_f = F.relu(torch.tanh(torch.matmul(self.e1_f, self.e2_f.t())))
        
        # Row-normalize to keep magnitudes stable across layers
        # Adding epsilon to avoid div by zero
        adj_s = adj_s / (adj_s.sum(dim=-1, keepdim=True) + 1e-6)
        adj_f = adj_f / (adj_f.sum(dim=-1, keepdim=True) + 1e-6)
        
        return [adj_s, adj_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, S, F, T)"""
        B, C, S, F, T = x.shape
        adjs = self.get_adjs()
        adj_s, adj_f = adjs[0], adjs[1]

        # 1. Store-wise
        # (B, C, S, F, T) -> (B, C, F, T, S)
        x = x.permute(0, 1, 3, 4, 2)
        x = torch.matmul(x, adj_s.t()) 
        
        # 2. Family-wise
        # (B, C, F, T, S) -> (B, C, S, T, F)
        x = x.permute(0, 1, 4, 3, 2)
        x = torch.matmul(x, adj_f.t())
        
        # Back to (B, C, S, F, T)
        x = x.permute(0, 1, 2, 4, 3)
        return self.proj(x)


class STGNNBlock(nn.Module):
    """Factored STGNN block operating on a 5D Grid."""

    def __init__(self, hidden_dim: int, num_stores: int, num_families: int, 
                 kernel_size: int, dilation: int, embedding_dim: int):
        super().__init__()
        self.tcn = TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation)
        self.gcn = FactoredGCN(hidden_dim, num_stores, num_families, embedding_dim)
        self.norm = nn.BatchNorm3d(hidden_dim)
        self.skip_proj = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, S, F, T)"""
        # 1. Temporal
        x_t = self.tcn(x)
        
        # 2. Spatial
        x_spatial = self.gcn(x_t)
        
        # 3. Residual & Norm (5D)
        out = self.norm(x_spatial + x)
        
        # Skip connection
        skip = self.skip_proj(x_spatial)
        
        return out, skip


class SalesGNN(nn.Module):
    """
    Optimized Factored STGNN.
    Ignores non-series nodes to save memory.
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
        self.horizon = horizon
        self.use_checkpointing = use_checkpointing

        # Internal feature projection (still 2D as we enter from the node layout)
        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
        self.layers = nn.ModuleList([
            STGNNBlock(hidden_dim, num_stores, num_families, kernel_size, 2**i, embedding_dim)
            for i in range(num_layers)
        ])

        # Output head (re-project from 3D grid back to 1D nodes)
        self.fc_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 2, horizon, kernel_size=1),
        )

    def forward(self, x_enc: torch.Tensor, is_open: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_enc: (Batch, Nodes, Time, Features)"""
        # 1. Extract ONLY Series Nodes (Ignore Store/Family hubs)
        x = x_enc[:, self.series_offset:, :, :].permute(0, 3, 1, 2).contiguous()
        x = self.input_proj(x)
        
        # Reshape to 5D Grid: (B, C, S, F, T)
        B, C, _, T = x.shape
        x = x.view(B, C, self.num_stores, self.num_families, T)

        skip_total = 0
        for layer in self.layers:
            if self.training and self.use_checkpointing:
                x, skip = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x, skip = layer(x)
            skip_total = skip_total + skip

        # Pool temporal dimensions (take last step)
        # skip_total is 5D: (B, C, S, F, T)
        final_feat = skip_total[..., -1:] # (B, C, S, F, 1)
        
        # Output: (B, Horizon, S, F, 1)
        out = self.fc_head(final_feat)
        
        # Flatten back to (Batch, Horizon, S*F)
        out = out.view(B, self.horizon, -1)
        
        # Format back to (Batch, Nodes, Horizon)
        out = out.permute(0, 2, 1).contiguous()

        if is_open is not None:
            # Slicing the mask to match only series nodes
            mask = is_open[:, self.series_offset:, :]
            out = out * mask

        # To maintain compatibility with the Dataset, we pad the summary nodes with zeros
        # Result: (B, 1869, Horizon)
        full_out = torch.zeros(B, self.num_nodes, self.horizon, device=out.device)
        full_out[:, self.series_offset:, :] = out
        return full_out

    @property
    def num_nodes(self):
        return self.series_offset + (self.num_stores * self.num_families)

    def get_adjacency(self) -> List[torch.Tensor]:
        """Returns the list of factorized adjs for regularization."""
        return self.layers[0].gcn.get_adjs()
