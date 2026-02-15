import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CausalConv3d(nn.Module):
    """3D Convolution with causal padding on the time dimension."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=(1, 1, kernel_size),
            padding=0,
            dilation=(1, 1, dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, F, T)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0)) # Pad last dim (Time)
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Gated Temporal Block in 5D."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.filter_conv = CausalConv3d(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv3d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure contiguous layout for Metal kernel efficiency
        x = x.contiguous()
        return torch.tanh(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))


class FactoredGCN(nn.Module):
    """
    Factored Spatial Mixer.
    Complexity: O(S^2 + F^2)
    """

    def __init__(
        self, 
        channels: int, 
        num_stores: int, 
        num_families: int, 
        embedding_dim: int,
        dist_matrix: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.S = num_stores
        self.F = num_families
        
        # Identity-plus-epsilon initialization strategy:
        # Initialize embeddings such that E1 @ E2.T has a strong diagonal bias
        self.e1_s = nn.Parameter(torch.randn(num_stores, embedding_dim) * 0.01)
        self.e2_s = nn.Parameter(torch.randn(num_stores, embedding_dim) * 0.01)
        
        if dist_matrix is not None:
            # Initialize based on spatial distance: Similarity = exp(-dist)
            # We can use the first few dimensions of the embedding to encode this
            similarity = torch.exp(-dist_matrix)
            # Simple low-rank approx bias: E1, E2 = sqrt(Similarity)
            # For simplicity, we just add the similarity as a bias in the initialization
            # or use it to seed the weights.
            with torch.no_grad():
                # U, S, V = torch.svd(similarity)
                # This might be slow for large matrices, but S is small (54)
                U, S, V = torch.svd(similarity)
                S_sqrt = torch.diag(torch.sqrt(S[:, :embedding_dim]))
                self.e1_s.copy_(U[:, :embedding_dim] @ S_sqrt)
                self.e2_s.copy_(V[:, :embedding_dim] @ S_sqrt)
        
        self.e1_f = nn.Parameter(torch.randn(num_families, embedding_dim) * 0.01)
        self.e2_f = nn.Parameter(torch.randn(num_families, embedding_dim) * 0.01)
        
        # Identity bias: bias the dot product to be positive on the diagonal
        # This is handled by adding an epsilon * I during the adjacency calculation
        # to ensure stable gradients even if embeddings are near zero.
        self.eps = 1e-3

        self.proj = nn.Conv3d(channels, channels, kernel_size=1)

    def get_adjs(self) -> List[torch.Tensor]:
        # Spatial Adj
        adj_s = torch.matmul(self.e1_s, self.e2_s.t())
        # Add identity bias (plus epsilon) for stability
        adj_s = adj_s + torch.eye(self.S, device=adj_s.device) * self.eps
        adj_s = F.relu(torch.tanh(adj_s))
        
        # Family Adj
        adj_f = torch.matmul(self.e1_f, self.e2_f.t())
        adj_f = adj_f + torch.eye(self.F, device=adj_f.device) * self.eps
        adj_f = F.relu(torch.tanh(adj_f))
        
        # Row-normalize
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
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = torch.matmul(x, adj_s.t()) 
        
        # 2. Family-wise
        # (B, C, F, T, S) -> (B, C, S, T, F)
        x = x.permute(0, 1, 4, 3, 2).contiguous()
        x = torch.matmul(x, adj_f.t())
        
        # Back to (B, C, S, F, T)
        x = x.permute(0, 1, 2, 4, 3).contiguous()
        return self.proj(x)


class STGNNBlock(nn.Module):
    """Factored STGNN block operating on a 5D Grid."""

    def __init__(self, hidden_dim: int, num_stores: int, num_families: int, 
                 kernel_size: int, dilation: int, embedding_dim: int,
                 dist_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.tcn = TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation)
        self.gcn = FactoredGCN(hidden_dim, num_stores, num_families, embedding_dim, dist_matrix)
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
    Operates ONLY on series nodes (S*F) to eliminate hub-node synchronization.
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
        dist_matrix: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        self.num_stores = num_stores
        self.num_families = num_families
        self.horizon = horizon
        self.use_checkpointing = use_checkpointing

        # Learnable InstanceNorm to handle varied feature scales across nodes
        # input_proj gets (B, F, N_series, T)
        # We wrap it with InstanceNorm1d applied to the feature dimension
        self.feature_norm = nn.InstanceNorm2d(feature_dim, affine=True)
        
        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
        self.layers = nn.ModuleList([
            STGNNBlock(hidden_dim, num_stores, num_families, kernel_size, 2**i, embedding_dim, dist_matrix)
            for i in range(num_layers)
        ])

        # Output head (re-project from 3D grid back to 1D nodes)
        self.fc_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 2, horizon, kernel_size=1),
        )

    def forward(self, x_enc: torch.Tensor, is_open: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_enc: (Batch, N_series, Time, Features)
        is_open: (Batch, N_series, Time)
        """
        # Layout: (B, N, T, F) -> (B, F, N, T)
        x = x_enc.permute(0, 3, 1, 2).contiguous()
        
        # 1. Feature Normalization
        x = self.feature_norm(x)
        
        # 2. Feature Projection
        x = self.input_proj(x)
        
        # 3. Reshape to 5D Grid: (B, C, S, F, T)
        B, C, _, T = x.shape
        x = x.view(B, C, self.num_stores, self.num_families, T)

        skip_total = 0
        for layer in self.layers:
            if self.training and self.use_checkpointing:
                x, skip = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x, skip = layer(x)
            skip_total = skip_total + skip

        # Pool temporal dimensions (take last step for prediction logic)
        final_feat = skip_total[..., -1:] # (B, C, S, F, 1)
        
        # Output: (B, Horizon, S, F, 1)
        out = self.fc_head(final_feat)
        
        # Flatten and Permute back to (Batch, N_series, Horizon)
        out = out.view(B, self.horizon, -1).permute(0, 2, 1).contiguous()

        if is_open is not None:
            # is_open: (Batch, N_series, Horizon) or (Batch, N_series, 1)
            # If temporal pool was used, we might need to take the last step of is_open mask too
            # or just assume the mask passed is already for the horizon.
            if is_open.dim() == 3 and is_open.shape[-1] > self.horizon:
                 # Take last horizon steps
                 is_open = is_open[..., -self.horizon:]
            out = out * is_open

        # Optimization: No longer padding to full 1869 nodes. 
        # The trainer and inference loop should handle the S*F shape directly.
        return out

    def get_adjacency(self) -> List[torch.Tensor]:
        """Returns the list of factorized adjs for regularization."""
        return self.layers[0].gcn.get_adjs()
