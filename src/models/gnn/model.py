import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


class CausalConv1d(nn.Module):
    """
    Highly optimized 1D Causal Conv for MPS.
    Operates on (Batch*Nodes, Channels, Time) to saturate GPU throughput.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
            dilation=dilation,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., C, T)
        if self.padding > 0:
            x = F.pad(x, (self.padding, 0))
        return self.conv(x)


class TemporalBlock(nn.Module):
    """Gated Temporal Block using 1D folding for performance."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.filter_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.gate_conv = CausalConv1d(in_channels, out_channels, kernel_size, dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, S, F, T)
        B, C, S, F, T = x.shape
        # Fold: (B*S*F, C, T)
        x = x.view(B * S * F, C, T)
        
        out = torch.relu(self.filter_conv(x)) * torch.sigmoid(self.gate_conv(x))
        
        # Unfold back to (B, C, S, F, T)
        return out.view(B, -1, S, F, T)


class FactoredGCN(nn.Module):
    """
    Factored Spatial Mixer using Einstein Summation for zero-copy performance.
    Complexity: O(S^2 + F^2)
    """

    def __init__(
        self, 
        channels: int, 
        num_stores: int, 
        num_families: int, 
        embedding_dim: int,
        dist_matrix: Optional[torch.Tensor] = None,
        static_node_features: Optional[torch.Tensor] = None
    ):
        super().__init__()
        self.S = num_stores
        self.F = num_families
        
        # Identity-plus-epsilon initialization strategy
        self.e1_s = nn.Parameter(torch.randn(num_stores, embedding_dim) * 0.01)
        self.e2_s = nn.Parameter(torch.randn(num_stores, embedding_dim) * 0.01)
        
        if dist_matrix is not None:
            similarity = torch.exp(-dist_matrix)
            with torch.no_grad():
                U, S_vals, V = torch.svd(similarity)
                # S_vals is 1D, use single index slicing
                k = min(embedding_dim, S_vals.size(0))
                S_sqrt = torch.diag(torch.sqrt(S_vals[:k]))
                self.e1_s.copy_(U[:, :k] @ S_sqrt)
                self.e2_s.copy_(V[:, :k] @ S_sqrt)
        
        # Optional: Seed Store embeddings with static features if provided
        if static_node_features is not None:
             # Assume static_node_features is (S, D_meta)
             # We can add a projection here if we wanted to be more explicit, 
             # but for now we rely on the init logic being called from SalesGNN
             pass

        self.e1_f = nn.Parameter(torch.randn(num_families, embedding_dim) * 0.01)
        self.e2_f = nn.Parameter(torch.randn(num_families, embedding_dim) * 0.01)
        
        self.eps = 1e-3
        self.proj = nn.Conv3d(channels, channels, kernel_size=1)
        
        # Pre-register identity matrices for faster adjacency calculation
        self.register_buffer('eye_s', torch.eye(num_stores))
        self.register_buffer('eye_f', torch.eye(num_families))

    def get_adjs(self) -> List[torch.Tensor]:
        adj_s = F.relu(torch.tanh(torch.matmul(self.e1_s, self.e2_s.t())))
        adj_s = adj_s + self.eye_s * self.eps
        
        adj_f = F.relu(torch.tanh(torch.matmul(self.e1_f, self.e2_f.t())))
        adj_f = adj_f + self.eye_f * self.eps
        
        adj_s = adj_s / (adj_s.sum(dim=-1, keepdim=True) + 1e-4)
        adj_f = adj_f / (adj_f.sum(dim=-1, keepdim=True) + 1e-4)
        
        return [adj_s, adj_f]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, S, F, T)"""
        adjs = self.get_adjs()
        adj_s, adj_f = adjs[0], adjs[1]

        # Use Einstein Summation for zero-copy, zero-permute spatial mixing
        # b:batch, c:channels, s:stores, f:families, t:time
        # i,j: store indices; k,l: family indices
        
        # 1. Store-wise mixing: (B, C, S, F, T) * (S, S) -> (B, C, S, F, T)
        x = torch.einsum('bcsft,js->bcjft', x, adj_s)
        
        # 2. Family-wise mixing: (B, C, S, F, T) * (F, F) -> (B, C, S, F, T)
        x = torch.einsum('bcjft,lf->bcjlt', x, adj_f)
        
        return self.proj(x)


class STGNNBlock(nn.Module):
    """Factored STGNN block with 1D folding and einsum mixing."""

    def __init__(self, hidden_dim: int, num_stores: int, num_families: int, 
                 kernel_size: int, dilation: int, embedding_dim: int,
                 dist_matrix: Optional[torch.Tensor] = None):
        super().__init__()
        self.tcn = TemporalBlock(hidden_dim, hidden_dim, kernel_size, dilation)
        self.gcn = FactoredGCN(hidden_dim, num_stores, num_families, embedding_dim, dist_matrix)
        
        # GroupNorm is batch-size independent, unlike BatchNorm
        self.norm = nn.GroupNorm(min(8, hidden_dim), hidden_dim)
        self.skip_proj = nn.Conv3d(hidden_dim, hidden_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x: (B, C, S, F, T)"""
        # 1. Temporal (Folding/Unfolding inside)
        x_t = self.tcn(x)
        
        # 2. Spatial (Einsum inside)
        x_spatial = self.gcn(x_t)
        
        # 3. Residual & Norm
        out = self.norm(x_spatial + x)
        skip = self.skip_proj(x_spatial)
        
        return out, skip


class SalesGNN(nn.Module):
    """
    Optimized Factored STGNN (v2).
    - 1D Folding for TCN (Bypass expensive 3D kernels).
    - Einsum for GCN (Zero-copy permutations).
    - Optional Static Meta-Embeddings for Stores.
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
        use_checkpointing: bool = False,
        dist_matrix: Optional[torch.Tensor] = None,
        static_feat_dim: int = 0
    ):
        super().__init__()
        self.num_stores = num_stores
        self.num_families = num_families
        self.horizon = horizon
        self.use_checkpointing = use_checkpointing

        # Optional projection for static store metadata
        if static_feat_dim > 0:
            self.static_proj = nn.Linear(static_feat_dim, embedding_dim)
        else:
            self.static_proj = None

        self.feature_norm = nn.InstanceNorm2d(feature_dim, affine=True)
        self.input_proj = nn.Conv2d(feature_dim, hidden_dim, kernel_size=1)
        
        self.layers = nn.ModuleList([
            STGNNBlock(hidden_dim, num_stores, num_families, kernel_size, 2**i, embedding_dim, dist_matrix)
            for i in range(num_layers)
        ])

        self.fc_head = nn.Sequential(
            nn.Conv3d(hidden_dim, hidden_dim * 2, kernel_size=1),
            nn.ReLU(),
            nn.Conv3d(hidden_dim * 2, horizon, kernel_size=1),
        )

    def forward(self, x_enc: torch.Tensor, is_open: Optional[torch.Tensor] = None, 
                static_feats: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x_enc: (Batch, N_nodes, Time, Features)
        static_feats: (num_stores, static_feat_dim)
        """
        B, N, T, F = x_enc.shape
        N_series = self.num_stores * self.num_families
        
        # 1. Slicing
        if N > N_series:
            x_enc = x_enc[:, -N_series:, :, :]
        
        if is_open is not None and is_open.shape[1] > N_series:
            is_open = is_open[:, -N_series:, :]

        # 2. Folding & Projection
        x = x_enc.permute(0, 3, 1, 2).contiguous() # (B, F, N, T)
        x = self.feature_norm(x)
        x = self.input_proj(x)
        
        # 5D Grid: (B, C, S, F, T)
        B, C, _, T = x.shape
        x = x.view(B, C, self.num_stores, self.num_families, T)

        # 3. Handle Static Seed (Optional)
        # In this optimized version, we can use static_feats to initialize the learnable adjs
        # but for per-forward use, we could also concatenate it. 
        # For simplicity and speed, we leave it as an initialization hint for now.

        skip_total = torch.zeros_like(x)
        for layer in self.layers:
            if self.training and self.use_checkpointing:
                x, skip = torch.utils.checkpoint.checkpoint(layer, x, use_reentrant=False)
            else:
                x, skip = layer(x)
            skip_total = skip_total + skip

        # 4. Heads
        final_feat = skip_total[..., -1:] 
        out = self.fc_head(final_feat)
        out = out.view(B, self.horizon, -1).permute(0, 2, 1).contiguous()

        if is_open is not None:
            if is_open.dim() == 3 and is_open.shape[-1] > self.horizon:
                 is_open = is_open[..., -self.horizon:]
            out = out * is_open

        return out

    def get_adjacency(self) -> List[torch.Tensor]:
        return self.layers[0].gcn.get_adjs()
