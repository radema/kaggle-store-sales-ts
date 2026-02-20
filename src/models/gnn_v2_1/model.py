import torch
import torch.nn as nn
from typing import Tuple
from torch_geometric.data import Data

# Import building blocks from gnn_v2
from src.models.gnn_v2.modules import TemporalBlock, SpatialBlock

class SalesGNNv2_1(nn.Module):
    """
    Hybrid-Style GNN v2.1 Architecture.
    Features a skip-connection trend block and a gating mechanism.
    Dual-output architecture returning (final_pred, trend).
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 2,
        gat_heads: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Encoder
        self.encoder = nn.Linear(in_channels, hidden_channels)
        
        # 2. Temporal Block
        self.temporal_block = TemporalBlock(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 3. Spatial Layers
        self.spatial_layers = nn.ModuleList([
            SpatialBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                heads=gat_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 4. Parallel Heads
        self.trend_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.residual_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )
        
        self.gate_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            data: PyG Data object containing x, edge_index, batch_size.
            
        Returns:
            Tuple containing (final_pred, trend) for the target nodes.
        """
        x, edge_index = data.x, data.edge_index
        batch_size = data.batch_size
        
        # Encode
        x = self.encoder(x)
        
        # Temporal Phase
        x_temporal_seq = self.temporal_block(x)
        
        # Take the last time step as the temporal summary
        x_temporal = x_temporal_seq[:, -1, :]
        
        # Extract trend BEFORE spatial mixing
        trend = self.trend_head(x_temporal)
        
        # Spatial Phase
        x_spatial = x_temporal
        for layer in self.spatial_layers:
            x_spatial = layer(x_spatial, edge_index)
            
        # Extract res and gate AFTER spatial mixing
        res = self.residual_head(x_spatial)
        gate = self.gate_head(x_spatial)
        
        # Compute final prediction
        final_pred = (trend + res) * torch.sigmoid(gate)
        
        # Slice outputs to only return the batch target nodes
        return final_pred[:batch_size], trend[:batch_size]
