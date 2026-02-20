import torch
import torch.nn as nn
from torch_geometric.data import Data
from .modules import TemporalBlock, SpatialBlock
from typing import Optional, List

class SalesGNNv2(nn.Module):
    """
    ST-GNN for Store Sales Forecasting.
    Architecture: Folded TCN -> GATv2 Spatial Mixing -> MLP Predictor.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int, # Horizon (default 16)
        num_layers: int = 2,
        gat_heads: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 1. Encoder: Project features to hidden dimension
        self.encoder = nn.Linear(in_channels, hidden_channels)
        
        # 2. Temporal Phase: TCN to extract temporal patterns
        # We use a single block here, but it could be expanded
        self.temporal_block = TemporalBlock(
            hidden_channels, 
            hidden_channels, 
            kernel_size=kernel_size,
            dropout=dropout
        )
        
        # 3. Spatial Phase: Multiple GATv2 layers
        self.spatial_layers = nn.ModuleList([
            SpatialBlock(
                hidden_channels, 
                hidden_channels, 
                heads=gat_heads,
                dropout=dropout
            ) for _ in range(num_layers)
        ])
        
        # 4. Predictor: MLP head to predict horizon
        self.predictor = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Args:
            data: PyG Data object containing:
                - x: (N_subgraph, T, F)
                - edge_index: (2, E_subgraph)
                - batch_size: number of target nodes (at the front)
        Returns:
            (batch_size, Horizon)
        """
        x, edge_index = data.x, data.edge_index
        batch_size = data.batch_size
        
        # 1. Encoder
        # Linear works on the last dimension of (N, T, F) -> (N, T, H)
        x = self.encoder(x)
        
        # 2. Temporal Phase
        # (N, T, H) -> (N, T, H)
        x = self.temporal_block(x)
        
        # Take the last time step for spatial mixing (Causal TCN history)
        # (N, T, H) -> (N, H)
        x_mixed = x[:, -1, :]
        
        # 3. Spatial Phase
        for layer in self.spatial_layers:
            x_mixed = layer(x_mixed, edge_index)
            
        # 4. Predictor
        # (N, H) -> (N, Horizon)
        out = self.predictor(x_mixed)
        
        # 5. Output Slicing: Only return target nodes
        return out[:batch_size]
