import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Optional

class TemporalBlock(nn.Module):
    """
    High-performance 1D-Conv layer (Folded).
    Treats nodes as an extension of the batch dimension.
    Input shape: (Num_Nodes, Window, Features)
    Output shape: (Num_Nodes, Window, Out_Features)
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.kernel_size = kernel_size
        self.padding = kernel_size - 1
        
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size, 
            padding=0 # We will pad manually for causality
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(out_channels)
        
        # Residual connection if shapes match
        self.res_conv = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, T, F) tensor
        Returns:
            (N, T, F') tensor
        """
        N, T, F_in = x.shape
        
        # Residual
        res = self.res_conv(x)
        
        # Prepare for Conv1d: (N, F_in, T)
        x_folded = x.transpose(1, 2)
        
        # Causal Padding: pad the front of the time dimension
        # padding format for pad(): (left, right, top, bottom, front, back)
        # For 1D (last dim), it's (left, right)
        x_padded = F.pad(x_folded, (self.padding, 0))
        
        # Convolution
        x_conv = self.conv(x_padded)
        
        # Back to (N, T, F_out)
        x_unfolded = x_conv.transpose(1, 2)
        x_unfolded = self.dropout(x_unfolded)
        
        # Add residual and norm
        out = self.norm(x_unfolded + res)
        return out

class SpatialBlock(nn.Module):
    """
    Wraps GATv2Conv with residuals, LayerNorm, and dropout.
    """
    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.conv = GATv2Conv(
            in_channels, 
            out_channels // heads, 
            heads=heads, 
            dropout=dropout,
            concat=True
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Residual connection
        self.res_lin = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, F_in)
            edge_index: (2, E)
        """
        res = self.res_lin(x)
        
        x_conv = self.conv(x, edge_index)
        x_conv = self.dropout(x_conv)
        
        out = self.norm(x_conv + res)
        return out
