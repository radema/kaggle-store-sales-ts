from dataclasses import dataclass, field
from typing import List

@dataclass
class GNNConfig:
    """Configuration for GNN v2.0 (PyTorch Geometric + GATv2)."""
    
    # Architecture
    hidden_dim: int = 128
    gat_heads: int = 8
    dropout: float = 0.2
    
    # Sampling & Batching
    neighbor_sizes: List[int] = field(default_factory=lambda: [15, 10])
    batch_size_nodes: int = 1024
    window: int = 90
    horizon: int = 16
    
    # Training Defaults
    learning_rate: float = 0.001
    epochs: int = 15
    early_stopping_patience: int = 4
    
    def __post_init__(self):
        # Validation or derived parameters can go here
        pass
