from dataclasses import dataclass, field
from typing import List

@dataclass
class GNNConfig:
    """Configuration for GNN v2.0 (PyTorch Geometric + GATv2)."""
    
    # Architecture
    hidden_dim: int = 64
    gat_heads: int = 4
    dropout: float = 0.1
    
    # Sampling & Batching
    neighbor_sizes: List[int] = field(default_factory=lambda: [10, 10])
    batch_size_nodes: int = 1024
    
    # Training Defaults
    learning_rate: float = 0.001
    epochs: int = 100
    early_stopping_patience: int = 10
    
    def __post_init__(self):
        # Validation or derived parameters can go here
        pass
