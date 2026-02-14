import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class LearnableAdjacency(nn.Module):
    """
    A dense learnable adjacency matrix that can be used for global node mixing
    and sparsity regularization.
    """

    def __init__(self, num_nodes):
        super().__init__()
        self.adj = nn.Parameter(torch.randn(num_nodes, num_nodes))

    def forward(self):
        # Return sigmoid-activated adjacency for [0, 1] range
        return torch.sigmoid(self.adj)


class GNNLayer(nn.Module):
    """
    Modular GNN layer wrapping PyG's GATv2Conv.
    """

    def __init__(self, in_channels, out_channels, heads=1):
        super().__init__()
        self.conv = GATv2Conv(in_channels, out_channels, heads=heads, concat=True)
        self.norm = nn.LayerNorm(out_channels * heads)
        self.relu = nn.ReLU()

    def forward(self, x, edge_index):
        x = self.conv(x, edge_index)
        x = self.norm(x)
        return self.relu(x)


class SalesGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network for Sales Forecasting.
    Implements a Seq2Seq architecture with:
    - GRU for temporal extraction
    - GNN for spatial/hierarchical mixing
    - Hard Gating for closed stores
    """

    def __init__(
        self, num_nodes, feature_dim, hidden_dim, edge_index, horizon, embedding_dim=16
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.embedding_dim = embedding_dim

        # Buffer for static graph structure
        self.register_buffer("edge_index", edge_index)

        # Node Embeddings: Learnable signature for each node (Store, Family, Series)
        self.node_embedding = nn.Embedding(num_nodes, embedding_dim)

        # Task 5.2: Learnable Adjacency
        self.learnable_adj = LearnableAdjacency(num_nodes)

        # Combined input dim: shared features + static identity embedding
        input_dim = feature_dim + embedding_dim

        # Encoder: Extract temporal features per node, then mix spatially
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.encoder_gnn = GNNLayer(hidden_dim, hidden_dim)

        # Decoder: Step-by-step prediction
        self.decoder_gru = nn.GRUCell(input_dim, hidden_dim)
        self.decoder_gnn = GNNLayer(hidden_dim, hidden_dim)

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x_enc, x_dec, is_open):
        """
        Forward pass for the GNN model.

        Args:
            x_enc: (Batch, Nodes, T_in, Features) - History
            x_dec: (Batch, Nodes, T_out, Features) - Future knowns
            is_open: (Batch, Nodes, T_out) - Hard gating mask

        Returns:
            torch.Tensor: (Batch, Nodes, T_out) - Predictions
        """
        batch_size = x_enc.size(0)
        num_nodes = x_enc.size(1)
        device = x_enc.device

        # 0. Prepare Node Embeddings
        # Shape: (Nodes, E) -> expand to (Batch, Nodes, T, E)
        node_ids = torch.arange(num_nodes, device=device)
        node_embed = self.node_embedding(node_ids)  # (Nodes, E)

        # 1. Temporal Encoding
        # Expand node embeddings for encoder history (Batch, Nodes, T_in, E)
        node_embed_enc = node_embed.view(1, num_nodes, 1, self.embedding_dim)
        node_embed_enc = node_embed_enc.expand(batch_size, -1, x_enc.size(2), -1)

        # Concat features and embeddings: (B, N, T_in, F+E)
        x_enc_combined = torch.cat([x_enc, node_embed_enc], dim=-1)

        # Process all nodes across the batch efficiently
        # Shape: (B * N, T_in, F+E)
        x_enc_flat = x_enc_combined.view(
            -1, x_enc.size(2), self.feature_dim + self.embedding_dim
        )
        _, h_temporal = self.encoder_gru(x_enc_flat)
        h_temporal = h_temporal.squeeze(0)  # (B * N, H)

        # 2. Spatial Encoding (Static Graph)
        edge_index_batch = self._get_batch_edge_index(batch_size, device)
        h_spatial = self.encoder_gnn(h_temporal, edge_index_batch)  # (B * N, H)

        # Initialize decoder state
        hidden = h_spatial
        outputs = []

        # 3. Recurrent Decoding
        # Prepare expanded embeddings for decoder (Batch, Nodes, E)
        node_embed_dec = node_embed.view(1, num_nodes, self.embedding_dim)
        node_embed_dec = node_embed_dec.expand(batch_size, -1, -1)

        for t in range(self.horizon):
            # Future known features at time t (B, N, F)
            x_t = x_dec[:, :, t, :]

            # Combine with embeddings (B, N, F+E)
            x_t_combined = torch.cat([x_t, node_embed_dec], dim=-1)
            x_t_flat = x_t_combined.reshape(
                -1, self.feature_dim + self.embedding_dim
            )  # (B * N, F+E)

            # Temporal update
            hidden = self.decoder_gru(x_t_flat, hidden)

            # Spatial update
            hidden = self.decoder_gnn(hidden, edge_index_batch)

            # Output projection (B * N, 1) -> (B, N)
            out_t = self.fc_out(hidden).view(batch_size, self.num_nodes)
            outputs.append(out_t)

        # Stack and Transpose: (T_out, B, N) -> (B, N, T_out)
        final_output = torch.stack(outputs, dim=2)

        # 4. Hard Gating: Zero out predictions for closed days
        final_output = final_output * is_open

        return final_output

    def _get_batch_edge_index(self, batch_size, device):
        """
        Adapts the static edge_index for use with a batch of graphs.
        PyG treats batches as a single large graph with disconnected components.
        """
        if batch_size == 1:
            return self.edge_index

        num_edges = self.edge_index.size(1)

        # Edge index: (2, B * NumEdges)
        repeated_edges = self.edge_index.repeat(1, batch_size)

        # Offsets per batch item: [0, ..., 0, N, ..., N, ..., (B-1)N, ..., (B-1)N]
        # (B, NumEdges) -> flatten
        offsets = torch.arange(batch_size, device=device).view(-1, 1) * self.num_nodes
        offsets = offsets.repeat(1, num_edges).view(-1)

        return repeated_edges + offsets

    def get_adjacency(self):
        """Returns the learnable adjacency matrix."""
        return self.learnable_adj()
