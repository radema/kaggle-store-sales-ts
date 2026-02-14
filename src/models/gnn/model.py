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
        self,
        num_stores,
        num_families,
        feature_dim,
        hidden_dim,
        edge_index,
        horizon,
        embedding_dim=16,
    ):
        super().__init__()
        self.num_stores = num_stores
        self.num_families = num_families
        self.num_nodes = num_stores + num_families + (num_stores * num_families)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.embedding_dim = embedding_dim

        # Buffer for static graph structure
        self.register_buffer("edge_index", edge_index)

        # 5.1 Hierarchical Embeddings
        self.store_embedding = nn.Embedding(num_stores, embedding_dim)
        self.family_embedding = nn.Embedding(num_families, embedding_dim)

        # 5.2 Learnable Adjacency
        self.learnable_adj = LearnableAdjacency(self.num_nodes)

        # Combined input dim
        self.total_embedding_dim = 2 * embedding_dim
        input_dim = feature_dim + self.total_embedding_dim

        # 0. Numerical Stability: Global Norm + Temporal Mix
        self.input_norm = nn.LayerNorm(input_dim)
        self.encoder_gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.spatial_mixer = GNNLayer(hidden_dim, hidden_dim)

        # Decoder: Iterative rollout using temporal GRU + pre-mixed spatial context
        self.decoder_gru = nn.GRUCell(input_dim, hidden_dim)

        self.fc_out = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

    def _get_node_embeddings(self, device):
        """Unified node embedding matrix of shape (TotalNodes, 2*E)."""
        all_stores = self.store_embedding(torch.arange(self.num_stores, device=device))
        all_families = self.family_embedding(
            torch.arange(self.num_families, device=device)
        )

        # 1. Store Nodes (S, 2E)
        store_nodes = torch.cat([all_stores, torch.zeros_like(all_stores)], dim=-1)

        # 2. Family Nodes (F, 2E)
        family_nodes = torch.cat([torch.zeros_like(all_families), all_families], dim=-1)

        # 3. Series Node Embeddings (S*F, 2E)
        store_expanded = all_stores.unsqueeze(1).expand(-1, self.num_families, -1)
        family_expanded = all_families.unsqueeze(0).expand(self.num_stores, -1, -1)
        series_nodes = torch.cat([store_expanded, family_expanded], dim=-1).reshape(
            -1, self.total_embedding_dim
        )

        return torch.cat([store_nodes, family_nodes, series_nodes], dim=0)

    def forward(self, x_enc, x_dec, is_open):
        batch_size = x_enc.size(0)
        num_nodes = x_enc.size(1)
        seq_len = x_enc.size(2)
        device = x_enc.device

        # 0. Cache Node Embeddings
        node_embed = self._get_node_embeddings(device)  # (TotalNodes, 2E)

        # Use embeddings for nodes present in batch
        node_embed = node_embed[:num_nodes]

        # Optimize Encoder: Concat dynamic features and static embeddings
        # x_enc: (B, N, T, F) -> (B*N, T, F)
        x_enc_flat = x_enc.reshape(-1, seq_len, self.feature_dim)

        # Efficient expansion: (N, 2E) -> (B*N, T, 2E)
        node_embed_expanded = (
            node_embed.view(1, num_nodes, 1, self.total_embedding_dim)
            .expand(batch_size, num_nodes, seq_len, self.total_embedding_dim)
            .reshape(batch_size * num_nodes, seq_len, self.total_embedding_dim)
        )

        x_enc_combined = torch.cat([x_enc_flat, node_embed_expanded], dim=-1)

        # Stability: Apply LayerNorm
        x_enc_combined = self.input_norm(x_enc_combined)

        _, h_temporal = self.encoder_gru(x_enc_combined)
        h_temporal = h_temporal.squeeze(0)  # (B*N, H)

        # Spatial MIXING: One-shot injection of neighbor context
        edge_index_batch = self._get_batch_edge_index(batch_size, num_nodes, device)
        h_spatial = self.spatial_mixer(h_temporal, edge_index_batch)  # (B*N, H)

        # 2. Decoder: State-Space Rollout
        hidden = h_spatial
        outputs = []

        # Pre-expand node embeddings for decoder rollout: (B*N, 2E)
        node_embed_dec_flat = (
            node_embed.view(1, num_nodes, self.total_embedding_dim)
            .expand(batch_size, num_nodes, self.total_embedding_dim)
            .reshape(batch_size * num_nodes, self.total_embedding_dim)
        )

        for t in range(self.horizon):
            x_t = x_dec[:, :, t, :].reshape(-1, self.feature_dim)  # (B*N, F)
            x_t_combined = torch.cat([x_t, node_embed_dec_flat], dim=-1)  # (B*N, F+2E)

            # Apply LayerNorm
            x_t_combined = self.input_norm(x_t_combined)

            # Spatial information is preserved in 'hidden' state
            hidden = self.decoder_gru(x_t_combined, hidden)

            out_t = self.fc_out(hidden).view(batch_size, num_nodes)
            outputs.append(out_t)

        final_output = torch.stack(outputs, dim=2)
        final_output = final_output * is_open

        return final_output

    def _get_batch_edge_index(self, batch_size, num_nodes, device):
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
        offsets = torch.arange(batch_size, device=device).view(-1, 1) * num_nodes
        offsets = offsets.repeat(1, num_edges).view(-1)

        return repeated_edges + offsets

    def get_adjacency(self):
        """Returns the learnable adjacency matrix."""
        return self.learnable_adj()
