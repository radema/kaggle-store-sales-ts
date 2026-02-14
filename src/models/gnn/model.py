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

        # 0. Numerical Stability & Architecture
        self.total_embedding_dim = 2 * embedding_dim

        # Temporal Encoder: Processes only dynamic features
        # (Much faster than processing combined features)
        self.encoder_gru_features = nn.GRU(feature_dim, hidden_dim, batch_first=True)

        # Projection for node embeddings to match hidden_dim
        self.node_projector = nn.Sequential(
            nn.Linear(self.total_embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim)
        )

        self.spatial_mixer = GNNLayer(hidden_dim, hidden_dim)

        # Decoder: processes dynamic features + static context
        self.decoder_gru = nn.GRU(feature_dim, hidden_dim, batch_first=True)

        self.input_norm = nn.LayerNorm(feature_dim)
        self.fc_out = nn.Sequential(nn.Dropout(0.1), nn.Linear(hidden_dim, 1))

        # Efficiency Caches
        self._edge_index_cache = {}

    def _get_node_embeddings(self, device):
        """Unified node embedding matrix of shape (TotalNodes, 2*E)."""
        E = self.embedding_dim
        # Retrieve weights directly for speed
        all_stores = self.store_embedding.weight
        all_families = self.family_embedding.weight

        # Pre-allocate zero padding
        z_stores = torch.zeros(self.num_stores, E, device=device)
        z_families = torch.zeros(self.num_families, E, device=device)

        # Series Node Embeddings (S*F, 2E)
        store_ext = (
            all_stores.unsqueeze(1).expand(-1, self.num_families, -1).reshape(-1, E)
        )
        family_ext = (
            all_families.unsqueeze(0).expand(self.num_stores, -1, -1).reshape(-1, E)
        )

        # Build columns efficiently
        col1 = torch.cat([all_stores, z_families, store_ext], dim=0)
        col2 = torch.cat([z_stores, all_families, family_ext], dim=0)

        return torch.cat([col1, col2], dim=-1)

    def forward(self, x_enc, x_dec, is_open):
        batch_size = x_enc.size(0)
        num_nodes = x_enc.size(1)
        seq_len = x_enc.size(2)
        device = x_enc.device

        # 1. Temporal Encoder
        # x_enc: (B, N, T, F) -> (B*N, T, F)
        x_enc_flat = x_enc.reshape(-1, seq_len, self.feature_dim)

        # MPS Efficiency: Ensure contiguous memory before RNN kernel
        x_enc_flat = self.input_norm(x_enc_flat).contiguous()
        _, h_temporal = self.encoder_gru_features(x_enc_flat)
        h_temporal = h_temporal.view(batch_size, num_nodes, self.hidden_dim)

        # 2. Inject Node Context
        node_embed_temp = self._get_node_embeddings(device)
        node_proj = self.node_projector(node_embed_temp[:num_nodes])  # (N, H)

        # Fast Contextualization: Broadcasting addition (B, N, H) + (1, N, H)
        h_contextualized = (h_temporal + node_proj.unsqueeze(0)).view(
            -1, self.hidden_dim
        )
        h_contextualized = h_contextualized.contiguous()

        # Spatial MIXING
        edge_index_batch = self._get_batch_edge_index(batch_size, num_nodes, device)
        h_spatial = self.spatial_mixer(h_contextualized, edge_index_batch)  # (B*N, H)

        # 3. Decoder: Vectorized Rollout
        h_decoder_init = h_spatial.unsqueeze(0)

        x_dec_flat = x_dec.reshape(-1, self.horizon, self.feature_dim)
        x_dec_flat = self.input_norm(x_dec_flat)

        decoder_out, _ = self.decoder_gru(x_dec_flat, h_decoder_init)  # (B*N, Hori, H)

        # Project and Apply to original batch shape
        proj_out = self.fc_out(decoder_out).view(batch_size, num_nodes, self.horizon)
        final_output = proj_out * is_open

        return final_output

    def _get_batch_edge_index(self, batch_size, num_nodes, device):
        """
        Adapts the static edge_index for use with a batch of graphs.
        Caches the result to avoid recomputing offsets every step.
        """
        cache_key = (batch_size, num_nodes, device.type)
        if cache_key in self._edge_index_cache:
            return self._edge_index_cache[cache_key]

        if batch_size == 1:
            return self.edge_index

        num_edges = self.edge_index.size(1)
        repeated_edges = self.edge_index.repeat(1, batch_size)

        offsets = torch.arange(batch_size, device=device).view(-1, 1) * num_nodes
        offsets = offsets.expand(-1, num_edges).reshape(-1)

        result = repeated_edges + offsets
        self._edge_index_cache[cache_key] = result
        return result

    def get_adjacency(self):
        """Returns the learnable adjacency matrix."""
        return self.learnable_adj()
