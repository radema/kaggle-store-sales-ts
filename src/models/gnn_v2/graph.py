import pandas as pd
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, remove_self_loops
from typing import List, Dict, Any, Tuple
import itertools

class GraphFactory:
    """
    Factory for building the GNN v2.0 homogeneous graph.
    Nodes represent (Store, Family) tuples.
    Total nodes: 54 stores * 33 families = 1782 nodes.
    """
    
    def __init__(self, stores_path: str, families: List[str]):
        self.stores_df = pd.read_csv(stores_path).sort_values('store_nbr')
        self.families = sorted(families)
        self.num_stores = len(self.stores_df)
        self.num_families = len(self.families)
        self.num_nodes = self.num_stores * self.num_families
        
        if self.num_nodes != 1782:
            import warnings
            warnings.warn(f"Expected 1782 nodes (54x33), but got {self.num_nodes} ({self.num_stores}x{self.num_families})")

        # Create lexical mapping: Store 1/Family 1, Store 1/Family 2 ...
        self.node_mapping: List[Tuple[int, str]] = []
        for store_nbr in self.stores_df['store_nbr'].unique():
            for family in self.families:
                self.node_mapping.append((store_nbr, family))
        
        # Pre-calculate metadata for each node
        self.node_metadata = [self._get_meta(s, f) for s, f in self.node_mapping]

    def _get_meta(self, store_nbr: int, family: str) -> Dict[str, Any]:
        store_row = self.stores_df[self.stores_df['store_nbr'] == store_nbr].iloc[0]
        return {
            'store_nbr': store_nbr,
            'family': family,
            'city': store_row['city'],
            'type': store_row['type'],
            'cluster': store_row['cluster']
        }

    def build_graph(self) -> Data:
        """Builds the spatial graph based on shared store or shared family attributes."""
        edges = []
        
        # Condition 1: Intra-Store (Same store_nbr)
        # Connect all families within the same store
        for s_idx in range(self.num_stores):
            # All families for this store have indices in this range
            start_node = s_idx * self.num_families
            end_node = (s_idx + 1) * self.num_families
            store_node_ids = range(start_node, end_node)
            
            for i, j in itertools.combinations(store_node_ids, 2):
                edges.append((i, j))
                edges.append((j, i))
                
        # Condition 2: Inter-Store (Same family AND (same city OR type OR cluster))
        for f_idx in range(self.num_families):
            # All nodes for this family across different stores
            family_node_ids = [s_idx * self.num_families + f_idx for s_idx in range(self.num_stores)]
            
            for i_idx, j_idx in itertools.combinations(family_node_ids, 2):
                meta_i = self.node_metadata[i_idx]
                meta_j = self.node_metadata[j_idx]
                
                if (meta_i['city'] == meta_j['city'] or 
                    meta_i['type'] == meta_j['type'] or 
                    meta_i['cluster'] == meta_j['cluster']):
                    edges.append((i_idx, j_idx))
                    edges.append((j_idx, i_idx))
        
        # Convert to tensor
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        
        # Remove duplicates and self-loops just in case
        edge_index, _ = remove_self_loops(edge_index)
        edge_index = coalesce(edge_index)
        
        # Node features: Static indices/embeddings
        # Mapping categories to indices
        cities = sorted(self.stores_df['city'].unique())
        types = sorted(self.stores_df['type'].unique())
        clusters = sorted(self.stores_df['cluster'].unique())
        
        city_map = {v: i for i, v in enumerate(cities)}
        type_map = {v: i for i, v in enumerate(types)}
        cluster_map = {v: i for i, v in enumerate(clusters)}
        family_map = {v: i for i, v in enumerate(self.families)}
        store_map = {v: i for i, v in enumerate(self.stores_df['store_nbr'].unique())}
        
        x_indices = []
        for meta in self.node_metadata:
            x_indices.append([
                store_map[meta['store_nbr']],
                family_map[meta['family']],
                city_map[meta['city']],
                type_map[meta['type']],
                cluster_map[meta['cluster']]
            ])
            
        x = torch.tensor(x_indices, dtype=torch.float)
        
        data = Data(x=x, edge_index=edge_index)
        
        # Attach readable metadata for debugging/sampling if needed
        data.node_store_nbr = torch.tensor([m['store_nbr'] for m in self.node_metadata])
        data.node_family_idx = torch.tensor([family_map[m['family']] for m in self.node_metadata])
        
        return data

    def get_node_index(self, store_nbr: int, family: str) -> int:
        """Helper to get node index from store and family."""
        store_idx = list(self.stores_df['store_nbr']).index(store_nbr)
        family_idx = self.families.index(family)
        return store_idx * self.num_families + family_idx

    def get_meta_from_index(self, index: int) -> Dict[str, Any]:
        """Helper to get metadata from node index."""
        return self.node_metadata[index]
