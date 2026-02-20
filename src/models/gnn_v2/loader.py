import torch
from torch_geometric.data import Data
from torch_geometric.utils import k_hop_subgraph, subgraph
from typing import List, Tuple, Optional, Union
import numpy as np

class TemporalGraphDataset:
    """
    Dataset container for static spatial graph and temporal feature tensors.
    """
    def __init__(
        self, 
        static_graph: Data, 
        features: torch.Tensor, 
        labels: torch.Tensor,
        window: int,
        horizon: int
    ):
        """
        Args:
            static_graph: PyG Data object with static edge_index and node features.
            features: Tensor of shape (Time_Steps, Num_Nodes, Num_Features)
            labels: Tensor of shape (Time_Steps, Num_Nodes, 1)
            window: int
            horizon: int
        """
        self.static_graph = static_graph
        self.features = features
        self.labels = labels
        self.window = window
        self.horizon = horizon
        
        self.num_nodes = static_graph.num_nodes
        # Number of valid starting points for a batch
        self.num_time_steps = features.size(0) - window - horizon + 1

class TemporalNeighborLoader:
    """
    Custom loader that wraps spatial sampling and temporal feature retrieval.
    Note: Uses k_hop_subgraph as a robust alternative to NeighborLoader.
    Enforces that target nodes appear first in the batch.
    """
    def __init__(
        self, 
        dataset: TemporalGraphDataset, 
        time_idx_list: List[int], 
        node_idx_list: List[int], 
        batch_size: int, 
        num_neighbors: List[int], 
        shuffle: bool = True
    ):
        self.dataset = dataset
        self.time_idx_list = torch.tensor(time_idx_list, dtype=torch.long)
        self.node_idx_list = torch.tensor(node_idx_list, dtype=torch.long)
        self.batch_size = batch_size
        self.num_neighbors = num_neighbors
        self.shuffle = shuffle
        
        self.num_samples = len(time_idx_list)
        self.indices = torch.arange(self.num_samples)

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.num_samples)
            self.indices = perm
        else:
            self.indices = torch.arange(self.num_samples)
            
        for i in range(0, self.num_samples, self.batch_size):
            batch_indices = self.indices[i : i + self.batch_size]
            t_slice = self.time_idx_list[batch_indices]
            n_slice = self.node_idx_list[batch_indices]
            
            # Ensure time homogeneity in batch
            unique_times = torch.unique(t_slice)
            time_idx = unique_times[0].item()
            
            # SPATIAL SAMPLING
            num_hops = len(self.num_neighbors)
            
            # OPTIMIZATION: If we are processing all nodes, skip subgraph sampling
            if len(n_slice) == self.dataset.num_nodes:
                n_id = torch.arange(self.dataset.num_nodes)
                sub_edge_index = self.dataset.static_graph.edge_index
                mapping = n_slice # they are likely the same
            else:
                n_id, sub_edge_index, mapping, _ = k_hop_subgraph(
                    node_idx=n_slice,
                    num_hops=num_hops,
                    edge_index=self.dataset.static_graph.edge_index,
                    relabel_nodes=True
                )
            
            # ENFORCE TARGET ORDER: Move target nodes to the front
            # mapping[i] is the position of n_slice[i] in n_id
            # We want mapping to be [0, 1, ..., len(n_slice)-1]
            
            # 1. Create a mask for target nodes in the sampled n_id
            is_target = torch.zeros(n_id.size(0), dtype=torch.bool)
            is_target[mapping] = True
            
            # 2. Reorder n_id: targets first, then others
            target_n_id = n_id[mapping]
            other_n_id = n_id[~is_target]
            new_n_id = torch.cat([target_n_id, other_n_id], dim=0)
            
            # 3. Create a remapping from old local indices to new local indices
            # old_idx -> new_idx
            reorder_map = torch.zeros(n_id.size(0), dtype=torch.long)
            reorder_map[mapping] = torch.arange(len(n_slice))
            
            other_indices = torch.where(~is_target)[0]
            range_others = torch.arange(len(n_slice), n_id.size(0))
            
            reorder_map[other_indices] = range_others
            
            # 4. Update sub_edge_index with new mapping
            new_edge_index = reorder_map[sub_edge_index]
            
            # Retrieve temporal features Sliced by new_n_id
            # time_idx is the index in the 4D virtual space (0 to num_time_steps-1)
            # In the 3D space, the window starts at time_idx
            # and ends at time_idx + window.
            window = self.dataset.window
            horizon = self.dataset.horizon
            
            # x_temporal: (window, num_sampled_nodes, feat) -> (num_sampled_nodes, window, feat)
            x_temporal = self.dataset.features[time_idx : time_idx + window, new_n_id, :]
            x_temporal = x_temporal.transpose(0, 1)
            
            # y_target: (horizon, num_target_nodes, 1) -> (num_target_nodes, horizon)
            y_target = self.dataset.labels[time_idx + window : time_idx + window + horizon, n_slice, 0]
            y_target = y_target.transpose(0, 1)
            
            batch = Data(
                x=x_temporal,
                edge_index=new_edge_index,
                y=y_target,
                n_id=new_n_id,
                batch_size=len(n_slice)
            )
            
            yield batch

    def __len__(self):
        return (self.num_samples + self.batch_size - 1) // self.batch_size

def make_gnn_v2_loader(
    dataset: TemporalGraphDataset,
    time_indices: List[int],
    node_indices: List[int],
    batch_size: int,
    num_neighbors: List[int] = [10, 10],
    shuffle: bool = True,
    **kwargs
) -> TemporalNeighborLoader:
    return TemporalNeighborLoader(
        dataset, time_indices, node_indices, batch_size, num_neighbors, shuffle
    )
