import torch
import pandas as pd
import numpy as np
from pathlib import Path
from src.models.gnn_v2.config import GNNConfig
from src.models.gnn_v2.graph import GraphFactory
from src.models.gnn_v2.loader import TemporalGraphDataset, make_gnn_v2_loader

def verify():
    config = GNNConfig()
    data_dir = Path("data/processed/gnn")
    proc_dir = Path("data/processed")
    raw_dir = Path("data/raw")
    
    # 1. Load Data
    features = torch.from_numpy(np.load(data_dir / "features.npy")).float()
    labels = torch.from_numpy(np.load(data_dir / "labels.npy")).float()
    train_df = pd.read_parquet(proc_dir / "train.parquet")
    families = sorted(train_df["family"].unique())
    factory = GraphFactory(raw_dir / "stores.csv", families)
    static_graph = factory.build_graph()
    
    dataset = TemporalGraphDataset(static_graph, features, labels, config.window, config.horizon)
    
    # 2. Create a Day-Batch
    time_idx = 0 
    times = [time_idx] * dataset.num_nodes
    nodes = list(range(dataset.num_nodes))
    
    loader = make_gnn_v2_loader(dataset, times, nodes, batch_size=dataset.num_nodes, shuffle=False)
    batch = next(iter(loader))
    
    print(f"Batch Size: {batch.batch_size}")
    
    # 3. Verify Lexical Alignment
    # The submission expectations (Lexical order) are:
    # 1. Automotive
    # 1. Baby Care...
    # The GraphFactory generates them in this order exactly.
    expected_ids = torch.arange(dataset.num_nodes)
    matches = torch.equal(batch.n_id[:batch.batch_size], expected_ids)
    print(f"Lexical Order Matches: {matches}")
    
    # Double check the metadata structure
    for i in range(3):
        store_nbr = factory.node_metadata[i]['store_nbr']
        family = factory.node_metadata[i]['family']
        print(f"Node {i}: Store {store_nbr}, Family {family}")

if __name__ == '__main__':
    verify()
