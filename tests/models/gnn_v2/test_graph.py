import os
import pytest
import torch
import pandas as pd
from src.models.gnn_v2.graph import GraphFactory

def test_graph_construction():
    # Mock or use real data
    stores_path = 'data/raw/stores.csv'
    # Use the 33 families identified
    families = [
        'AUTOMOTIVE', 'BABY CARE', 'BEAUTY', 'BEVERAGES', 'BOOKS', 'BREAD/BAKERY', 
        'CELEBRATION', 'CLEANING', 'DAIRY', 'DELI', 'EGGS', 'FROZEN FOODS', 
        'GROCERY I', 'GROCERY II', 'HARDWARE', 'HOME AND KITCHEN I', 
        'HOME AND KITCHEN II', 'HOME APPLIANCES', 'HOME CARE', 'LADIESWEAR', 
        'LAWN AND GARDEN', 'LINGERIE', 'LIQUOR,WINE,BEER', 'MAGAZINES', 
        'MEATS', 'PERSONAL CARE', 'PET SUPPLIES', 'PLAYERS AND ELECTRONICS', 
        'POULTRY', 'PREPARED FOODS', 'PRODUCE', 'SCHOOL AND OFFICE SUPPLIES', 
        'SEAFOOD'
    ]
    
    factory = GraphFactory(stores_path, families)
    data = factory.build_graph()
    
    # 1. Verify Node Count
    assert data.num_nodes == 1782, f"Expected 1782 nodes, got {data.num_nodes}"
    
    # 2. Verify Edge Index properties
    assert data.edge_index.dtype == torch.long
    assert data.edge_index.shape[0] == 2
    
    # 3. Verify Connectivity: Intra-Store
    # Pick Store 1, Family 'AUTOMOTIVE' (index 0) and 'BABY CARE' (index 1)
    # They should be connected
    node_0 = factory.get_node_index(1, 'AUTOMOTIVE')
    node_1 = factory.get_node_index(1, 'BABY CARE')
    
    edges = data.edge_index.t().tolist()
    assert [node_0, node_1] in edges or [node_1, node_0] in edges, "Intra-store connection missing"
    
    # 4. Verify Connectivity: Inter-Store (Same Family, Same City)
    # Store 1 and Store 2 are both in Quito
    # node for 'AUTOMOTIVE' in Store 1 and Store 2 should be connected
    node_s1 = factory.get_node_index(1, 'AUTOMOTIVE')
    node_s2 = factory.get_node_index(2, 'AUTOMOTIVE')
    assert [node_s1, node_s2] in edges or [node_s2, node_s1] in edges, "Inter-store same city connection missing"
    
    # 5. Verify no isolated nodes
    # Check degree of all nodes
    degrees = torch.zeros(data.num_nodes)
    for i in range(data.edge_index.shape[1]):
        degrees[data.edge_index[0, i]] += 1
        
    assert (degrees > 0).all(), "Found isolated nodes"
    
    print(f"Graph verification successful!")
    print(f"Total nodes: {data.num_nodes}")
    print(f"Total edges: {data.edge_index.shape[1]}")
    print(f"Average degree: {data.edge_index.shape[1] / data.num_nodes:.2f}")

if __name__ == "__main__":
    test_graph_construction()
