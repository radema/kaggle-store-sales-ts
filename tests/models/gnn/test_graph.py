import pandas as pd
from src.models.gnn.graph import build_adjacency


def test_build_adjacency_spatial():
    # Mock stores: 2 in Quito, 1 in Guayaquil
    stores = pd.DataFrame(
        {"store_nbr": [1, 2, 3], "city": ["Quito", "Quito", "Guayaquil"]}
    )

    # Empty families for now to focus on spatial
    adj = build_adjacency(stores, families=[])
    spatial_edges = adj["spatial"]

    # Store 1 and 2 are in Quito, so they should be connected (bidirectional)
    # Edge index should have (1, 2) and (2, 1) or similar depending on indexing
    # Usually we use 0-based indexing for internal graph logic

    edges_list = spatial_edges.t().tolist()
    assert [0, 1] in edges_list
    assert [1, 0] in edges_list
    assert [0, 2] not in edges_list


def test_build_adjacency_hierarchy():
    stores = pd.DataFrame({"store_nbr": [1], "city": ["Quito"]})
    families = ["BEVERAGES", "MEATS"]

    # Total nodes: 1 Store + 2 Families + 2 Series (Store1-Bev, Store1-Meat)
    adj = build_adjacency(stores, families)
    hierarchy_edges = adj["hierarchy"]

    # We need to check if Series nodes are connected to Store and Family
    # The implementation will define the node order, let's assume:
    # Nodes: [Store1, Bev, Meat, Series1_Bev, Series1_Meat]
    # Series1_Bev (index 3) -> Store1 (index 0) and Bev (index 1)

    edges_list = hierarchy_edges.t().tolist()
    assert len(edges_list) > 0
