import torch


def build_adjacency(stores_df, families):
    """
    Builds edge indices for spatial and hierarchical relationships.

    Node indexing scheme:
    1. Stores: 0 to S-1
    2. Families: S to S+F-1
    3. Series (Store-Family): S+F to S+F+(S*F)-1

    Args:
        stores_df: DataFrame with 'store_nbr', 'city'.
        families: List of family names.

    Returns:
        dict: {
            'spatial': torch.tensor (edge_index),
            'hierarchy': torch.tensor (edge_index)
        }
    """
    num_stores = len(stores_df)
    num_families = len(families)

    # Store mapping
    store_map = {nbr: i for i, nbr in enumerate(stores_df["store_nbr"])}

    # 1. Spatial Edges (Store -> Store)
    spatial_sources = []
    spatial_targets = []

    cities = stores_df["city"].unique()
    for city in cities:
        city_stores = stores_df[stores_df["city"] == city]["store_nbr"].tolist()
        for i in range(len(city_stores)):
            for j in range(len(city_stores)):
                if i != j:
                    spatial_sources.append(store_map[city_stores[i]])
                    spatial_targets.append(store_map[city_stores[j]])

    spatial_edge_index = torch.tensor(
        [spatial_sources, spatial_targets], dtype=torch.long
    )

    # 2. Hierarchical Edges
    # Series nodes start after Stores and Families
    hierarchy_sources = []
    hierarchy_targets = []

    series_offset = num_stores + num_families

    for s_idx, store_nbr in enumerate(stores_df["store_nbr"]):
        for f_idx, family_name in enumerate(families):
            series_node_idx = series_offset + (s_idx * num_families) + f_idx

            # Series -> Store
            hierarchy_sources.append(series_node_idx)
            hierarchy_targets.append(s_idx)

            # Store -> Series (Bidirectional)
            hierarchy_sources.append(s_idx)
            hierarchy_targets.append(series_node_idx)

            # Series -> Family
            family_node_idx = num_stores + f_idx
            hierarchy_sources.append(series_node_idx)
            hierarchy_targets.append(family_node_idx)

            # Family -> Series (Bidirectional)
            hierarchy_sources.append(family_node_idx)
            hierarchy_targets.append(series_node_idx)

    hierarchy_edge_index = torch.tensor(
        [hierarchy_sources, hierarchy_targets], dtype=torch.long
    )

    return {"spatial": spatial_edge_index, "hierarchy": hierarchy_edge_index}
