# ST-GNN Architecture & Pipeline Documentation

This document outlines the Spatio-Temporal Graph Neural Network (ST-GNN) system implemented for the Store Sales forecasting challenge.

## 1. Scripts Dependency & Architecture

The GNN module is designed to be decoupled from the standard tabular pipelines while sharing the same processed data sources.

### Core Workflow
1.  **Preprocessing (`preprocess.py`)**: Generates the standard `train.parquet` and `test.parquet`.
2.  **Dataset Building (`scripts/build_gnn_dataset.py`)**: 
    - Reads the tabular parquet files.
    - Pivots data into **3D Tensors** (Nodes × Time × Features).
    - Saves binary snapshots into `data/processed/gnn/` for memory-efficient loading via `mmap`.
3.  **Training (`scripts/train_gnn.py`)**:
    - Discovers/Loads the binary cache.
    - Constructs the graph adjacency using `src/models/gnn/graph.py`.
    - Orchestrates the training loop via `GNNTrainer`.
    - Outputs model weights, logs, and a `submission.csv` to `artifacts/gnn/`.

### Directory Structure
- `src/models/gnn/`: Model, Dataset, and Trainer components.
- `scripts/`: Entry points for caching and training.
- `configs/gnn.yaml`: Hyperparameters and feature definitions.

## 2. Chosen GNN Architecture: Seq2Seq ST-GNN

The model is a **Sequence-to-Sequence Spatio-Temporal Graph Neural Network**. Unlike standard regressions, it processes all entities as nodes in a graph to capture cross-series correlations.

### Key Components
- **Learnable Node Embeddings**: Every node (Store, Family, Series) has a unique 16-dimensional identity vector. This allows the model to learn individual biases (e.g., "Store 1 is high volume").
- **Temporal Encoder (GRU)**: Processes the 30-day history per node to extract dynamic temporal states.
- **Spatial Mixer (GATv2)**: Uses Graph Attention (GATv2) to allow nodes to "share" information with neighbors (e.g., a "Bakery" node learns from other "Bakery" nodes).
- **Seq2Seq Decoder**: Uses a GRUCell to predict 16 days ahead, injecting future known features (promotions, oil prices, calendar) at each step.
- **Hard Gating**: A final custom layer that forces predictions to zero if the `is_closed` mask is active for a specific store/day.

## 3. Chosen Graph Model

We use a **Heterogeneous-to-Homogeneous Unified Graph**. To simplify computation, we represent all entities as nodes in a shared index.

### Node Indexing
1.  **Store Nodes (0-53)**: Represent the 54 physical stores.
2.  **Family Nodes (54-86)**: Represent the 33 product categories.
3.  **Series Nodes (87-1781)**: Represent the unique Store-Family combinations (e.g., Store 1 - Bakery).

### Edge Connections (Adjacency)
The graph is built using three logic layers:
1.  **Spatial Edges**: Connect Store nodes that are in the same **City**. This captures regional economic trends.
2.  **Hierarchical Edges**: Connect each "Series Node" to its parent "Store Node" and parent "Family Node". This allows a specific product's forecast to be influenced by the store's overall health and the category's global trend.
3.  **Learnable Adjacency**: A dense, trainable matrix with **L1 Regularization** that allows the model to discover "hidden" relationships between series that we didn't explicitly define.

### Feature Set
We use the same high-signal features as the Hybrid model to ensure parity:
- **Exogenous**: Oil Price, Promotions.
- **Calendar**: Cyclical encodings, Paydays, Holidays.
- **Recursive**: Lags (7, 14, 21, 28) and Rolling 30-day means.
