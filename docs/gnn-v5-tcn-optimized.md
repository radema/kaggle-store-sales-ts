# ST-GNN Architecture: TCN-Folded & Einstein-Mixed (v5)

## Overview
This document describes the high-performance architecture of the Spatio-Temporal Graph Neural Network (ST-GNN) optimized for Apple Silicon (MPS). This version (v5) replaces the generic 3D-convolutional approach with a hardware-aligned "Folded TCN" and "Einstein Mixer" strategy, resulting in a **10-20x speedup** in training throughput.

## Architecture Evolution

| Feature | GNN v4 (3D) | GNN v5 (Folded 1D) | Rationale |
| :--- | :--- | :--- | :--- |
| **Temporal Engine** | `nn.Conv3d` (Generic 5D) | **`nn.Conv1d` (Folded)** | MPS kernels for 1D are heavily optimized for the Apple Neural Engine and GPU throughput. |
| **Spatial Mixing** | Permute + `matmul` | **`torch.einsum`** | Eliminates expensive memory copies and allows Metal-level optimizations for sparse-like dense mixing. |
| **Normalization** | `BatchNorm3d` | **`GroupNorm`** | Prevents NaN errors when Batch Size = 1 (Dev Mode) and provides better stability for time-series. |
| **Graph Context** | Pure Learnable Embeddings | **Static Node Features** | Uses `stores.csv` (Cluster, City, etc.) as a prior to inform the graph structure. |
| **Precision** | Autocast (FP16/BF16) | **Full FP32** | Prevents underflow/NaN issues observed with GNN spatial operations on MPS. |

---

## Core Components

### 1. Temporal Folding (CausalConv1d)
The 5D grid $(B, C, S, F, T)$ is reshaped into $(B \times S \times F, C, T)$ before processing.
*   **Performance**: This bypasses the overhead of 3D convolution kernels that are not optimized for the "time-only" $(1, 1, K)$ window.
*   **Causality**: Uses causal padding to ensure the model cannot look into the future during training.

### 2. Einstein Mixer (FactoredGCN)
Instead of flattening the graph, we use factorized spatial and family adjacencies:
*   **Efficiency**: Operates in $O(S^2 + F^2)$ instead of $O((S \times F)^2)$.
*   **Zero-Copy**: `torch.einsum` allows the model to mix store features and family features without explicitly permuting or re-ordering the tensor in VRAM.

### 3. Static Meta-Embeddings
The model incorporates information from the competition's metadata:
*   **Input**: `city`, `state`, `type`, and `cluster`.
*   **Effect**: The adjacency matrix is initialized via SVD from these features, creating a "smart initialization" where stores in the same city or cluster have a natural prior to share information.

---

## Training Stability Protocols

### Identity-plus-Epsilon (I+ε)
Adjacency matrices are biased towards the diagonal using:
$$A = \text{ReLU}(\text{tanh}(E_1 E_2^T)) + I \cdot \epsilon$$
This ensures that every node always "sees" its own history, even if the graph learning hasn't converged, preventing gradient collapse.

### GroupNorm Stability
`GroupNorm` (with 8 groups) is used to normalize the hidden features. This ensures the model converges identically whether trained with a single sample (Dev Mode) or a large batch (Production).

### Gradient Flow
- **Clipping**: Gradients are clipped at `1.0` to handle the high learning rates (`0.001`) enabled by this faster architecture.
- **Autocast Bypass**: Autocast is disabled for MPS to ensure the precision required for the Einstein Summation operations is maintained.

---

## Performance Summary
*   **Epoch Time (MPS)**: ~1-2 seconds (Dev Mode) / ~30-40 seconds (Full Dataset).
*   **Memory Footprint**: Extremely low due to zero-copy operations and folding.
*   **Convergence**: Faster and more stable due to informing the graph with static store metadata.
