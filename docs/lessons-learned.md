Technical Post-Mortem: GNN Training on Apple Silicon (MPS)

Date: October 26, 2023
Topic: Accelerating Graph Neural Networks on M-Series Chips
Status: Architecture Pivot (RNN -> TCN)

Executive Summary

Training complex Graph Neural Networks (GNNs) on Apple's Metal Performance Shaders (MPS) requires a fundamental shift in architecture compared to NVIDIA (CUDA) workflows. Standard RNN-based approaches (LSTM/GRU) and sparse graph operations incur massive CPU-GPU synchronization overheads, rendering training impractically slow.

This document outlines the specific bottlenecks identified and the architectural patterns required to bypass them.

1. The "Sequential Killer": Why RNNs Fail on MPS

Observation:
A standard nn.GRU or nn.LSTM taking 10 minutes on a T4 GPU takes 4+ hours on an M1/M2/M3 Max chip.

Root Cause:

Kernel Dispatch Overhead: MPS has a higher fixed cost for launching a GPU kernel than CUDA.

Sequential Dependency: RNNs cannot be parallelized. A sequence length of $T=60$ requires 60 separate kernel launches per layer, per batch.

Lack of Fused Kernels: Unlike CUDA's cuDNN (which fuses the entire RNN loop into a single optimized kernel), PyTorch on MPS often executes the RNN loop in Python/C++ logic, dispatching ops one by one.

Solution: 1D Convolutions (TCN)

Strategy: Replace RNNs with Dilated 1D Convolutions (nn.Conv1d).

Benefit: A convolution processes the entire time sequence $T$ in a single kernel launch. This saturates the GPU shader cores and eliminates the dispatch bottleneck.

2. Sparse vs. Dense: The $N < 3000$ Rule

Observation:
Using PyTorch Geometric's GATv2Conv or SAGEConv (which rely on torch_scatter or sparse matrix multiplication) triggers "NotImplemented" warnings or silent CPU fallbacks.

Root Cause:

Sparse Support: MPS support for sparse tensors and scatter/gather operations is immature.

Memory Bandwidth: When an op falls back to CPU, data must be copied from VRAM -> RAM -> VRAM, killing throughput.

Solution: Force Dense Operations

Strategy: For graphs with $N < 3000$ nodes, strictly use Dense Matrix Multiplication (torch.mm or torch.einsum).

The Math: A dense $1782 \times 1782$ matmul is trivial for the M-series AMX (Matrix Coprocessor). It is significantly faster to compute zeros in a dense matrix than to pay the overhead of sparse indexing on this hardware.

3. The "Ghost Gradient" Bug

Observation:
A learnable adjacency matrix was implemented but showed no change in values over epochs.

Root Cause:

Detached Computation: The adjacency matrix was calculated but used to generate an edge_index (integer list).

Gradient Break: Passing integers (edge_index) breaks the backpropagation chain. Gradients cannot flow through discrete indices.

Solution: Differentiable Message Passing

Strategy: Pass the dense, continuous Adjacency Matrix ($A \in \mathbb{R}^{N \times N}$) directly to the GCN layer.

Implementation: $H^{(l+1)} = \sigma( A \cdot H^{(l)} W )$. This ensures $\frac{\partial Loss}{\partial A}$ is non-zero, allowing the network to learn the graph structure.

4. Architecture Recommendation: Graph WaveNet / MTGNN

For time-series forecasting on MPS, the optimal architecture is Graph WaveNet or MTGNN:

Ingestion: TCN (Dilated Conv) to extract temporal features.

Graph Learning: $A = \text{ReLU}(\text{tanh}(E_1 E_2^T))$.

Mixing: Dense GCN to mix spatial features using $A$.

Prediction: Direct projection to horizon (avoiding autoregressive decoding).

Summary Checklist for MPS GNNs

[ ] No RNNs: Use Conv1d or Transformers.

[ ] No Sparse Ops: Use Dense MatMul if nodes < 5000.

[ ] Contiguous Memory: Call .contiguous() before reshaping.

[ ] Profile: Watch for "Falling back to CPU" warnings in the console.