# GNN Architecture v6: Decomposition with Gated Residuals

## Overview
This document proposes an architectural evolution for the SaleGNN model to explicitly mimic the successful "Hybrid Model" strategy (Trend + Residuals) within an end-to-end differentiable neural network.

The core hypothesis is that the current GNN struggles to learn the basic trend/seasonality from scratch while simultaneously learning complex interactions. By explicitly separating these concerns, we can stabilize training and improve convergence.

## The Architecture: "Decomposition with Gated Residuals"

We propose splitting the model into two parallel paths: a **Trend Component** (Linear) and a **Residual Component** (The ST-GNN), combined via a learnable **Gated Linear Unit (GLU)**.

### Conceptual Diagram

```mermaid
graph LR
    Input[Input Window] --> TrendPath[Trend Path <br/> (Linear Layer)]
    Input --> ResidualPath[Residual Path <br/> (ST-GNN Layers)]
    
    ResidualPath --> Gate[Gated Linear Unit <br/> (GLU)]
    
    TrendPath --> Add((+))
    Gate --> Add
    
    Add --> Output
```

### Component Details

#### 1. The Trend Path (Linear Baseline)
A simple, interpretable linear layer that projects the input window directly to the forecast horizon.
- **Input**: Raw sales history window $(B, N, T_{in})$.
- **Operation**: `nn.Linear(window_size, horizon)`.
- **Purpose**: Captures the "Easy" signal—moving averages, general direction, and basic seasonality.
- **Benefit**: Provides a stable gradient immediately at Epoch 0. The model starts with a "reasonable" guess (regression) rather than random noise.

#### 2. The Residual Path (ST-GNN)
The existing Spatio-Temporal Graph Neural Network.
- **Input**: The full feature set (Sales, Calendar, static features, etc.).
- **Operation**: `STGNNBlock` stack $\to$ `FC Head`.
- **Purpose**: Focuses entirely on the *residuals*—the difference between the linear trend and the ground truth. It learns complex interactions (e.g., "Sales spike on holidays for this specific cluster").

#### 3. The Gate (Gated Linear Unit)
A mechanism to dynamically suppress or amplify the GNN's contribution.
- **Operation**: $Output = GNN(x) \otimes \sigma(Gate(x))$.
- **Purpose**: Acts as a noise filter. If the GNN is uncertain or producing noise (e.g., on a regular Tuesday with no events), the gate closes, and the model falls back to the robust Linear Trend.

## Expected Benefits
1.  **Faster Convergence**: The Linear layer learns the mean/trend almost instantly.
2.  **Stability**: The GNN doesn't waste capacity relearning basic autoregressive patterns.
3.  **Interpretability**: We can inspect the weights of the Linear layer to baseline behavior.
4.  **Hybrid Parity**: This explicitly aligns the NN architecture with the `Ridge + LightGBM` hybrid strategy that is currently the state-of-the-art in the project.
