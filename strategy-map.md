# Strategic Roadmap: Store Sales Forecasting

## Project Objective
Predict store sales using a modular, scientifically rigorous approach, experimenting with both traditional Hybrid methods and modern Graph Neural Networks.

## Architectural Guidelines
*   **Pipeline**: `sklearn.pipeline` compatible for all tabular tasks.
*   **Validation**: Robust Cross-Validation with Time Series Split (Embargo/Purging) and RMSLE metric.
*   **Design**: Strategy Pattern (Transformers), Factory Pattern (Config Catalog).
*   **Simplicity**: Prefer existing libraries (scikit-learn) over custom over-engineering.

## The Bolt Roadmap

### Phase 1: Foundation & Hybrid Baseline
*   **Bolt #1: Validation Framework [COMPLETED]**
    *   Implement `TimeSeriesSplit` (custom/extended) and competition metric (RMSLE).
    *   Establish the ground truth evaluation harness.

*   **Bolt #2: Component - Data Pipeline Core [COMPLETED]**
    *   Base Data Loader.
    *   Abstract Feature Transformer (BaseEstimator).
    *   YAML Feature Catalog Parser (Factory).


*   **Bolt #3: Feature Engineering - Hybrid [COMPLETED]**
    *   Concrete Transformers: Lags, Rolling Windows, Date Parts.
    *   Integration with YAML catalog.

*   **Bolt #4: Model - Hybrid Regressor [COMPLETED]**
    *   `HybridRegressor` Class (sklearn-compatible).
    *   Logic: Linear Model (Trend) -> Residuals -> LightGBM (Seasonality/Interactions).

*   **Bolt #5: Experiment - Hybrid Baseline [COMPLETED]**
    *   Unified runner with YAML config.
    *   Integrated `time_idx`, `OrdinalEncoding`, and `HybridRegressor`.
    *   Automated artifact generation (Metrics, Model Card).

*   **Bolt #6: Reporting - Hybrid Analysis [COMPLETED]**
    *   Generate visualization report (Notebook/HTML).
    *   Plot Forecast vs Actuals, Residual Analysis, Error by Store/Family.

*   **Bolt #7: Leveling & Grouped Validation [COMPLETED]**
    *   Implement `LevelTransformer`: Static Target Encoding & Lagged Moving Averages (Axiom: must end at $t-16$).
    *   Fix K-Fold: Shift to `GroupedTimeSeriesSplit` (Store x Family groups) to ensure representative performance metrics.
    *   Axiom: The "Overall Average" level for a segment must be computed using a rolling anchor ending at $t-16$ relative to the prediction point.

*   **Bolt #8: Contextual Features & Trend Refinement [COMPLETED]**
    *   **Goal**: Integrate "World Context" and fix Trend/Level weighting issues.
    *   **Context Features**:
        *   **Oil**: Ingest `oil.csv` (fill missing, moving averages).
        *   **Store Metadata**: Merge `stores.csv` features (cluster, type, city, state).
        *   **Transactions**: Create `TransactionsLagTransformer`.
    *   **Trend Model Repair**: Investigate why `time_idx` vs `levels` weighting is suboptimal. (Consider scaling or interaction terms).
    *   **Analysis Upgrade**: Update notebook to drill down by Store Type, Cluster, and City to isolate skewed errors (e.g., Poultry/Meats).
    *   **Axiom**: Transaction counts are lagged-only.

*   **Bolt #9: Quality Assurance [TODO]**
    *   Integrate `ruff` for linting and pre-commits.

*   **Parallel Track: Error Volatility Investigation [TODO]**
    *   **High Volatility Families**: Deep dive into Lingerie, Liquor, and School Supplies.
    *   **High Volatility Segments**: Analyze specifically Cluster (14, 11, 6), Type (C), and outlier stores/cities.
    *   **July Seasonality**: Investigate the systemic high error volatility at the start of every July.

## Architectural Axioms (Knowledge Assets)
1. **Feature Partitioning**: Trend models use temporal features and Levels; Residual models use non-linear features (Seasonal Lags, Holiday IDs).
2. **Direct Forecasting**: Lags must be $\ge H$ (forecast horizon, usually 16 days) to maintain inference consistency without recursive loops.
3. **Leveling Axiom**: Any volume-based "Level" or "Average" feature must be computed using a window ending at $t-16$ relative to the prediction point to prevent look-ahead bias.

### Phase 2: Graph Neural Networks (GNN)
*   **Bolt #10: Data Prep - GNN [PARALLELABLE]**
    *   *Can start after Bolt #2, independent of Hybrid model.*
    *   Multivariate Tensor Fabrication: `(Batch, Time, Nodes, Features)`.
    *   Pre-computation of Correlation-based Adjacency Matrix (Learnable initialization).

*   **Bolt #11: Model - GNN [SEQUENTIAL]**
    *   Architecture: Graph WaveNet or GCN-LSTM.
    *   Core Feature: "Learnable Adjacency" layer.
    *   Training loop with Validation integration.

*   **Bolt #12: Reporting - GNN Analysis [SEQUENTIAL]**
    *   Visualize Learned Adjacency Matrix (Heatmap).
    *   Training/Validation Loss Curves.
    *   Forecast performance plots.

### Phase 3: Benchmark
*   **Bolt #13: Comparative Analysis [SEQUENTIAL]**
    *   *Requires Bolt #7 and #12 outputs.*
    *   Load artifacts from Hybrid and GNN.
    *   Comparative plots: Error distribution, "Better-than" analysis.
    *   Final conclusion on architecture suitability.

### Future Improvements (Backlog)
*   **Skill: `notebook-creator`**: Develop a specialized skill to programmatically generate standardized notebooks (EDA, Reports) using `nbformat`, ensuring consistent imports and settings.

## Execution Graph

```mermaid
graph TD
    subgraph Foundation
        B1[Bolt #1] --> B2[Bolt #2]
    end

    subgraph Hybrid Stream
        B2 --> B3[Bolt #3]
        B3 --> B4[Bolt #4]
        B4 --> B5[Bolt #5]
        B5 --> B6[Bolt #6]
        B6 --> B7[Bolt #7: Leveling]
        B7 --> B8[Bolt #8: Context]
        B8 --> B9[Bolt #9: Standardization]
    end

    subgraph GNN Stream
        B2 --> B10[Bolt #10]
        B10 --> B11[Bolt #11]
        B11 --> B12[Bolt #12]
    end

    subgraph Conclusion
        B9 --> B13[Bolt #13]
        B12 --> B13
    end

    style B1 fill:#f9f,stroke:#333
    style B2 fill:#f9f,stroke:#333
    style B10 fill:#ccf,stroke:#333
```
