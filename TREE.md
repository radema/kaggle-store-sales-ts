# Source Tree Structure

`src/` contains the core business logic, organized by functional module (Bolts).

## Directory Layout

```
.
├── .bolts/                  # Task-specific documentation (Specs, ADRs, MRPs)
│   ├── baseline-rearchitecture/ # Bolt #9: Gated Hybrid Baseline
│   └── data-processing-refactor/ # Legacy refactor artifacts
├── artifacts/               # Model outputs, metrics, and submissions (GitIgnored)
│   └── baseline/            # Current baseline OOFs and Submissions
├── configs/                 # YAML pipeline definitions
│   └── baseline_gated.yaml  # Main config for Gated Hybrid Model
├── data/
│   ├── raw/                 # Original Kaggle CSVs
│   ├── processed/           # Materialized Parquet files (train/test)
├── logs/                    # Timestamped operation logs
├── notebooks/
│   ├── reports/
│   │   └── 02-residual-analysis.ipynb # Main model analysis tool
│   └── 01-eda-raw-data.ipynb
├── scripts/
│   ├── preprocess.py        # Pipeline for feature building
│   └── train_baseline.py    # Iterative learning & submission runner
├── src/
│   ├── main.py              # Tool entry point (Legacy)
│   ├── data/                # Data loading utilities
│   ├── features/            # Feature Engineering Logic (Transformers)
│   ├── models/              # Model Architectures (HybridRegressor)
│   ├── pipeline/            # Component Orchestration & Runners
│   ├── utils/               # Shared utilities (Logging, Integrity)
│   └── validation/          # Validation Framework (EvaluationSuite)
├── tests/                   # Test suite (pytest)
└── TREE.md                  # This file
```

## Module Responsibilities

### `scripts/preprocess.py`
- **Goal**: Transform raw data into materialized Parquet features.
- **Behavior**: Deterministic, procedural logic for joins, holidays, and scaling.

### `scripts/train_baseline.py`
- **Goal**: Train and generate submissions for the Hybrid model.
- **Behavior**: Implements iterative inference (day-by-day) to handle time-series lags correctly.

### `src/models/hybrid.py`
- **Goal**: Implements `HybridRegressor` class.
- **Behavior**: Combines a Trend model (Linear) and Residual model (GBM) with deterministic gating support.

### `src/validation/harness.py`
- **Goal**: Standardized model evaluation.
- **Behavior**: Calculates RMSLE at global, store, and family levels; generates OOF prediction files.
