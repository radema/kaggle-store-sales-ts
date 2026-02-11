# Source Tree Structure

`src/` contains the core business logic, organized by functional module (Bolts).

## Directory Layout

```
.
├── .bolts/                  # Task-specific documentation (Specs, ADRs, MRPs)
│   ├── hybrid-baseline/     # Bolt #5: Hybrid experiment artifacts
│   └── data-processing-pipeline/ # Bolt #8: Preprocessing pipeline artifacts
├── artifacts/               # Model outputs, metrics, and submissions (GitIgnored)
│   └── runs/                # Timestamped run outputs
├── configs/                 # YAML pipeline definitions
│   ├── hybrid_baseline.yaml # Main config for Hybrid Model
│   └── preprocessing.yaml   # Config for Bolt #8 Data Pipeline
├── data/
│   ├── raw/                 # Original Kaggle CSVs
│   └── processed/           # Materialized Parquet files (train/test)
├── logs/                    # Timestamped operation logs
├── src/
│   ├── main.py              # CLI Entry point for all runners
│   ├── data/                # Bolt #2/8: Data Pipeline Core
│   │   ├── __init__.py
│   │   ├── loader.py        # DataLoader with unified loading
│   │   └── make_dataset.py  # Data processing pipeline runner
│   ├── features/            # Feature Engineering Logic
│   │   ├── __init__.py
│   │   ├── base.py          # BaseTimeSeriesTransformer (Logging & Validation)
│   │   ├── alignment.py     # DateGridTransformer (Optional gap filling)
│   │   ├── dates.py         # DatePartTransformer
│   │   ├── encoding.py      # CategoricalEncoder
│   │   ├── imputation.py    # ConfigurableImputer (Intentional handling)
│   │   ├── impute.py        # Standard TimeSeriesImputer
│   │   ├── lags.py          # LagTransformer
│   │   ├── meta.py          # OilMerger, StoreMerger
│   │   ├── rolling.py       # RollingWindowTransformer
│   │   └── transactions.py  # TransactionMerger (Validation & Imputation)
│   ├── pipeline/            # Component Orchestration
│   │   ├── __init__.py
│   │   ├── factory.py       # Scikit-learn Pipeline Factory
│   │   ├── processing.py    # ProcessingPipelineFactory (Bolt #8)
│   │   └── runners/         # Concrete experiment strategies
│   ├── models/              # Bolt #4: Model Architectures
│   │   ├── __init__.py
│   │   └── hybrid.py        # HybridRegressor (Trend + Residuals)
│   ├── utils/               # Shared utilities
│   │   ├── logging.py       # Standardized session logging
│   │   └── validation.py    # DataFrame sanity checks (Duplicates, Schema)
│   └── validation/          # Bolt #1: Validation Framework
├── tests/                   # Test suite (pytest)
├── run_processing.py        # Root runner for data transformation
└── TREE.md                  # This file
```

## Module Responsibilities

### `data/make_dataset.py`
- **Goal**: Transform raw data into a materialized feature store (Parquet).
- **Behavior**: Config-driven, unified loading of train/test.

### `features/base.py`
- **Goal**: Standardize transformer interfaces with logging and validation hooks.
- **Hook**: `_transform` (Implementation) vs `transform` (Boilerplate).

### `utils`
- **logging.py**: Unique session logs per run with timestamps.
- **validation.py**: Defensive checks for data integrity (nulls, duplicates, shape).
