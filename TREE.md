# Source Tree Structure

`src/` contains the core business logic, organized by functional module (Bolts).

## Directory Layout

```
.
├── .bolts/                  # Task-specific documentation (Specs, ADRs, MRPs)
│   └── hybrid-baseline/     # Bolt #5: Hybrid experiment artifacts
├── artifacts/               # Model outputs, metrics, and submissions (GitIgnored)
│   └── runs/                # Timestamped run outputs
├── configs/                 # YAML pipeline definitions
│   └── hybrid_baseline.yaml # Main config for Hybrid Model
├── src/
│   ├── main.py              # CLI Entry point for all runners
│   ├── data/                # Bolt #2: Data Pipeline Core
│   │   ├── __init__.py
│   │   └── loader.py        # DataLoader with schema/date handling
│   ├── features/            # Feature Engineering Logic
│   │   ├── __init__.py
│   │   ├── base.py          # BaseTimeSeriesTransformer (Leakage Checks)
│   │   ├── encoding.py          # CategoricalEncoder (Ordinal encoding)
│   │   ├── impute.py        # TimeSeriesImputer (ffill, bfill, interpolate)
│   │   ├── dates.py         # DatePartTransformer (Year, Month, Day, Weekday)
│   │   ├── lags.py          # LagTransformer (Shifted features)
│   │   └── rolling.py       # RollingWindowTransformer (Mean, Std, Min, Max)
│   ├── pipeline/            # Component Orchestration (Bolt #5)
│   │   ├── __init__.py
│   │   ├── factory.py       # FeaturePipelineFactory
│   │   ├── runners/         # Concrete experiment strategies
│   │   │   └── hybrid.py    # HybridRunner (Load -> CV -> Train -> Predict)
│   │   ├── config.py        # YAML loader utilities
│   │   └── documentation.py # ModelCardGenerator
│   ├── models/              # Bolt #4: Model Architectures
│   │   ├── __init__.py
│   │   └── hybrid.py        # HybridRegressor (Trend + Residuals)
│   └── validation/          # Bolt #1: Validation Framework
│       ├── __init__.py
│       ├── splitters.py     # TimeSeries Cross-Validation
│       ├── harness.py       # EvaluationSuite
│       └── metrics.py       # RMSLE
└── tests/                   # Test suite (pytest)
```

## Module Responsibilities

### `pipeline/runners`
- **Goal**: End-to-end experiment orchestration.
- **Key Classes**: `HybridRunner`.

### `pipeline`
- **Goal**: Decouple logic from configuration and document models.
- **Key Classes**: `FeaturePipelineFactory`, `ModelCardGenerator`.

### `main.py`
- **Goal**: Standard interface for running any experiment via CLI.

### `models`
- **Goal**: Modular ML model implementations.
- **Key Classes**: `HybridRegressor` (Trend + Residual additive model).
