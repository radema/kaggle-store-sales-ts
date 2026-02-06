# Source Tree Structure

`src/` contains the core business logic, organized by functional module (Bolts).

## Directory Layout

```
src/
├── data/                    # Bolt #2: Data Pipeline Core
│   ├── __init__.py
│   └── loader.py            # DataLoader with schema/date handling
├── features/                # Feature Engineering Logic
│   ├── __init__.py
│   ├── base.py              # BaseTimeSeriesTransformer (Leakage Checks)
│   └── impute.py            # TimeSeriesImputer (ffill, bfill, interpolate)
├── pipeline/                # Component Orchestration
│   ├── __init__.py
│   └── factory.py           # FeaturePipelineFactory (YAML -> Sklearn Pipeline)
└── validation/              # Bolt #1: Validation Framework
    ├── __init__.py
    ├── splitters.py         # TimeSeries Cross-Validation
    ├── harness.py           # EvaluationSuite
    └── metrics.py           # RMSLE
```

## Module Responsibilities

### `data`
- **Goal**:Agostic data ingestion.
- **Key Classes**: `DataLoader` (Unified access to raw CSVs).

### `features`
- **Goal**: Extensible feature transformations.
- **Key Classes**: 
    - `BaseTimeSeriesTransformer`: Scikit-learn compatible base with leakage hooks.
    - `TimeSeriesImputer`: Configurable filling strategies.

### `pipeline`
- **Goal**: Decouple logic from configuration.
- **Key Classes**: `FeaturePipelineFactory` (Dynamic instantiation from YAML).

### `validation`
- **Goal**: Mimic the Kaggle evaluation process locally.

