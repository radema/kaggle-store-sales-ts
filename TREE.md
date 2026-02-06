# Source Tree Structure

`src/` contains the core business logic, organized by functional module (Bolts).

## Directory Layout

```
src/
└── validation/              # Bolt #1: Validation Framework
    ├── __init__.py          # Package initialization
    ├── splitters.py         # TimeSeries Cross-Validation strategies (SlidingWindowTS)
    ├── harness.py           # EvaluationSuite for reporting and plotting
    └── metrics.py           # Official competition metric (RMSLE) definitions
```

## Module Responsibilities

### `validation`
- **Goal**: Mimic the Kaggle evaluation process locally.
- **Key Classes**:
    - `SlidingWindowTS`: Generates train/validation indices for backtesting.
    - `EvaluationSuite`: Takes predictions, calculates errors granularly (per store/family), and produces summary reports/plots.
    - `rmsle`: Standalone competition metric function for quick checks.
