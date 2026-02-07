
# Store Sales - Time Series Forecasting

## Context
This repository contains the solution for the Kaggle competition: [Store Sales - Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting).

**Goal**: Predict sales for thousands of product families at Corporación Favorita stores in Ecuador.
**Metric**: Root Mean Squared Logarithmic Error (RMSLE).

## Architecture
The project follows a modular "Bolt" architecture (managed by `bolt-roadmap`):
1.  **Validation Framework**: Robust cross-validation strategy.
2.  **Data Pipeline**: ETL and Feature Engineering.
3.  **Hybrid Model**: Linear Trend + LightGBM Residuals.
4.  **GNN Model**: Graph Neural Network experiment.

## Current Phase: Bolt #3 (Feature Engineering)
We are currently building the feature engineering pipeline, focusing on:
-   **Date Features**: Calendar parts, Ecuador wage days.
-   **Lags & Rolling Windows**: Multi-series aware (Store x Family).
-   **Holidays**: Local, Regional, and National holiday handling.
