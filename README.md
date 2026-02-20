# Agentic Data Science: Kaggle Store Sales Time Series Forecasting

![Kaggle](https://img.shields.io/badge/Kaggle-Store_Sales-blue)
![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)
![Agentic Workflow](https://img.shields.io/badge/Workflow-Agentic-blueviolet)

This repository is an **experiment in using an autonomous Coding Agent system to execute an end-to-end Data Science and Machine Learning project**. 

The target problem is the popular Kaggle competition: [Store Sales - Time Series Forecasting](https://www.kaggle.com/c/store-sales-time-series-forecasting), which requires predicting the sales of thousands of product families across Corporación Favorita stores in Ecuador.

---

## 🤖 The Agentic Experiment

Unlike traditional data science workflows where a human writes every line of code, this project was architected, planned, and implemented by a large language model operating as an autonomous software engineer. 

The agent operates through a custom **"Bolt" architecture**—a structured framework for breaking down highly complex ML work into atomic, verifiable units of work:
1. **Strategic Intent**: Brainstorming features, defining model architectures, and creating detailed implementation specifications (`spec.md`).
2. **Execution Plans**: Outlining step-by-step TDD (Test-Driven Development) technical tasks (`plan.md`).
3. **Execution & Auditing**: Writing Python code, executing terminal commands to run the pipeline, parsing error logs, fixing bugs, and interpreting evaluation metrics (like RMSLE) independently.

### Key Capabilities Demonstrated by the Agent:
*   **Domain Feature Engineering:** Automatically deriving advanced features like 364-day exact historical lags, cyclic calendar coordinates, rolling geographic averages, and interaction flags (e.g., `is_wage_day_weekend`, `is_back_to_school`).
*   **Modular Pipeline Design:** Writing production-grade, object-oriented ML pipelines using `scikit-learn` BaseEstimator contracts rather than messy Jupyter notebooks.
*   **Performance Diagnostics:** Reading Cross-Validation output logs to identify the "worst-performing stores and families" (like Lingerie or School Supplies) and autonomously proposing new feature interventions to fix them.

---

## 🏗️ Project Architecture

The forecasting system leverages multiple mathematical paradigms:

### 1. The Gated Hybrid Baseline (Current Lead)
A two-stage model taking advantage of the strengths of both linear models and gradient-boosted trees:
*   **Trend Model (Ridge Regression):** Receives macro indicators (oil prices, year) and annual seasonality (364-day exact lags) to establish a baseline scale.
*   **Residual Model (LightGBM):** Learns the localized volatility and interaction effects (holidays, promotions, days of the week) on the residuals of the trend model.
*   **Hard Gating:** Rule-based heuristics to forcefully zero-out predictions on days when stores are statically known to be closed (e.g., New Year's Day or no recorded transactions).

### 2. The Spatio-Temporal GNN (Experimental)
An advanced architecture aiming to treat the Corporación Favorita store network as a graph structure to capture geographical and demographic spillover effects:
*   Graph Convolutions to mix features across nearby clusters and cities.
*   Temporal Convolutional Networks (TCNs) / WaveNets to capture local temporal patterns.

---

## 🛠️ Tech Stack

*   **Environment & Tooling:** `uv` (fast Python package installer and resolver), `pytest`.
*   **Data Processing:** `pandas`, `numpy`.
*   **Machine Learning:** `scikit-learn`, `lightgbm`.
*   **Deep Learning:** `torch`, `torch_geometric` (PyG).

---

## 🚀 Getting Started

To run the pipeline and replicate the agent's work locally:

1. **Install Dependencies (using `uv`):**
   ```bash
   uv sync
   ```

2. **Run Feature Engineering (Preprocessing):**
   ```bash
   PYTHONPATH=. uv run python scripts/preprocess.py
   ```

3. **Train the Hybrid Baseline Model & Generate Submission:**
   ```bash
   PYTHONPATH=. uv run python scripts/train_baseline.py --config configs/baseline_gated.yaml
   ```

Outputs such as the Out-of-Fold predictions (`oof_predictions.csv`) and the Kaggle submission file (`submission.csv`) will be cleanly generated in the `artifacts/baseline/` directory.

---
*This repository serves as a testament to the future of data science: where humans act as high-level strategic orchestrators, and AI agents handle the rigorous execution, debugging, and iterative refinement of machine learning systems.*
