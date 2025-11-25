# Predicting Loan Payback 🏦 (Kaggle Playground Series)

![Python](https://img.shields.io/badge/python-3.11-blue.svg)
![Poetry](https://img.shields.io/badge/packaging-poetry-cyan.svg)
![Hydra](https://img.shields.io/badge/config-hydra-orange.svg)
![CatBoost](https://img.shields.io/badge/model-catboost-yellow.svg)

A complete MLOps project to predict loan repayment probability using a Data-Centric approach.
**Best Public Leaderboard Score: 0.92331 (Top Tier Performance)**

## 🎯 Project Objective
Predict whether a borrower will pay back their loan based on synthetic financial data (derived from real-world banking datasets).
The evaluation metric is **ROC AUC**.

## 🏆 Key Results & Strategy

This project moved away from "brute-force tuning" to a rigorous **Diagnostic-Driven approach**.

| Strategy | CV Score | Public LB | Notes |
|----------|----------|-----------|-------|
| **Baseline (XGBoost)** | 0.9216 | 0.9219 | Standard features. |
| **CatBoost (Tuned)** | 0.9223 | 0.9228 | Better handling of categorical features. |
| **Hill Climbing Ensemble** | **0.9230** | **0.9233** | **Optimization of OOF weights + Feature Eng V2.** |

### 💡Methodology
Using **SHAP Values** and **Residual Analysis**, I diagnosed that the model was systematically overestimating "good profiles" (High Credit Score + Low Debt Ratio).
* **Diagnosis:** A subset of "Low Debt" candidates were actually high-risk (Unemployed/Students), but the model missed this interaction.
* **Solution:** Engineering of targeted features like `flag_low_dti_risky_job` and `score_status_interactions`.
* **Outcome:** +0.0004 AUC gain on CatBoost solo model.

## 🛠 Architecture

The project follows a strict **MLOps structure** for reproducibility:

```text
├── configs/             # Hydra configuration files (Model hyperparameters)
├── data/                # Raw and Processed data (gitignored)
├── notebooks/           # EDA and SHAP Analysis notebooks
├── src/                 # Source code
│   ├── feature_engineering.py  # Domain-specific transformations
│   ├── training.py             # Main training pipeline (Hydra + MLflow)
│   ├── hill_climbing.py        # Advanced Ensembling script
│   └── diagnosis.py            # Model Debugging & Interpretation
├── pyproject.toml       # Poetry dependencies
└── README.md            # Project documentation


🚀 How to Run
1. Installation
Clone the repository and install dependencies using Poetry:
git clone https://github.com/aldrick77/predicting-loan-kaggle.git
cd predicting-loan-payback
poetry install

2. Data Setup
Place the competition data (train.csv, test.csv, sample_submission.csv) in the data/raw/ folder.

3. Training (Pipeline)
Train the different models to generate OOF (Out-Of-Fold) predictions:
# Train CatBoost 
poetry run python src/training.py model=catboost

# Train 
poetry run python src/training.py model=xgboost
poetry run python src/training.py model=lightgbm

4. Ensembling (Hill Climbing)
Run the optimization script to find the perfect mathematical blend:
poetry run python src/hill_climbing.py

