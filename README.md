## Current Performance
| Model | CV Score (AUC) | Public LB Score | Notes |
|-------|----------------|-----------------|-------|
| XGBoost Baseline | 0.92163 | 0.92198 | Basic features + Label Encoding |
| CatBoost | 0.92233 | 0.92279 | Native categorical support |
| **Ensemble (Blend)** | **N/A** | **0.92288** | **Weighted Avg (35% XGB, 65% Cat)** |