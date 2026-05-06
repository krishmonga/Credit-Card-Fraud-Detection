# Credit Card Fraud Detection — Project Report

Last updated: 2026-05-06

## Project Overview

An ensemble learning pipeline for detecting fraudulent credit card transactions using scikit-learn, addressing severe class imbalance through SMOTE and undersampling techniques.

**Dataset:** Kaggle Credit Card Fraud Dataset  
**Size:** 284,807 transactions, 31 features  
**Class Distribution:** 226,602 normal | 378 fraud (0.13% fraud)

## Models Trained

1. **Logistic Regression** — Linear baseline
2. **Decision Tree** — Single tree classifier
3. **Random Forest** — Bagging ensemble
4. **Gradient Boosting** — Sequential boosting
5. **Voting Classifier** — Soft voting ensemble

## Pipeline Components

| Stage | Methods |
|-------|---------|
| **Data Loading** | CSV parsing, shape validation |
| **EDA** | Distribution analysis, correlation study |
| **Preprocessing** | StandardScaler, 80-20 train-test split |
| **Resampling** | SMOTE (oversampling), Random Undersampling |
| **Training** | Fit on SMOTE-resampled data |
| **Evaluation** | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Visualization** | Confusion matrices, ROC curves, feature importance |

## Output & Usage

### Run ML Pipeline
```bash
cd scripts
python main.py
```
- Outputs: Visualizations in `scripts/output/`, models saved as `.pkl`

### Interactive Dashboard
```bash
streamlit run app.py
```
- Explore EDA, model performance, live predictions

## Key Findings

- **Class imbalance** severely impacts accuracy metrics
- **SMOTE + Undersampling** effectively balances dataset without data loss
- **Ensemble methods** outperform single classifiers
- **Precision-Recall tradeoff** tunable via threshold adjustment

## Files Generated

- `scripts/output/*.png` — Visualizations (ROC, confusion matrices, features)
- `scripts/output/best_model.pkl` — Serialized best-performing model
- `scripts/output/scaler.pkl` — StandardScaler for inference

---
_Report note:_ Updated project report formatting and usage on 2026-05-06.

