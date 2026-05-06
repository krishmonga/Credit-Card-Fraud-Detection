# Credit Card Fraud Detection using Ensemble Learning

## Overview

A machine learning pipeline to detect fraudulent credit card transactions using ensemble learning techniques. The dataset is highly imbalanced (0.17% fraud), making it a challenging real-world classification problem.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 features
- **Class Imbalance:** 226,602 normal vs 378 fraud transactions

## Models Used

| Model               | Type                |
| ------------------- | ------------------- |
| Logistic Regression | Linear Classifier   |
| Decision Tree       | Tree-based          |
| Random Forest       | Ensemble (Bagging)  |
| Gradient Boosting   | Ensemble (Boosting) |
| Voting Classifier   | Ensemble (Voting)   |

## Quick Start

```bash
pip install -r requirements.txt
```

### Run ML Pipeline (Scripts)

```bash
cd scripts
python main.py
```

All visualizations are saved to `scripts/output/`.

### Interactive Dashboard (Streamlit)

```bash
streamlit run app.py
```

Launch an interactive web interface to explore EDA, model performance, and test fraud predictions.

## Pipeline Steps

1. **Data Loading** — Load and inspect Kaggle dataset
2. **EDA** — Class distribution, correlations, feature analysis
3. **Preprocessing** — Scaling, train-test split
4. **Resampling** — SMOTE and random undersampling to handle imbalance
5. **Model Training** — Train 5 ensemble and base models
6. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC
7. **Visualization** — Confusion matrices, ROC curves, feature importance
8. **Predictions** — Demo predictions on sample transactions

## Project Structure

```
Credit-Card-Fraud-Detection/
├── scripts/
│   ├── main.py              # Run ML pipeline
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── output/              # Auto-created visualizations
├── app.py                   # Streamlit dashboard
├── requirements.txt
├── README.md
└── REPORT.md
```

## Key Insights

- **Accuracy alone is misleading** — Precision & Recall matter more for fraud detection
- **SMOTE effectively handles class imbalance** without data loss
- **Ensemble models outperform base models** consistently
- **Threshold tuning enables precision-recall tradeoffs**

