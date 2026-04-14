# Credit Card Fraud Detection using Ensemble Learning

## Overview

This project builds a machine learning pipeline to detect fraudulent credit card transactions using ensemble learning techniques. The dataset is highly imbalanced (only 0.17% fraud), making it a challenging real-world classification problem.

## Dataset

- **Source:** [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size:** 284,807 transactions, 31 columns
- **Features:** Time, V1–V28 (PCA-transformed), Amount, Class (target)
- **Imbalance:** 284,315 normal vs 492 fraud transactions

Place the dataset at `archive/creditcard.csv`.

## Project Pipeline

1. **Data Loading & Exploration** — Understand distributions, correlations, and imbalance
2. **Preprocessing** — Handle missing values, scale features, train-test split
3. **Imbalance Handling** — SMOTE (oversampling) and Random Undersampling
4. **Model Training** — Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Voting Classifier
5. **Evaluation** — Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrices
6. **Visualization** — ROC curves, feature importance, threshold tuning
7. **Prediction** — Demonstrate prediction on new samples

## Models Compared


| Model               | Type                |
| ------------------- | ------------------- |
| Logistic Regression | Base (Linear)       |
| Decision Tree       | Base (Non-linear)   |
| Random Forest       | Ensemble (Bagging)  |
| Gradient Boosting   | Ensemble (Boosting) |
| Voting Classifier   | Ensemble (Voting)   |


## Setup & Running

```bash
pip install -r requirements.txt
```

### Option 1: Jupyter Notebook (Recommended for learning/lab submission)

```bash
cd notebook
jupyter notebook credit_card_fraud_detection.ipynb
```

The notebook includes rich markdown explanations, inline visualizations, and step-by-step narrative — ideal for academic review. Output plots are saved to `notebook/output/`.

### Option 2: Python Scripts (Modular structure)

```bash
cd scripts
python main.py
```

All `.py` files live inside the `scripts/` folder. Running `main.py` saves all output plots to `scripts/output/`.


| File                       | Purpose                                                                       |
| -------------------------- | ----------------------------------------------------------------------------- |
| `scripts/config.py`        | Constants, paths, random seeds, plot settings                                 |
| `scripts/data_loader.py`   | Dataset loading and class distribution                                        |
| `scripts/eda.py`           | Exploratory Data Analysis — all visualizations                                |
| `scripts/preprocessing.py` | Scaling, train-test split, SMOTE, undersampling                               |
| `scripts/models.py`        | Model definitions and training loop                                           |
| `scripts/evaluation.py`    | Metrics, confusion matrices, ROC curves, feature importance, threshold tuning |
| `scripts/main.py`          | Main pipeline — orchestrates everything end-to-end                            |
| `scripts/output/`          | Auto-created folder where all output PNGs are saved                           |


Both approaches produce identical results.

## Key Takeaways

- Accuracy alone is misleading with imbalanced data — Recall and F1-Score matter more
- SMOTE effectively addresses class imbalance without losing majority-class data
- Ensemble models (Random Forest, Gradient Boosting) consistently outperform base models
- Threshold tuning allows fine-grained control over the precision-recall tradeoff

## Output Visualizations

Each approach saves output plots into its own `output/` subfolder (`notebook/output/` or `scripts/output/`):

- `class_distribution.png` — Class imbalance visualization
- `amount_analysis.png` — Transaction amount comparison
- `time_analysis.png` — Temporal patterns
- `correlation_analysis.png` — Feature correlations with fraud
- `feature_comparison.png` — Feature distributions for fraud vs normal
- `resampling_comparison.png` — Effect of SMOTE and undersampling
- `confusion_matrices.png` — All model confusion matrices
- `roc_curves.png` — ROC curve comparison
- `feature_importance.png` — Top features from tree-based models
- `threshold_tuning.png` — Precision-recall tradeoff analysis
- `model_comparison.png` — Bar chart comparing all metrics

## Project Structure

```
credit card fraud detection/
├── archive/
│   └── creditcard.csv                      # Dataset
├── notebook/                               # Jupyter Notebook
│   ├── credit_card_fraud_detection.ipynb   # Run this
│   └── output/                             # Auto-created on run
│       ├── class_distribution.png
│       ├── ...
│       └── model_comparison.png
├── scripts/                                # Modular Python files
│   ├── config.py
│   ├── data_loader.py
│   ├── eda.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── main.py                             # Run this
│   └── output/                             # Auto-created on run
│       ├── class_distribution.png
│       ├── ...
│       └── model_comparison.png
├── requirements.txt
└── README.md
```

