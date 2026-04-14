# Run Report — Credit Card Fraud Detection

Run date: 2026-04-14

## Summary
- Dataset: Kaggle Credit Card Fraud (284,807 rows, 31 cols)
- Imbalance: 284,315 normal vs 492 fraud (≈0.17% fraud)
- Pipeline executed end-to-end via `scripts/main.py`.

## Key Results
- Trained models: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Voting Classifier
- Best overall model: Random Forest
  - Accuracy: 0.9994
  - Precision: 0.8315
  - Recall: 0.7789
  - F1-Score: 0.8043
  - ROC-AUC: 0.9672
- Gradient Boosting achieved highest ROC-AUC: 0.9795

## Files produced
- Models & scaler saved to: `scripts/output/` (includes serialized model and scaler)
- Visualizations (PNGs) saved to: `scripts/output/` (ROC curves, confusion matrices, feature importance, etc.)

## How to reproduce
```bash
pip install -r requirements.txt
python scripts/main.py
```

## Notes & Next Steps
- Consider removing heavy dev-only packages from `requirements.txt` for production (e.g., full `jupyter` meta-package).
- Add unit tests for data preprocessing and model inference.
- Convert best model into a lightweight inference endpoint (FastAPI or Streamlit) for demo.

