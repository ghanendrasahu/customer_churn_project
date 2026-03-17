# Customer Churn Prediction — End-to-End Data Science Project

> **Interview-ready ML project** covering every step from raw data to deployed dashboard.

---

## Project Structure

```
customer-churn-prediction/
├── generate_dataset.py    # Creates a realistic synthetic Telco Churn CSV
├── churn_analysis.py      # Main pipeline: clean → EDA → features → models → explain
├── app.py                 # Streamlit deployment dashboard
├── requirements.txt       # Python dependencies
├── telco_churn.csv        # Generated dataset (after running generate_dataset.py)
├── model_artifacts.pkl    # Saved model + scaler (after running churn_analysis.py)
└── outputs/
    ├── 01_churn_overview.png
    ├── 02_numeric_distributions.png
    ├── 03_categorical_churn_rates.png
    ├── 04_correlation_heatmap.png
    ├── 05_violin_plots.png
    ├── 06_model_comparison.png
    ├── 07_roc_curves.png
    ├── 08_confusion_matrices.png
    ├── 09_feature_importance.png
    ├── 10_permutation_importance.png
    └── business_summary.txt
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate the dataset
python generate_dataset.py

# 3. Run full analysis (EDA + model training + explainability)
python churn_analysis.py

# 4. Launch the interactive Streamlit dashboard
streamlit run app.py
```

---

## What Each File Does

### `generate_dataset.py`
Creates a 7,043-row synthetic Telco dataset that mirrors the statistics and
structure of the Kaggle Telco Customer Churn dataset — including the notorious
whitespace-padded TotalCharges blanks that are a common interview discussion point.

### `churn_analysis.py` — The Main Pipeline

| Section | What happens |
|---------|-------------|
| **1 — Data Cleaning** | Load CSV, fix dtypes, handle TotalCharges nulls with logical imputation (tenure × monthly), remove dupes, inspect outliers |
| **2 — EDA** | Summary statistics, 5 publication-quality plots (churn distribution, numeric histograms, categorical bar charts, correlation heatmap, violin plots) |
| **3 — Feature Engineering** | Tenure groups, service count, cost-per-service ratio, long-term contract flag, unprotected-internet risk flag; binary encoding, one-hot encoding, StandardScaler |
| **4 — Model Training** | LogisticRegression, RandomForest, GradientBoosting (or XGBoost if installed); Stratified 5-fold CV + held-out test set; Accuracy/Precision/Recall/F1/ROC-AUC; Confusion matrices; Classification reports |
| **5 — Explainability** | Built-in feature importance or LR coefficients; Permutation importance (model-agnostic); SHAP summary plot (if shap installed) |
| **6 — Business Summary** | Dollar-value framing, key findings, recommended actions |

### `app.py` — Streamlit Dashboard
- **Prediction tab**: Input sliders/dropdowns → churn probability gauge + risk band + personalised retention recommendations
- **Insights tab**: Model benchmark chart, feature importance plot, all saved EDA plots
- **About tab**: Full methodology table + interview talking points

---

## Key Interview Talking Points

| Question | Answer |
|----------|--------|
| Why ROC-AUC over accuracy? | Class imbalance (77/23) makes accuracy misleading — a model predicting "no churn" always gets 77% |
| Why prioritise Recall? | A false negative (missed churner) is more expensive than a false positive (unnecessary retention offer) |
| Why permutation importance? | Tree impurity importance is biased toward high-cardinality features; permutation importance measures true predictive value |
| Why scale only for LR? | Tree models are scale-invariant; StandardScaler adds no value and can hurt interpretability |
| Why stratify the split? | Preserves class ratio across train/test — critical for imbalanced datasets |
| How to handle imbalance? | class_weight='balanced' in LR/RF; could also try SMOTE (imbalanced-learn) |
| Production path? | Retrain monthly, monitor for concept drift (PSI/KS test), A/B test retention campaigns |

---

## Adding XGBoost & SHAP

```bash
pip install xgboost shap
python churn_analysis.py   # automatically uses XGBoost + SHAP when detected
```

---

## Business Value

Churn prediction enables a **proactive retention strategy**:

1. Score every customer monthly
2. Flag top-N highest-risk customers
3. Route to retention specialists or trigger automated offers
4. Measure lift vs control group in an A/B test
5. Iterate on model + intervention design

A 1% improvement in churn rate on a 1M-customer base worth $50/month = **$6M/year** in protected revenue.
