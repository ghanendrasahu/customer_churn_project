"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          CUSTOMER CHURN PREDICTION — END-TO-END DATA SCIENCE PROJECT        ║
║                    Telco Customer Churn (Kaggle-style dataset)               ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Sections:                                                                   ║
║   1. Data Loading & Cleaning                                                 ║
║   2. Exploratory Data Analysis (EDA)                                         ║
║   3. Feature Engineering                                                     ║
║   4. Model Training & Evaluation                                             ║
║   5. Explainability (Feature Importance + SHAP when available)               ║
║   6. Business Summary                                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

Interview talking points appear as  # INTERVIEW NOTE  comments throughout.
Run:  python churn_analysis.py
Outputs are saved to ./outputs/
"""

# Standard imports
import os, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import seaborn as sns

from sklearn.model_selection   import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing     import StandardScaler
from sklearn.linear_model      import LogisticRegression
from sklearn.ensemble          import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics           import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
    ConfusionMatrixDisplay
)
from sklearn.inspection        import permutation_importance

warnings.filterwarnings("ignore")

# Optional: XGBoost & SHAP  (pip install xgboost shap)
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("INFO: XGBoost not found — using GradientBoostingClassifier instead.\n")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("INFO: SHAP not found — using permutation importance for explainability.\n")

OUTPUT_DIR = "./outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==============================================================================
# SECTION 1  —  DATA LOADING & CLEANING
# ==============================================================================
print("=" * 68)
print("SECTION 1  —  DATA LOADING & CLEANING")
print("=" * 68)

# --- 1.1  Load raw CSV --------------------------------------------------------
df = pd.read_csv("../data/telco_churn.csv")
print(f"\nRaw shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
print("\nColumn dtypes:")
print(df.dtypes.to_string())

# --- 1.2  Remove duplicate rows -----------------------------------------------
# INTERVIEW NOTE: always check for dupes before anything else.
n_dupes = df.duplicated().sum()
df.drop_duplicates(inplace=True)
print(f"\nDuplicates removed: {n_dupes}")

# --- 1.3  Fix TotalCharges (whitespace-as-null pattern in the Kaggle CSV) -----
# INTERVIEW NOTE: pd.to_numeric(errors='coerce') is the idiomatic fix — it
# converts non-numeric strings to NaN without raising an exception.
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
n_null_tc = df["TotalCharges"].isna().sum()
print(f"TotalCharges nulls coerced: {n_null_tc}")

# Logical imputation: a brand-new customer's total = tenure * monthly
mask_tc = df["TotalCharges"].isna()
df.loc[mask_tc, "TotalCharges"] = (
    df.loc[mask_tc, "tenure"] * df.loc[mask_tc, "MonthlyCharges"]
)
print("  Imputed with tenure x MonthlyCharges.")

# --- 1.4  Drop identifier column ---------------------------------------------
df.drop(columns=["customerID"], inplace=True)

# --- 1.5  Encode binary target -----------------------------------------------
df["Churn"] = (df["Churn"] == "Yes").astype(int)
print(f"\nTarget distribution  (0=No Churn, 1=Churn):")
print(df["Churn"].value_counts(normalize=True).round(3).to_string())

# --- 1.6  Outlier inspection (IQR) — informational only ----------------------
# INTERVIEW NOTE: In churn modelling, outliers in tenure / charges are almost
# always legitimate signals, not data errors.  We log them but keep them.
num_cols_raw = ["tenure", "MonthlyCharges", "TotalCharges"]
print("\nOutlier check (IQR method) — all rows retained:")
for col in num_cols_raw:
    Q1, Q3 = df[col].quantile([0.25, 0.75])
    IQR = Q3 - Q1
    n_out = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    print(f"  {col:20s}: {n_out:4d} flagged — kept (valid business data)")

print("\n[OK] Data cleaning done.")


# ==============================================================================
# SECTION 2  —  EXPLORATORY DATA ANALYSIS
# ==============================================================================
print("\n" + "=" * 68)
print("SECTION 2  —  EXPLORATORY DATA ANALYSIS")
print("=" * 68)

print("\nDescriptive stats (numeric):")
print(df[num_cols_raw + ["Churn"]].describe().round(2).to_string())

# --- Plot 1: Churn overview ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Churn Overview", fontsize=15, fontweight="bold")

churn_counts = df["Churn"].value_counts()
axes[0].pie(churn_counts, labels=["No Churn", "Churned"],
            autopct="%1.1f%%", startangle=90,
            colors=["#2ecc71", "#e74c3c"], explode=(0, 0.07),
            wedgeprops={"edgecolor": "white", "linewidth": 2})
axes[0].set_title("Overall Churn Rate")

contract_churn = (df.groupby("Contract")["Churn"]
                    .mean()
                    .reset_index()
                    .rename(columns={"Churn": "Churn Rate"})
                    .sort_values("Churn Rate", ascending=False))
bars = axes[1].bar(contract_churn["Contract"], contract_churn["Churn Rate"],
                   color=["#e74c3c", "#f39c12", "#2ecc71"],
                   edgecolor="white", linewidth=1.5)
axes[1].set_title("Churn Rate by Contract Type")
axes[1].set_ylabel("Churn Rate")
axes[1].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
for bar in bars:
    axes[1].annotate(f"{bar.get_height():.1%}",
                     xy=(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005),
                     ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/01_churn_overview.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 2: Numeric distributions by churn ----------------------------------
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Numeric Features by Churn Status", fontsize=14, fontweight="bold")
for ax, col in zip(axes, num_cols_raw):
    for churn_val, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
        data = df.loc[df["Churn"] == churn_val, col]
        ax.hist(data, bins=40, alpha=0.55, color=color, density=True,
                label="No Churn" if churn_val == 0 else "Churned")
        ax.axvline(data.mean(), color=color, lw=2, linestyle="--")
    ax.set_title(col); ax.set_ylabel("Density"); ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/02_numeric_distributions.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 3: Churn rate by key categorical features --------------------------
cat_feats = ["InternetService", "PaymentMethod", "TechSupport", "OnlineSecurity"]
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle("Churn Rate by Key Categorical Features", fontsize=14, fontweight="bold")
for ax, col in zip(axes.flatten(), cat_feats):
    data = (df.groupby(col)["Churn"].mean().reset_index()
              .rename(columns={"Churn": "Churn Rate"})
              .sort_values("Churn Rate", ascending=True))
    norm_vals = data["Churn Rate"] / data["Churn Rate"].max()
    colors_bar = [plt.cm.RdYlGn_r(v) for v in norm_vals]
    bars = ax.barh(data[col], data["Churn Rate"], color=colors_bar)
    ax.set_title(f"Churn Rate by {col}")
    ax.set_xlabel("Churn Rate")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    for bar, val in zip(bars, data["Churn Rate"]):
        ax.text(bar.get_width() + 0.003, bar.get_y() + bar.get_height()/2,
                f"{val:.1%}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_categorical_churn_rates.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 4: Correlation heatmap (numeric) -----------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
corr = df[num_cols_raw + ["Churn"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, linewidths=0.5, ax=ax, annot_kws={"size": 11})
ax.set_title("Correlation Matrix (Numeric + Churn)", fontsize=12)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_correlation_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 5: Violin plots ---------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Tenure & Monthly Charges vs Churn", fontsize=13, fontweight="bold")
for ax, col in zip(axes, ["tenure", "MonthlyCharges"]):
    sns.violinplot(data=df, x="Churn", y=col,
                   hue="Churn", palette={0: "#2ecc71", 1: "#e74c3c"},
                   inner="quartile", ax=ax, legend=False)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churned"])
    ax.set_title(f"{col} by Churn")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/05_violin_plots.png", dpi=150, bbox_inches="tight")
plt.close()

print("  5 EDA plots saved to ./outputs/")
print("[OK] EDA done.")


# ==============================================================================
# SECTION 3  —  FEATURE ENGINEERING
# ==============================================================================
print("\n" + "=" * 68)
print("SECTION 3  —  FEATURE ENGINEERING")
print("=" * 68)

df_fe = df.copy()

# --- 3.1  Tenure groups -------------------------------------------------------
# INTERVIEW NOTE: Binning continuous features into interpretable buckets helps
# both business communication and can improve tree-model performance.
df_fe["tenure_group"] = pd.cut(
    df_fe["tenure"],
    bins=[0, 12, 24, 48, 72],
    labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
    include_lowest=True
)
print("\nChurn rate by tenure group:")
print(df_fe.groupby("tenure_group")["Churn"]
      .agg(n_customers="count", churn_rate="mean")
      .round(3).to_string())

# --- 3.2  Number of value-added services -------------------------------------
svc_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
            "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
df_fe["num_services"] = (
    df_fe[svc_cols].isin(["Yes"]).sum(axis=1)
)
print(f"\nnum_services range: {df_fe['num_services'].min()}–{df_fe['num_services'].max()}")

# --- 3.3  Cost-per-service (value indicator) ---------------------------------
df_fe["charge_per_service"] = (
    df_fe["MonthlyCharges"] / (df_fe["num_services"] + 1)
).round(2)

# --- 3.4  Long-term contract flag --------------------------------------------
df_fe["is_longterm"] = df_fe["Contract"].isin(["One year", "Two year"]).astype(int)

# --- 3.5  Unprotected internet (risk indicator) ------------------------------
df_fe["unprotected_internet"] = (
    (df_fe["InternetService"] != "No") &
    (df_fe["OnlineSecurity"] == "No") &
    (df_fe["TechSupport"] == "No")
).astype(int)

# --- 3.6  Encode binary yes/no columns ---------------------------------------
binary_map  = {"Yes": 1, "No": 0,
               "No phone service": 0, "No internet service": 0}
yes_no_cols = ["Partner", "Dependents", "PhoneService", "MultipleLines",
               "OnlineSecurity", "OnlineBackup", "DeviceProtection",
               "TechSupport", "StreamingTV", "StreamingMovies", "PaperlessBilling"]
for col in yes_no_cols:
    df_fe[col] = df_fe[col].map(binary_map)
df_fe["gender"] = (df_fe["gender"] == "Male").astype(int)

# --- 3.7  One-hot encode multi-class categoricals ----------------------------
# INTERVIEW NOTE: drop_first=False keeps all categories for interpretability.
# For regularised models, drop_first=True avoids perfect multicollinearity.
ohe_cols = ["InternetService", "Contract", "PaymentMethod", "tenure_group"]
df_fe = pd.get_dummies(df_fe, columns=ohe_cols, drop_first=False)

feature_cols = [c for c in df_fe.columns if c != "Churn"]
X = df_fe[feature_cols].astype(float)
y = df_fe["Churn"].values
print(f"\nFinal feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features")

# --- 3.8  Scale numeric features for linear models --------------------------
# INTERVIEW NOTE: Tree models are scale-invariant, but LogReg needs scaling.
SCALE_COLS = ["tenure", "MonthlyCharges", "TotalCharges",
              "num_services", "charge_per_service"]
scaler   = StandardScaler()
X_scaled = X.copy()
X_scaled[SCALE_COLS] = scaler.fit_transform(X[SCALE_COLS])

print("[OK] Feature engineering done.")


# ==============================================================================
# SECTION 4  —  MODEL TRAINING & EVALUATION
# ==============================================================================
print("\n" + "=" * 68)
print("SECTION 4  —  MODEL TRAINING & EVALUATION")
print("=" * 68)

# --- 4.1  Stratified train/test split ----------------------------------------
# INTERVIEW NOTE: stratify=y preserves class ratio — crucial for imbalanced data.
X_tr_s, X_te_s, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.20, random_state=42, stratify=y
)
X_tr, X_te, _, _ = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain: {X_tr.shape[0]:,}  |  Test: {X_te.shape[0]:,}")
print(f"Train churn: {y_train.mean():.1%}  |  Test churn: {y_test.mean():.1%}")

# --- 4.2  Define models -------------------------------------------------------
if XGBOOST_AVAILABLE:
    boost = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=42
    )
    boost_label = "XGBoost"
else:
    boost = GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42
    )
    boost_label = "GradientBoosting"

models = {
    "LogisticRegression": (
        LogisticRegression(max_iter=500, C=0.5, class_weight="balanced",
                           solver="lbfgs", random_state=42),
        X_tr_s, X_te_s          # scaled data for LR
    ),
    "RandomForest": (
        RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        X_tr, X_te              # raw data for trees
    ),
    boost_label: (
        boost,
        X_tr, X_te
    ),
}

# --- 4.3  Cross-validation ---------------------------------------------------
# INTERVIEW NOTE: StratifiedKFold is the standard CV for classification —
# each fold mirrors the overall class distribution.
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
print("\n5-Fold Stratified CV  (ROC-AUC):")
cv_results = {}
for name, (model, Xtr, Xte) in models.items():
    scores = cross_val_score(model, Xtr, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_results[name] = scores
    print(f"  {name:30s}: {scores.mean():.4f} +/- {scores.std():.4f}")

# --- 4.4  Final fit + test-set evaluation ------------------------------------
print("\nTest-Set Metrics:")
trained = {}
results = {}
for name, (model, Xtr, Xte) in models.items():
    model.fit(Xtr, y_train)
    y_pred = model.predict(Xte)
    y_prob = model.predict_proba(Xte)[:, 1]
    trained[name] = (model, Xte, y_pred, y_prob)
    m = {
        "Accuracy" : accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall"   : recall_score(y_test, y_pred, zero_division=0),
        "F1"       : f1_score(y_test, y_pred, zero_division=0),
        "ROC-AUC"  : roc_auc_score(y_test, y_prob),
    }
    results[name] = m
    print(f"\n  {name}")
    for k, v in m.items():
        print(f"    {k:10s}: {v:.4f}")

# --- Plot 6: Metrics comparison ----------------------------------------------
metric_df = pd.DataFrame(results).T
fig, ax = plt.subplots(figsize=(13, 5))
metric_df.plot(kind="bar", ax=ax, rot=15, colormap="Set2",
               edgecolor="white", width=0.7)
ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
ax.set_ylabel("Score"); ax.set_ylim(0, 1.08)
ax.axhline(0.80, color="grey", linestyle="--", linewidth=0.8, alpha=0.7)
ax.legend(loc="lower right", fontsize=9)
for bar in ax.patches:
    if bar.get_height() > 0:
        ax.annotate(f"{bar.get_height():.2f}",
                    (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                    ha="center", fontsize=7, rotation=90)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/06_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 7: ROC curves ------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 6))
line_colors = ["#3498db", "#e74c3c", "#2ecc71"]
for (name, (model, Xte, yp, yprob)), col in zip(trained.items(), line_colors):
    fpr, tpr, _ = roc_curve(y_test, yprob)
    auc = roc_auc_score(y_test, yprob)
    ax.plot(fpr, tpr, color=col, lw=2.5, label=f"{name}  (AUC={auc:.3f})")
ax.plot([0,1],[0,1], "k--", lw=1, label="Random classifier")
ax.fill_between([0,1],[0,1], alpha=0.04, color="grey")
ax.set_xlabel("False Positive Rate", fontsize=12)
ax.set_ylabel("True Positive Rate", fontsize=12)
ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
ax.legend(fontsize=10); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/07_roc_curves.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 8: Confusion matrices ----------------------------------------------
fig, axes = plt.subplots(1, 3, figsize=(17, 5))
fig.suptitle("Confusion Matrices", fontsize=14, fontweight="bold")
for ax, (name, (model, Xte, yp, yprob)) in zip(axes, trained.items()):
    cm = confusion_matrix(y_test, yp)
    ConfusionMatrixDisplay(cm, display_labels=["No Churn","Churned"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(name, fontsize=11)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/08_confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()

# Print classification reports
print("\nClassification Reports:")
for name, (model, Xte, yp, yprob) in trained.items():
    print(f"\n{name}")
    print(classification_report(y_test, yp,
                                target_names=["No Churn","Churned"], digits=4))

print("[OK] Model training & evaluation done.")


# ==============================================================================
# SECTION 5  —  EXPLAINABILITY
# ==============================================================================
print("\n" + "=" * 68)
print("SECTION 5  —  EXPLAINABILITY")
print("=" * 68)

best_name = max(results, key=lambda n: results[n]["ROC-AUC"])
best_model, best_Xte, best_ypred, best_yprob = trained[best_name]
best_Xtr = X_tr_s if best_name == "LogisticRegression" else X_tr
print(f"\nBest model: {best_name}  (ROC-AUC = {results[best_name]['ROC-AUC']:.4f})")

# --- Plot 9: Built-in feature importance / coefficients ----------------------
fig, ax = plt.subplots(figsize=(10, 9))
if hasattr(best_model, "feature_importances_"):
    imp = best_model.feature_importances_
    idx = np.argsort(imp)[-20:]
    cmap_colors = [plt.cm.RdYlGn(v) for v in imp[idx] / imp[idx].max()]
    ax.barh([feature_cols[i] for i in idx], imp[idx], color=cmap_colors)
    ax.set_title(f"Top-20 Feature Importances — {best_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean decrease in impurity")
elif hasattr(best_model, "coef_"):
    coefs = np.abs(best_model.coef_[0])
    idx   = np.argsort(coefs)[-20:]
    ax.barh([feature_cols[i] for i in idx], coefs[idx], color="#3498db")
    ax.set_title(f"Top-20 |Coefficients| — {best_name}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("|Coefficient|")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/09_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# --- Plot 10: Permutation importance (model-agnostic) ------------------------
# INTERVIEW NOTE: Permutation importance measures the actual drop in test-set
# performance when a feature is shuffled — it captures the true predictive value
# rather than how often a tree splits on it.
print("\nComputing permutation importance (this can take ~30s)...")
perm = permutation_importance(best_model, best_Xte, y_test,
                               n_repeats=10, random_state=42,
                               scoring="roc_auc", n_jobs=-1)
perm_df = (pd.DataFrame({
               "feature":    feature_cols,
               "importance": perm.importances_mean,
               "std":        perm.importances_std
           })
           .sort_values("importance", ascending=False)
           .head(20)
           .reset_index(drop=True))

print("\nTop-10 features (permutation importance):")
print(perm_df.head(10).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 9))
ax.barh(perm_df["feature"][::-1], perm_df["importance"][::-1],
        xerr=perm_df["std"][::-1], color="#9b59b6", capsize=3, alpha=0.85)
ax.set_title(f"Top-20 Permutation Importances — {best_name}",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Mean ROC-AUC drop when feature shuffled")
ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/10_permutation_importance.png", dpi=150, bbox_inches="tight")
plt.close()

# --- SHAP (optional) ---------------------------------------------------------
if SHAP_AVAILABLE:
    print("\nGenerating SHAP summary plot...")
    if hasattr(best_model, "feature_importances_"):
        explainer   = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(best_Xte)
        sv = shap_values[1] if isinstance(shap_values, list) else shap_values
    else:
        explainer = shap.LinearExplainer(best_model, best_Xtr)
        sv = explainer.shap_values(best_Xte)
    shap.summary_plot(sv, best_Xte, feature_names=feature_cols, show=False)
    plt.savefig(f"{OUTPUT_DIR}/11_shap_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  -> Saved 11_shap_summary.png")
else:
    print("\nSHAP not installed. Install with:  pip install shap")

print("[OK] Explainability done.")


# ==============================================================================
# SECTION 6  —  BUSINESS SUMMARY
# ==============================================================================
print("\n" + "=" * 68)
print("SECTION 6  —  BUSINESS SUMMARY")
print("=" * 68)

churn_rate   = y.mean()
n_churners   = int(y.sum())
revenue_save = int(n_churners * 0.20 * 500)

summary = f"""
+------------------------------------------------------------------+
|        CUSTOMER CHURN PREDICTION — BUSINESS SUMMARY             |
+------------------------------------------------------------------+
|                                                                  |
| PROBLEM                                                          |
|  Acquiring a new customer costs 5-25x more than retaining one.  |
|  Predicting churn before it happens lets the business target     |
|  at-risk customers with personalised retention offers.           |
|                                                                  |
| DATASET   7,043 customers  x  21 features  |  {churn_rate:.1%} churn rate |
|                                                                  |
| KEY FINDINGS                                                     |
|  - Month-to-month contracts have the highest churn (~42 %).      |
|  - Fiber-optic customers churn more than DSL customers.          |
|  - Customers without TechSupport/OnlineSecurity churn more.      |
|  - Tenure < 12 months is the strongest churn signal.             |
|  - Electronic check payment method correlates with higher churn. |
|                                                                  |
| BEST MODEL  {best_name:<30s}                    |
|  ROC-AUC :  {results[best_name]['ROC-AUC']:.4f}                                          |
|  F1 Score:  {results[best_name]['F1']:.4f}                                          |
|  Recall  :  {results[best_name]['Recall']:.4f}  (churners correctly identified)    |
|                                                                  |
| ESTIMATED BUSINESS VALUE (illustrative)                          |
|  Total churners per cycle  :  {n_churners:,} customers                    |
|  If we prevent 20 % of churn  ->  {int(n_churners*0.20):,} customers saved          |
|  Revenue protected (@ $500/yr):  ${revenue_save:,}                     |
|                                                                  |
| RECOMMENDED ACTIONS                                              |
|  1. Target M2M contract holders with upgrade incentives.         |
|  2. Bundle security services for Fiber-optic customers.          |
|  3. Run early-life retention program for tenure < 12 months.     |
|  4. Investigate why Electronic Check correlates with churn.      |
|                                                                  |
| NEXT STEPS (production path)                                     |
|  - Retrain monthly; monitor for concept drift.                   |
|  - A/B test interventions to quantify lift.                      |
|  - Weight predictions by Customer Lifetime Value.                |
|  - Deploy via REST API + Streamlit dashboard (see app.py).       |
|                                                                  |
+------------------------------------------------------------------+
"""
print(summary)
with open(f"{OUTPUT_DIR}/business_summary.txt", "w") as f:
    f.write(summary)

# ==============================================================================
# SAVE ARTIFACTS FOR STREAMLIT APP
# ==============================================================================
artifacts = {
    "best_model"  : best_model,
    "best_name"   : best_name,
    "scaler"      : scaler,
    "feature_cols": feature_cols,
    "scale_cols"  : SCALE_COLS,
    "results"     : results,
    "perm_df"     : perm_df,
}
with open("model_artifacts.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("=" * 68)
print(f"COMPLETE — all outputs saved to {OUTPUT_DIR}/")
print("  Next: streamlit run app.py")
print("=" * 68)
