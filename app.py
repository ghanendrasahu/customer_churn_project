"""
app.py  —  Streamlit Dashboard for Customer Churn Prediction
------------------------------------------------------------
Prerequisites:
  pip install streamlit pandas numpy scikit-learn matplotlib seaborn

Run:
  streamlit run app.py
"""

import pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import streamlit as st

warnings.filterwarnings("ignore")

# ===========================================================================
# Page config
# ===========================================================================
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished look
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stMetric { background-color: white; border-radius: 10px; padding: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,.1); }
    .metric-card { background: white; border-radius: 10px; padding: 20px;
                   box-shadow: 0 2px 8px rgba(0,0,0,.08); margin: 5px 0; }
    h1 { color: #2c3e50; }
    h2, h3 { color: #34495e; }
    .risk-high { color: #e74c3c; font-size: 24px; font-weight: bold; }
    .risk-low  { color: #27ae60; font-size: 24px; font-weight: bold; }
    .sidebar-header { font-size: 18px; font-weight: bold; color: #2c3e50;
                      margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# ===========================================================================
# Load model artifacts
# ===========================================================================
@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    with open("model_artifacts.pkl", "rb") as f:
        return pickle.load(f)

try:
    art         = load_artifacts()
    model       = art["best_model"]
    model_name  = art["best_name"]
    scaler      = art["scaler"]
    feature_cols = art["feature_cols"]
    scale_cols  = art["scale_cols"]
    results_dict = art["results"]
    perm_df     = art["perm_df"]
    MODEL_LOADED = True
except FileNotFoundError:
    MODEL_LOADED = False

# ===========================================================================
# Helpers
# ===========================================================================
YES_NO = ["No", "Yes"]
BINARY = {s: i for i, s in enumerate(YES_NO)}

def build_input_row(cfg: dict) -> pd.DataFrame:
    """Convert sidebar widget values into the feature matrix expected by the model."""
    row = {col: 0 for col in feature_cols}

    # Numeric raw features
    row["tenure"]          = cfg["tenure"]
    row["MonthlyCharges"]  = cfg["monthly_charges"]
    row["TotalCharges"]    = cfg["tenure"] * cfg["monthly_charges"]
    row["SeniorCitizen"]   = 1 if cfg["senior_citizen"] else 0

    # Binary yes/no
    row["gender"]            = 1 if cfg["gender"] == "Male" else 0
    row["Partner"]           = BINARY[cfg["partner"]]
    row["Dependents"]        = BINARY[cfg["dependents"]]
    row["PhoneService"]      = BINARY[cfg["phone_service"]]
    row["MultipleLines"]     = BINARY[cfg["multiple_lines"]]
    row["OnlineSecurity"]    = BINARY[cfg["online_security"]]
    row["OnlineBackup"]      = BINARY[cfg["online_backup"]]
    row["DeviceProtection"]  = BINARY[cfg["device_protection"]]
    row["TechSupport"]       = BINARY[cfg["tech_support"]]
    row["StreamingTV"]       = BINARY[cfg["streaming_tv"]]
    row["StreamingMovies"]   = BINARY[cfg["streaming_movies"]]
    row["PaperlessBilling"]  = BINARY[cfg["paperless_billing"]]

    # Engineered features
    svc_keys = ["phone_service","multiple_lines","online_security","online_backup",
                "device_protection","tech_support","streaming_tv","streaming_movies"]
    row["num_services"]         = sum(1 for k in svc_keys if cfg[k] == "Yes")
    row["charge_per_service"]   = cfg["monthly_charges"] / (row["num_services"] + 1)
    row["is_longterm"]          = 1 if cfg["contract"] in ["One year","Two year"] else 0
    row["unprotected_internet"] = int(
        cfg["internet_service"] != "No"
        and cfg["online_security"] == "No"
        and cfg["tech_support"] == "No"
    )

    # One-hot: InternetService
    for cat in ["DSL", "Fiber optic", "No"]:
        row[f"InternetService_{cat}"] = int(cfg["internet_service"] == cat)

    # One-hot: Contract
    for cat in ["Month-to-month", "One year", "Two year"]:
        row[f"Contract_{cat}"] = int(cfg["contract"] == cat)

    # One-hot: PaymentMethod
    for cat in ["Bank transfer (automatic)", "Credit card (automatic)",
                "Electronic check", "Mailed check"]:
        row[f"PaymentMethod_{cat}"] = int(cfg["payment_method"] == cat)

    # One-hot: tenure_group
    t = cfg["tenure"]
    tg_map = {
        "0-12mo" : t <= 12,
        "13-24mo": 13 <= t <= 24,
        "25-48mo": 25 <= t <= 48,
        "49-72mo": t >= 49,
    }
    for label, flag in tg_map.items():
        row[f"tenure_group_{label}"] = int(flag)

    df_row = pd.DataFrame([row])
    # Ensure all expected columns exist (fill missing with 0)
    for col in feature_cols:
        if col not in df_row.columns:
            df_row[col] = 0
    return df_row[feature_cols].astype(float)


def make_gauge_fig(prob: float) -> plt.Figure:
    """Draw a semi-circular gauge showing churn probability."""
    fig, ax = plt.subplots(figsize=(5, 3),
                            subplot_kw={"projection": "polar"})
    theta = np.linspace(0, np.pi, 200)
    # Background track
    ax.plot(theta, [1]*200, color="#ecf0f1", linewidth=18, solid_capstyle="round")
    # Filled arc proportional to probability
    fill_theta = np.linspace(0, np.pi * prob, 200)
    color = "#e74c3c" if prob > 0.5 else ("#f39c12" if prob > 0.3 else "#2ecc71")
    ax.plot(fill_theta, [1]*len(fill_theta), color=color,
            linewidth=18, solid_capstyle="round")
    ax.set_xlim(0, np.pi)
    ax.set_ylim(0, 1.5)
    ax.axis("off")
    ax.text(np.pi/2, 0.3, f"{prob:.1%}", ha="center", va="center",
            fontsize=28, fontweight="bold", color=color,
            transform=ax.transData)
    ax.text(np.pi/2, -0.25, "Churn Probability", ha="center", va="center",
            fontsize=11, color="#7f8c8d", transform=ax.transData)
    fig.patch.set_facecolor("white")
    plt.tight_layout(pad=0)
    return fig


def make_feature_importance_fig(perm_df: pd.DataFrame, n: int = 12) -> plt.Figure:
    df_plot = perm_df.head(n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(7, 5))
    cmap_vals = df_plot["importance"] / df_plot["importance"].max()
    colors = [plt.cm.RdYlGn(v) for v in cmap_vals]
    ax.barh(df_plot["feature"], df_plot["importance"], color=colors, edgecolor="white")
    ax.set_title(f"Top-{n} Drivers of Churn (Permutation Importance)",
                 fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Mean ROC-AUC drop when feature is shuffled")
    ax.grid(axis="x", alpha=0.3)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig


def make_model_metrics_fig(results_dict: dict) -> plt.Figure:
    df = pd.DataFrame(results_dict).T
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    df = df[metrics]
    fig, ax = plt.subplots(figsize=(9, 4))
    df.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white",
            width=0.7, rot=15)
    ax.set_title("Model Benchmark (Test Set)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", fontsize=8)
    ax.axhline(0.75, color="grey", linestyle="--", linewidth=0.7, alpha=0.6)
    for bar in ax.patches:
        if bar.get_height() > 0.01:
            ax.annotate(f"{bar.get_height():.2f}",
                        (bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01),
                        ha="center", fontsize=6.5, rotation=90)
    fig.patch.set_facecolor("white")
    plt.tight_layout()
    return fig

# ===========================================================================
# SIDEBAR  —  Customer Input Form
# ===========================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-header">📋 Customer Profile</p>',
                unsafe_allow_html=True)
    st.caption("Adjust the inputs to predict churn probability.")

    with st.expander("👤 Demographics", expanded=True):
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.checkbox("Senior Citizen")
        partner        = st.selectbox("Partner", YES_NO)
        dependents     = st.selectbox("Dependents", YES_NO)

    with st.expander("📅 Account", expanded=True):
        tenure         = st.slider("Tenure (months)", 0, 72, 24)
        contract       = st.selectbox("Contract",
                             ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", YES_NO)
        payment_method = st.selectbox("Payment Method",
                             ["Electronic check", "Mailed check",
                              "Bank transfer (automatic)",
                              "Credit card (automatic)"])

    with st.expander("🌐 Services", expanded=True):
        phone_service     = st.selectbox("Phone Service", YES_NO)
        multiple_lines    = st.selectbox("Multiple Lines", YES_NO)
        internet_service  = st.selectbox("Internet Service",
                                          ["DSL", "Fiber optic", "No"])
        if internet_service != "No":
            online_security   = st.selectbox("Online Security", YES_NO)
            online_backup     = st.selectbox("Online Backup", YES_NO)
            device_protection = st.selectbox("Device Protection", YES_NO)
            tech_support      = st.selectbox("Tech Support", YES_NO)
            streaming_tv      = st.selectbox("Streaming TV", YES_NO)
            streaming_movies  = st.selectbox("Streaming Movies", YES_NO)
        else:
            online_security = online_backup = device_protection = "No"
            tech_support    = streaming_tv = streaming_movies   = "No"

    with st.expander("💳 Billing", expanded=True):
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 50.0, step=0.5)

    predict_btn = st.button("🔮  Predict Churn", use_container_width=True, type="primary")

# ===========================================================================
# MAIN AREA
# ===========================================================================
st.title("📡 Customer Churn Prediction Dashboard")
st.caption("Telco Customer Churn  ·  End-to-End ML Demo  ·  Interview-Ready Project")

if not MODEL_LOADED:
    st.error("⚠️ `model_artifacts.pkl` not found. "
             "Please run `python churn_analysis.py` first to train and save the model.")
    st.stop()

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Model Insights", "📖 About"])

# ---------------------------------------------------------------------------
# TAB 1 — PREDICTION
# ---------------------------------------------------------------------------
with tab1:
    cfg = dict(
        gender=gender, senior_citizen=senior_citizen,
        partner=partner, dependents=dependents,
        tenure=tenure, contract=contract,
        paperless_billing=paperless_billing, payment_method=payment_method,
        phone_service=phone_service, multiple_lines=multiple_lines,
        internet_service=internet_service,
        online_security=online_security, online_backup=online_backup,
        device_protection=device_protection, tech_support=tech_support,
        streaming_tv=streaming_tv, streaming_movies=streaming_movies,
        monthly_charges=monthly_charges,
    )

    X_input = build_input_row(cfg)

    # Apply scaling if model uses it
    if model_name == "LogisticRegression":
        X_pred = X_input.copy()
        X_pred[scale_cols] = scaler.transform(X_input[scale_cols])
    else:
        X_pred = X_input

    churn_prob = float(model.predict_proba(X_pred)[0, 1])
    churn_pred = int(churn_prob >= 0.5)

    col1, col2 = st.columns([1, 1.4], gap="large")

    with col1:
        st.subheader("Prediction Result")

        if churn_pred == 1:
            st.markdown(
                f'<div class="metric-card"><p class="risk-high">'
                f'⚠️ HIGH CHURN RISK</p>'
                f'<p style="color:#7f8c8d">This customer is likely to churn.</p></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="metric-card"><p class="risk-low">'
                f'✅ LOW CHURN RISK</p>'
                f'<p style="color:#7f8c8d">This customer is likely to stay.</p></div>',
                unsafe_allow_html=True
            )

        st.pyplot(make_gauge_fig(churn_prob), use_container_width=True)

        # Risk band label
        if churn_prob < 0.3:
            risk_label, risk_color = "LOW", "#27ae60"
        elif churn_prob < 0.6:
            risk_label, risk_color = "MEDIUM", "#f39c12"
        else:
            risk_label, risk_color = "HIGH", "#e74c3c"

        st.markdown(
            f"<div style='text-align:center;font-size:18px;font-weight:bold;"
            f"color:{risk_color};margin-top:8px'>Risk Band: {risk_label}</div>",
            unsafe_allow_html=True
        )

    with col2:
        st.subheader("Customer Summary")

        metrics_row1 = st.columns(3)
        metrics_row1[0].metric("Tenure",    f"{tenure} mo")
        metrics_row1[1].metric("Monthly $", f"${monthly_charges:.0f}")
        metrics_row1[2].metric("Contract",  contract.split()[0])

        metrics_row2 = st.columns(3)
        metrics_row2[0].metric("Internet",  internet_service)
        metrics_row2[1].metric("Services",  str(int(X_input["num_services"].iloc[0])))
        metrics_row2[2].metric("Long-term", "Yes" if int(X_input["is_longterm"].iloc[0]) else "No")

        st.markdown("---")
        st.subheader("Retention Recommendations")

        recs = []
        if contract == "Month-to-month":
            recs.append("📝 Offer **discounted annual contract** — M2M customers churn 3× more.")
        if internet_service == "Fiber optic":
            recs.append("🌐 Bundle **OnlineSecurity + TechSupport** for Fiber customers.")
        if tenure < 12:
            recs.append("🎁 Enrol in **early-life loyalty programme** (tenure < 12 mo).")
        if payment_method == "Electronic check":
            recs.append("💳 Incentivise switch to **auto-pay** (credit card / bank transfer).")
        if float(X_input["unprotected_internet"].iloc[0]) == 1:
            recs.append("🔒 Upsell **security bundle** — unprotected internet users churn more.")
        if not recs:
            recs.append("✅ Customer profile looks stable — continue standard engagement.")

        for r in recs:
            st.markdown(f"- {r}")

        st.markdown("---")
        total_est = tenure * monthly_charges
        clv_est   = monthly_charges * 24    # simple 2-year forward CLV
        col_a, col_b = st.columns(2)
        col_a.metric("Total Spend to Date", f"${total_est:,.0f}")
        col_b.metric("Est. 2-yr Forward CLV", f"${clv_est:,.0f}")

# ---------------------------------------------------------------------------
# TAB 2 — MODEL INSIGHTS
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Model Performance Benchmark")
    st.markdown(f"**Active model:** `{model_name}`")
    st.pyplot(make_model_metrics_fig(results_dict), use_container_width=True)

    st.markdown("---")
    st.subheader("Top Churn Drivers (Permutation Importance)")
    st.markdown(
        "Permutation importance measures how much the model's ROC-AUC drops "
        "when each feature is randomly shuffled — a true measure of predictive value."
    )
    st.pyplot(make_feature_importance_fig(perm_df, n=12), use_container_width=True)

    st.markdown("---")
    st.subheader("Saved EDA Plots")
    import os
    plot_files = sorted([f for f in os.listdir("./outputs") if f.endswith(".png")])
    if plot_files:
        cols = st.columns(2)
        for i, fname in enumerate(plot_files):
            cols[i % 2].image(f"./outputs/{fname}",
                              caption=fname.replace("_", " ").replace(".png", ""),
                              use_column_width=True)
    else:
        st.info("Run `python churn_analysis.py` to generate EDA plots.")

# ---------------------------------------------------------------------------
# TAB 3 — ABOUT
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("About This Project")
    st.markdown("""
    ### Customer Churn Prediction — End-to-End DS Project

    **Business Problem**
    Customer churn is one of the most critical KPIs for subscription businesses.
    Predicting it accurately allows teams to intervene *before* a customer leaves,
    protecting revenue and reducing acquisition spend.

    **Dataset**
    Telco Customer Churn (Kaggle-style) — 7,043 customers, 21 features, ~23% churn rate.

    **Methodology**
    | Step | Detail |
    |------|--------|
    | Cleaning | Coerce TotalCharges, logical imputation, outlier inspection |
    | EDA | Distribution plots, categorical churn rates, correlation heatmap |
    | Feature Eng. | Tenure groups, service count, cost-per-service, risk flags |
    | Encoding | Binary map for yes/no, one-hot for multi-class |
    | Scaling | StandardScaler applied to numeric features (for LR only) |
    | Models | Logistic Regression, Random Forest, GradientBoosting / XGBoost |
    | Validation | Stratified 5-fold CV + held-out 20% test set |
    | Metrics | Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix |
    | Explainability | Built-in importance + Permutation importance + SHAP (optional) |

    **Key Interview Talking Points**
    - *Why ROC-AUC over accuracy?* — Class imbalance (77/23) makes accuracy misleading.
    - *Why recall matters here?* — Missing a churner (FN) costs more than a false alarm (FP).
    - *Why permutation importance?* — Tree impurity importance is biased toward high-cardinality features.
    - *Why scale only for LR?* — Tree models are scale-invariant; scaling adds no value.
    - *Stratify=y in train_test_split?* — Preserves class ratio across splits.
    - *Business framing?* — Always connect model metrics to dollar impact (CLV, CAC).

    **Stack:** Python · pandas · scikit-learn · matplotlib · seaborn · Streamlit
    Optional: XGBoost · SHAP
    """)
