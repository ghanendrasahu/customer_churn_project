"""
generate_dataset.py
-------------------
Creates a realistic synthetic Telco Customer Churn dataset that mirrors
the structure and statistics of the well-known Kaggle Telco Churn dataset.

Run this once:  python generate_dataset.py
Output:         telco_churn.csv  (7 043 rows × 21 columns)
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 7043

# ── Demographics ──────────────────────────────────────────────────────────────
gender          = np.random.choice(["Male", "Female"], N)
senior_citizen  = np.random.choice([0, 1], N, p=[0.84, 0.16])
partner         = np.random.choice(["Yes", "No"], N, p=[0.48, 0.52])
dependents      = np.random.choice(["Yes", "No"], N, p=[0.30, 0.70])

# ── Account ───────────────────────────────────────────────────────────────────
tenure          = np.random.randint(0, 72, N)                          # months
contract        = np.random.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]
)
paperless_billing = np.random.choice(["Yes", "No"], N, p=[0.59, 0.41])
payment_method = np.random.choice(
    ["Electronic check", "Mailed check",
     "Bank transfer (automatic)", "Credit card (automatic)"],
    N, p=[0.34, 0.23, 0.22, 0.21]
)

# ── Services ──────────────────────────────────────────────────────────────────
phone_service   = np.random.choice(["Yes", "No"], N, p=[0.90, 0.10])
multiple_lines  = np.where(
    phone_service == "No", "No phone service",
    np.random.choice(["Yes", "No"], N, p=[0.42, 0.58])
)
internet_service = np.random.choice(
    ["DSL", "Fiber optic", "No"], N, p=[0.34, 0.44, 0.22]
)

def internet_addon(internet_service, yes_p=0.28):
    out = np.where(
        internet_service == "No", "No internet service",
        np.random.choice(["Yes", "No"], N, p=[yes_p, 1 - yes_p])
    )
    return out

online_security   = internet_addon(internet_service, 0.29)
online_backup     = internet_addon(internet_service, 0.34)
device_protection = internet_addon(internet_service, 0.34)
tech_support      = internet_addon(internet_service, 0.29)
streaming_tv      = internet_addon(internet_service, 0.38)
streaming_movies  = internet_addon(internet_service, 0.39)

# ── Charges ───────────────────────────────────────────────────────────────────
monthly_charges = np.round(
    20 + 10 * (internet_service == "Fiber optic").astype(int)
      + 5  * (internet_service == "DSL").astype(int)
      + 3  * (multiple_lines == "Yes").astype(int)
      + 2  * (online_security == "Yes").astype(int)
      + 2  * (online_backup == "Yes").astype(int)
      + 2  * (device_protection == "Yes").astype(int)
      + 2  * (tech_support == "Yes").astype(int)
      + 3  * (streaming_tv == "Yes").astype(int)
      + 3  * (streaming_movies == "Yes").astype(int)
      + np.random.normal(0, 5, N),
    2
)
monthly_charges = np.clip(monthly_charges, 18.25, 118.75)

# Realistic total: tenure × monthly ± small noise; zero if tenure == 0
total_charges = np.where(
    tenure == 0, 0.0,
    np.round(tenure * monthly_charges + np.random.normal(0, 20, N), 2)
)
total_charges = np.maximum(total_charges, 0)

# Randomly blank ~11 TotalCharges to mimic the original dataset's missing values
blank_idx = np.random.choice(N, size=11, replace=False)
total_charges_str = total_charges.astype(str)
total_charges_str[blank_idx] = " "

# ── Churn (target) ────────────────────────────────────────────────────────────
# Realistic churn probability: higher for Fiber, M2M, high charges, low tenure
churn_prob = (
    0.05
    + 0.25 * (contract == "Month-to-month").astype(float)
    + 0.15 * (internet_service == "Fiber optic").astype(float)
    + 0.05 * (payment_method == "Electronic check").astype(float)
    + 0.08 * (tenure < 6).astype(float)
    - 0.10 * (tenure > 36).astype(float)
    - 0.10 * (contract == "Two year").astype(float)
    + 0.03 * (tech_support == "No").astype(float) * (internet_service != "No").astype(float)
    + np.random.normal(0, 0.04, N)
)
churn_prob = np.clip(churn_prob, 0.02, 0.90)
churn = np.where(np.random.rand(N) < churn_prob, "Yes", "No")

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "customerID":         [f"xxxx-{str(i).zfill(5)}" for i in range(N)],
    "gender":             gender,
    "SeniorCitizen":      senior_citizen,
    "Partner":            partner,
    "Dependents":         dependents,
    "tenure":             tenure,
    "PhoneService":       phone_service,
    "MultipleLines":      multiple_lines,
    "InternetService":    internet_service,
    "OnlineSecurity":     online_security,
    "OnlineBackup":       online_backup,
    "DeviceProtection":   device_protection,
    "TechSupport":        tech_support,
    "StreamingTV":        streaming_tv,
    "StreamingMovies":    streaming_movies,
    "Contract":           contract,
    "PaperlessBilling":   paperless_billing,
    "PaymentMethod":      payment_method,
    "MonthlyCharges":     monthly_charges,
    "TotalCharges":       total_charges_str,
    "Churn":              churn,
})

df.to_csv("telco_churn.csv", index=False)
print(f"Dataset saved → telco_churn.csv  ({len(df):,} rows × {len(df.columns)} cols)")
print(f"Churn rate: {(df.Churn=='Yes').mean():.1%}")
