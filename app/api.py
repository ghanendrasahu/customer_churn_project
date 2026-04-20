from fastapi import FastAPI
import pickle
import pandas as pd
import numpy as np
from pathlib import Path

app = FastAPI()

# 1. Setup paths
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model_artifacts.pkl"

# 2. Load the artifacts
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file at: {MODEL_PATH}")

with open(MODEL_PATH, "rb") as f:
    artifacts = pickle.load(f)

# 3. Extract variables
model = artifacts["best_model"]
feature_cols = artifacts["feature_cols"]
scaler = artifacts.get("scaler")
scale_cols = artifacts.get("scale_cols")

print(f"✅ Pipeline ready. Loaded {len(feature_cols)} features.")

@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}

def preprocess(data: dict):
    # Convert input dict to DataFrame
    df = pd.DataFrame([data])

    # --- DATA CLEANING ---
    # Convert specific columns to numeric to avoid "concatenate str" errors
    # and to allow math operations like .sum()
    svc_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    
    for col in svc_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)

    # --- FEATURE ENGINEERING ---

    # 1. num_services (Now works because columns are 1s and 0s)
    df["num_services"] = df[svc_cols].sum(axis=1)

    # 2. charge_per_service
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    # 3. is_longterm
    df["is_longterm"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    # 4. unprotected_internet 
    # (Updated to check for "no" as a string or 0)
    df["unprotected_internet"] = (
        (df["InternetService"] != "No") &
        (df["OnlineSecurity"] == 0) &
        (df["TechSupport"] == 0)
    ).astype(int)

    # 5. tenure_group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
        include_lowest=True
    )

    # --- DUMMIES & ALIGNMENT ---
    # One-hot encoding
    df = pd.get_dummies(df)

    # Align features: This ensures the 34 features match your training set
    df = df.reindex(columns=feature_cols, fill_value=0)

    # --- SCALING ---
    if scaler is not None and scale_cols is not None:
        # Ensure we pass a 2D DataFrame to the scaler
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df

@app.post("/predict")
def predict(data: dict):
    try:
        # Get processed features
        df_processed = preprocess(data)

        # Generate prediction
        prob = model.predict_proba(df_processed)[0][1]
        prediction = int(prob > 0.5)

        return {
            "churn_probability": round(float(prob), 4),
            "prediction": prediction,
            "status": "Success"
        }

    except Exception as e:
        import traceback
        print(traceback.format_exc()) # Prints the full error to your console
        return {"error": str(e)}