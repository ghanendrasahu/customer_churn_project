from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI()

# Load artifacts
with open("model_artifacts.pkl", "rb") as f:
    artifacts = pickle.load(f)

model = artifacts["best_model"]
feature_cols = artifacts["feature_cols"]
scaler = artifacts.get("scaler")
scale_cols = artifacts.get("scale_cols")


@app.get("/")
def home():
    return {"message": "Churn Prediction API Running"}


def preprocess(data: dict):
    df = pd.DataFrame([data])

    # --- SAME FEATURE ENGINEERING AS TRAINING ---

    # num_services
    svc_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    df["num_services"] = df[svc_cols].sum(axis=1)

    # charge_per_service
    df["charge_per_service"] = df["MonthlyCharges"] / (df["num_services"] + 1)

    # is_longterm
    df["is_longterm"] = df["Contract"].isin(["One year", "Two year"]).astype(int)

    # unprotected_internet
    df["unprotected_internet"] = (
        (df["InternetService"] != "No") &
        (df["OnlineSecurity"] == 0) &
        (df["TechSupport"] == 0)
    ).astype(int)

    # tenure_group
    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=["0-12mo", "13-24mo", "25-48mo", "49-72mo"],
        include_lowest=True
    )

    # One-hot encoding
    df = pd.get_dummies(df)

    # Align features
    df = df.reindex(columns=feature_cols, fill_value=0)

    # Scaling
    if scaler is not None:
        df[scale_cols] = scaler.transform(df[scale_cols])

    return df


@app.post("/predict")
def predict(data: dict):
    try:
        df = preprocess(data)

        prob = model.predict_proba(df)[0][1]
        prediction = int(prob > 0.5)

        return {
            "churn_probability": float(prob),
            "prediction": prediction
        }

    except Exception as e:
        return {"error": str(e)}