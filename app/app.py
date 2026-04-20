import streamlit as st
import requests

st.title("Churn Prediction")

# -------- INPUT --------
data = {
    "gender": st.selectbox("Gender", [1, 0]),
    "SeniorCitizen": st.selectbox("Senior Citizen", [0, 1]),
    "Partner": st.selectbox("Partner", [0, 1]),
    "Dependents": st.selectbox("Dependents", [0, 1]),
    "tenure": st.slider("Tenure", 0, 72, 12),
    "PhoneService": st.selectbox("Phone Service", [1, 0]),
    "MultipleLines": st.selectbox("Multiple Lines", [1, 0]),
    "OnlineSecurity": st.selectbox("Online Security", [1, 0]),
    "OnlineBackup": st.selectbox("Online Backup", [1, 0]),
    "DeviceProtection": st.selectbox("Device Protection", [1, 0]),
    "TechSupport": st.selectbox("Tech Support", [1, 0]),
    "StreamingTV": st.selectbox("Streaming TV", [1, 0]),
    "StreamingMovies": st.selectbox("Streaming Movies", [1, 0]),
    "PaperlessBilling": st.selectbox("Paperless Billing", [1, 0]),
    "MonthlyCharges": st.number_input("Monthly Charges", 0.0, 200.0, 70.0),
    "TotalCharges": st.number_input("Total Charges", 0.0, 10000.0, 840.0),
    "InternetService": st.selectbox("Internet", ["DSL", "Fiber optic", "No"]),
    "Contract": st.selectbox("Contract", ["Month-to-month", "One year", "Two year"]),
    "PaymentMethod": st.selectbox("Payment", [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)"
    ])
}

# -------- PREDICT --------
if st.button("Predict"):
    try:
        res = requests.post("http://127.0.0.1:8000/predict", json=data).json()

        if "error" in res:
            st.error(res["error"])
        else:
            st.write(f"Probability: {res['churn_probability']:.2f}")
            st.write("Churn" if res["prediction"] else "No Churn")

    except Exception as e:
        st.error(f"Connection Error: {e}")