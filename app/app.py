import streamlit as st
import requests

# Page Config
st.set_page_config(page_title="Telco Churn Analytics", layout="wide")

st.title("📡 Customer Churn Prediction Dashboard")
st.markdown("---")

# Use columns to organize the input form
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])

with col2:
    st.subheader("Subscription Details")
    tenure = st.slider("Tenure (Months)", 0, 72, 12)
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", 
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

with col3:
    st.subheader("Services & Charges")
    internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    monthly = st.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total = st.number_input("Total Charges ($)", 0.0, 10000.0, 840.0)

# Optional Services (Expander to save space)
with st.expander("Additional Services (Add-ons)"):
    s1, s2, s3 = st.columns(3)
    phone = s1.selectbox("Phone Service", ["Yes", "No"])
    multiple = s1.selectbox("Multiple Lines", ["Yes", "No"])
    security = s2.selectbox("Online Security", ["Yes", "No"])
    backup = s2.selectbox("Online Backup", ["Yes", "No"])
    protection = s3.selectbox("Device Protection", ["Yes", "No"])
    support = s3.selectbox("Tech Support", ["Yes", "No"])
    tv = st.selectbox("Streaming TV", ["Yes", "No"])
    movies = st.selectbox("Streaming Movies", ["Yes", "No"])

# Map UI labels to API expected values
data = {
    "gender": gender,
    "SeniorCitizen": 1 if senior == "Yes" else 0,
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": protection,
    "TechSupport": support,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "PaperlessBilling": paperless,
    "MonthlyCharges": monthly,
    "TotalCharges": total,
    "InternetService": internet,
    "Contract": contract,
    "PaymentMethod": payment
}

st.markdown("---")

# -------- PREDICTION LOGIC --------
if st.button("Analyze Customer Risk", use_container_width=True):
    try:
        # Connect to FastAPI backend
        res = requests.post("http://127.0.0.1:8000/predict", json=data).json()

        if "error" in res:
            st.error(f"Logic Error: {res['error']}")
        else:
            prob = res['churn_probability']
            risk_val = prob * 100
            
            # Display Result in Columns
            res_col1, res_col2 = st.columns(2)
            
            with res_col1:
                st.metric(label="Churn Probability", value=f"{risk_val:.1f}%")
                if prob > 0.7:
                    st.error("🚨 HIGH RISK: This customer is likely to leave.")
                elif prob > 0.4:
                    st.warning("⚠️ MEDIUM RISK: Targeted retention recommended.")
                else:
                    st.success("✅ LOW RISK: This is a loyal customer.")

            with res_col2:
                st.subheader("Retention Strategy")
                if prob > 0.5:
                    st.write("1. Offer upgrade to **Long-term Contract**.")
                    st.write("2. Bundle **Tech Support** or **Online Security**.")
                    st.write("3. Provide a loyalty discount on **Monthly Charges**.")
                else:
                    st.write("Maintain current service level. Upsell premium streaming content.")

    except Exception as e:
        st.error(f"Connection Error: Is your FastAPI server running? (uvicorn app.api:app)")