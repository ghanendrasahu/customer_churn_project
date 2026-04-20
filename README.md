# Customer Churn Prediction — End-to-End Data Science Project

# 📡 Telco Customer Churn Prediction System

> **Interview-Ready ML Project:** A professional, end-to-end Machine Learning pipeline. This project demonstrates a production-style architecture by decoupling the training environment from a high-performance **FastAPI** inference engine and a **Streamlit** business dashboard.

---

## 🏗️ Project Structure

```text
customer_churn_project/
├── app/
│   ├── api.py              # The Backend: FastAPI service for real-time predictions.
│   └── app.py              # The Frontend: Streamlit dashboard for business users.
├── data/
│   └── telco_churn.csv     # The Dataset: Synthetic customer data for analysis.
├── models/
│   └── model_artifacts.pkl # The Artifacts: Saved model, scaler, and feature names.
├── src/
│   ├── generate_dataset.py # Data Factory: Script to create synthetic Telco data.
│   └── churn_analysis.py   # The Pipeline: EDA, Cleaning, and Model Training.
├── outputs/                # Visualizations: ROC curves, SHAP, and Importance plots.
├── requirements.txt        # Dependencies: Python libraries required.
└── README.md               # Documentation: You are here.
```

---

## 🚀 Quick Start

Follow these steps to get the project running on your computer:

```bash
#1. Installation
pip install -r requirements.txt

# 2. Train the Model
# Step A: Generate the dataset
python -m src.generate_dataset

# Step B: Train the model (This creates the file in the /models folder)
python -m src.churn_analysis

# 3. Run the Application
# To see the project in action, you need to open two separate terminal windows:

# Terminal 1 (Start the API):
uvicorn app.api:app --reload

# Terminal 2 (Start the Dashboard):
streamlit run app/app.py

# 4. Test the API
# Visit: http://127.0.0.1:8000/docs
```

---

## 📂 Component Breakdown

### 🛠️ `src/churn_analysis.py` — The ML Pipeline

This script orchestrates the entire machine learning workflow from raw data to actionable insights. It includes:

* **Data Cleaning:** Responsible for loading the raw CSV data, correcting data types, handling null values in 'TotalCharges' using logical imputation (tenure × monthly charges), and removing any duplicate entries.
*   **EDA (Exploratory Data Analysis):** Generates five publication-quality plots to provide deep insights into the dataset, including churn distribution, numeric feature histograms, categorical churn rates, correlation heatmaps, and violin plots.
*   **Feature Engineering:** Creates new predictive features such as tenure groups, service counts, cost-per-service ratios, and high-risk flags like `unprotected_internet`.
*   **Model Training:** Employs robust machine learning algorithms like Random Forest (or XGBoost) with Stratified 5-fold Cross-Validation. Evaluation focuses on a comprehensive set of metrics: Accuracy, Precision, Recall, F1-score, and ROC-AUC.
*   **Explainability:** Computes Permutation Importance to measure the true predictive value of features and generates SHAP summary plots for local interpretability, providing transparent insights into model predictions.

### 🧠 `app/api.py` — The Inference Engine (FastAPI)

This component serves as the high-performance backend for real-time churn probability predictions. It:

*   **Real-time Serving:** Acts as the central 'brain,' receiving customer data via JSON requests and returning a churn probability score.
*   **Feature Alignment:** Automatically handles essential preprocessing steps like one-hot encoding and scaling, ensuring that raw input data matches the format and requirements of the trained machine learning model.

### 🖥️ `app/app.py` — The Business Dashboard (Streamlit)

This interactive dashboard provides business users with a user-friendly interface to leverage the churn prediction model. It:

*   **Prediction Tab:** Features intuitive sliders and dropdowns that allow users to input customer parameters and trigger real-time risk assessments through the API.
*   **Business Logic:** Automatically suggests data-driven retention recommendations tailored to the calculated churn risk (classified as Low, Medium, or High), enabling targeted and effective customer outreach.

---


## 💬 Interview Talking Points

*   **Why ROC-AUC over accuracy?**
    *   Class imbalance (77% no churn / 23% churn) makes accuracy misleading; a model predicting "no churn" always gets 77% accuracy, which is not useful.

*   **Why prioritize Recall?**
    *   A false negative (missing a customer who is about to churn) is much more expensive than a false positive (offering a retention incentive to a customer who wasn't going to leave anyway).

*   **Why permutation importance?**
    *   Tree impurity importance can be biased toward high-cardinality features. Permutation importance measures the true predictive value by shuffling a feature's values and observing the impact on model performance.

*   **Why scale only for Logistic Regression?**
    *   Tree-based models (like Random Forest or Gradient Boosting) are scale-invariant, so `StandardScaler` adds no value and can reduce interpretability. Logistic Regression, however, benefits from feature scaling.

*   **How to handle imbalance?**
    *   The model uses `class_weight='balanced'` in Logistic Regression and Random Forest to ensure it pays extra attention to the minority class (churners). Other methods like SMOTE (Synthetic Minority Over-sampling Technique) from `imbalanced-learn` could also be considered.

*   **Why use FastAPI?**
    *   FastAPI was chosen for its high performance, enabling the model to be consumed by other systems (e.g., mobile apps, websites) independently of the Streamlit dashboard.

---

## 📈 Business Value & Impact

Churn prediction enables a **proactive retention strategy** by:

*   Score every customer monthly to identify high-risk accounts.
*   Route at-risk customers to retention specialists or trigger automated loyalty offers.
*   Measure lift vs. control groups in A/B tests to quantify ROI.

**Impact:** A 1% improvement in churn rate on a 1M-customer base (avg $50/mo) is worth $6M/year in protected revenue.

---

---

## 🤝 Contributing & Feedback
This project was built as a demonstration of a production-grade machine learning pipeline. If you have suggestions for improvement—such as adding **Docker** support, implementing **SHAP** for deeper explainability, or testing different architectures—feel free to open an issue or submit a pull request!

## 👋 Final Thoughts
Thank you for checking out this project! Building this end-to-end system was a deep dive into the intersection of **Data Science** and **Software Engineering**. I hope this serves as a useful template for anyone looking to bridge the gap between a Jupyter Notebook and a real-world application.

## 📬 Contact & Support
**Ghanendra Sahu**  🔗 [LinkedIn](https://www.linkedin.com/in/ghanendrasahu)

---

### 🌟 Thank You!
If you found this project helpful for your learning or interview preparation, please consider giving it a **Star**! ⭐️

**Happy Coding!** 🚀