import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import altair as alt
import plotly.express as px

# Load model metadata and scaler
model_df = pd.read_csv("top_models.csv")
scaler = joblib.load("scaler.joblib")

# Set Streamlit page configuration
st.set_page_config(page_title="AutoML Loan Predictor", layout="wide")
st.title("🔮 AutoML Loan Approval Predictor")
st.sidebar.title("Navigation")

# Navigation menu
page = st.sidebar.radio("Go to", ["🏠 Model Dashboard", "📥 Make Predictions"])

# Dashboard: Model Performance
if page == "🏠 Model Dashboard":
    st.subheader("📊 Model Performance Summary")

    st.dataframe(model_df, use_container_width=True)

    # Metrics Conversion
    metric_cols = ["F1_Score", "Accuracy", "Precision", "Recall"]
    for col in metric_cols:
        model_df[col] = pd.to_numeric(model_df[col], errors='coerce')

    # Pie Chart Only
    st.markdown("### 🥧 F1 Score Share")
    pie_chart = px.pie(
        model_df,
        names="Model_Name",
        values="F1_Score",
        title="F1 Score Contribution by Model",
        color_discrete_sequence=px.colors.sequential.RdBu
    )
    st.plotly_chart(pie_chart, use_container_width=True)

    # Best model
    best_model = model_df.sort_values("F1_Score", ascending=False).iloc[0]
    st.success(f"🏆 Best Model: **{best_model['Model_Name']}** (F1 Score: {best_model['F1_Score']:.4f})")

# Prediction View
elif page == "📥 Make Predictions":
    st.subheader("🧾 Manual Loan Application Prediction")

    # Select model
    model_choice = st.selectbox("Select a model", model_df["Model_Name"])
    model_path = model_df[model_df["Model_Name"] == model_choice]["Model_File"].values[0]

    if not os.path.exists(model_path):
        st.error(f"Model file not found: {model_path}")
        st.stop()

    model = joblib.load(model_path)

    with st.form("predict_form"):
        person_age = st.slider("Age", 18, 100, 30)
        person_gender = st.selectbox("Gender", ["male", "female"])
        person_education = st.selectbox("Education", ["High School", "Bachelor", "Master"])
        person_income = st.number_input("Monthly Income ($)", 0.0, 200000.0, 5000.0)
        person_emp_exp = st.number_input("Employment Experience (Years)", 0.0, 40.0, 3.0)
        person_home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE"])
        loan_amnt = st.number_input("Loan Amount ($)", 1000.0, 100000.0, 5000.0)
        loan_intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL"])
        loan_int_rate = st.number_input("Interest Rate (%)", 0.0, 40.0, 12.0)
        loan_percent_income = st.number_input("Loan % of Income", 0.0, 1.0, 0.3)
        cb_person_cred_hist_length = st.number_input("Credit History Length (Years)", 0, 50, 3)
        credit_score = st.number_input("Credit Score", 300.0, 850.0, 600.0)
        previous_loan_defaults_on_file = st.selectbox("Previous Loan Defaults?", ["Yes", "No"])

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_data = {
            "person_age": person_age,
            "person_gender": 1 if person_gender == "male" else 0,
            "person_education": {"High School": 0, "Bachelor": 1, "Master": 2}[person_education],
            "person_income": person_income,
            "person_emp_exp": person_emp_exp,
            "person_home_ownership": {"RENT": 0, "OWN": 1, "MORTGAGE": 2}[person_home_ownership],
            "loan_amnt": loan_amnt,
            "loan_intent": {"PERSONAL": 0, "EDUCATION": 1, "MEDICAL": 2}[loan_intent],
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": 1 if previous_loan_defaults_on_file == "Yes" else 0
        }

        expected_cols = [
            "person_age", "person_gender", "person_education", "person_income",
            "person_emp_exp", "person_home_ownership", "loan_amnt", "loan_intent",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
            "credit_score", "previous_loan_defaults_on_file"
        ]

        numerical_cols = [
            "person_age", "person_income", "person_emp_exp", "loan_amnt",
            "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length",
            "credit_score"
        ]

        input_df = pd.DataFrame([input_data])[expected_cols]
        input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])

        prediction = model.predict(input_df)[0]
        st.success("✅ Loan Approved!" if prediction == 1 else "❌ Loan Rejected.")
