import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("top4_churn_model.pkl")
features = joblib.load("top4_features.pkl")

st.title("📉 Customer Churn Prediction App")
st.subheader("Using Top 4 Features")

# Input form
st.write("Enter the following customer details:")

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)

contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Encode contract type same way as in training
contract_mapping = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}

contract_encoded = contract_mapping[contract_type]

# Predict button
if st.button("Predict Churn"):
    input_df = pd.DataFrame([{
        "MonthlyCharges": monthly_charges,
        "tenure": tenure,
        "TotalCharges": total_charges,
        "Contract": contract_encoded
    }])
    
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    st.markdown(f"### 🔍 Prediction: **{'Churn' if pred else 'No Churn'}**")
    st.markdown(f"**Churn Probability:** {pred_proba:.2%}")