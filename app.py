import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go

# Load model and features
model = joblib.load("top4_churn_model.pkl")
features = joblib.load("top4_features.pkl")

st.set_page_config(page_title="Customer Churn Prediction", layout="centered")
st.title("📉 Customer Churn Prediction App")
st.subheader("Using Top 4 Features")

# Input form
st.write("Enter the following customer details:")

monthly_charges = st.number_input("Monthly Charges", min_value=0.0, step=1.0)
tenure = st.number_input("Tenure (months)", min_value=0, step=1)
total_charges = st.number_input("Total Charges", min_value=0.0, step=1.0)
contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

# Encode contract type
contract_mapping = {
    "Month-to-month": 0,
    "One year": 1,
    "Two year": 2
}
contract_encoded = contract_mapping[contract_type]

# Predict button
if st.button("Predict Churn"):
    # Prepare input
    input_df = pd.DataFrame([{
        "MonthlyCharges": monthly_charges,
        "tenure": tenure,
        "TotalCharges": total_charges,
        "Contract": contract_encoded
    }])

    # Make prediction
    pred = model.predict(input_df)[0]
    pred_proba = model.predict_proba(input_df)[0][1]

    # Color-coded card (HTML)
    churn_label = "Churn" if pred else "No Churn"
    color = "#ff4b4b" if pred else "#28a745"

    st.markdown(
        f"""
        <div style="padding: 1rem; background-color:{color}; color:white; border-radius:10px; text-align:center;">
            <h3>Prediction: {churn_label}</h3>
            <p>Churn Probability: {pred_proba:.2%}</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Gauge Chart using Plotly
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pred_proba * 100,
        delta={'reference': 50, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgreen"},
                {'range': [50, 100], 'color': "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': pred_proba * 100
            }
        },
        title={'text': "Churn Probability (%)"}
    ))

    st.plotly_chart(fig)
