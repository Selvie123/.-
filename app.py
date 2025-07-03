import streamlit as st
import pandas as pd
import smtplib
from sklearn.ensemble import RandomForestClassifier
import joblib

st.title("🏥 Hospital Risk Alert System")

uploaded_file = st.file_uploader("Upload Patient CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Patient Data", df)

    model = joblib.load("model/risk_model.pkl")
    predictions = model.predict(df.drop("Risk", axis=1))
    df['Prediction'] = predictions
    st.write("Predictions", df)

    high_risk_patients = df[df['Prediction'] == 1]
    if not high_risk_patients.empty:
        st.warning("High-risk patients detected! Sending alerts...")
        high_risk_patients.to_csv("output/high_risk_alerts.csv", index=False)
        st.success("Alerts saved to output/high_risk_alerts.csv")