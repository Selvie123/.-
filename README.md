# 🏥 Hospital Risk Alert System

This project uses a machine learning model (Random Forest) to predict whether hospital patients are **high-risk** based on their medical data. If high-risk patients are detected, the system **saves alerts** and can be extended to send notifications via email/SMS.

## 📌 Features

- ✅ Upload patient data in CSV format
- 🤖 Predict high-risk patients using a trained ML model (Random Forest)
- 📊 Streamlit GUI for ease of use
- 📩 Generate alerts for high-risk cases (email/SMS extension)
- 🗃️ Stores prediction results in `output/high_risk_alerts.csv`

---

## 📁 Project Structure

Hospital_Risk_Alert/
│
├── app.py # Streamlit interface
├── hospital_data.csv # Sample training data
├── model/
│ ├── train_model.py # Model training script
│ └── risk_model.pkl # Trained ML model (generated after running training)
├── output/
│ └── high_risk_alerts.csv # Alerts generated for high-risk patients
└── README.md # Project overview (this file)

