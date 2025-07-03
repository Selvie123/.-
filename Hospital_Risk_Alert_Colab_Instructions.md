✅ Step 1: Install Required Libraries
!pip install pandas scikit-learn joblib streamlit -q
✅ Step 2: Create Project Folders
import os
os.makedirs("Hospital_Risk_Alert/model", exist_ok=True)
os.makedirs("Hospital_Risk_Alert/output", exist_ok=True)
os.chdir("Hospital_Risk_Alert")
✅ Step 3: Create Sample Dataset
sample_data = """Age,BP,Cholesterol,Diabetes,Risk
65,130,250,1,1
45,120,190,0,0
70,150,280,1,1
30,110,180,0,0"""

with open("hospital_data.csv", "w") as f:
    f.write(sample_data)

print("✅ Sample data created: hospital_data.csv")
✅ Step 4: Train the ML Model
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("hospital_data.csv")
X = df.drop("Risk", axis=1)
y = df["Risk"]

model = RandomForestClassifier()
model.fit(X, y)
joblib.dump(model, "model/risk_model.pkl")
print("✅ Model trained and saved to model/risk_model.pkl")
✅ Step 5: Upload New Patient Data & Predict
from google.colab import files
import io

uploaded = files.upload()

for file_name in uploaded.keys():
    test_df = pd.read_csv(io.BytesIO(uploaded[file_name]))

print("🩺 Uploaded Patient Data:")
print(test_df)

# Drop Risk if present
if 'Risk' in test_df.columns:
    test_df = test_df.drop('Risk', axis=1)

model = joblib.load("model/risk_model.pkl")
predictions = model.predict(test_df)
test_df['Prediction'] = predictions

print("\n🔍 Predictions:")
print(test_df)

high_risk = test_df[test_df['Prediction'] == 1]
high_risk.to_csv("output/high_risk_alerts.csv", index=False)
print("\n⚠️ High-risk patients saved to output/high_risk_alerts.csv")

Sample Input CSV (for upload)
Age,BP,Cholesterol,Diabetes
68,145,270,1
32,115,185,0
