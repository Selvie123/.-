import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("hospital_data.csv")
X = df.drop("Risk", axis=1)
y = df["Risk"]

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model/risk_model.pkl")
print("Model saved.")