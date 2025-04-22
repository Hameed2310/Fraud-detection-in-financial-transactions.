import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the preprocessed dataset
df = pd.read_csv("./data/processed_fraud_detection_data.csv")  # Update with actual path

# Drop non-relevant columns
df = df.drop(columns=["Transaction ID"])  # Drop Transaction ID since it's not useful for testing

# Encode categorical features (Ensure encoding matches training)
categorical_cols = ["Location", "Transaction Type", "Device Used"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Fit and transform
    label_encoders[col] = le  # Store encoder for consistency

# Split features and target
X = df.drop(columns=["Fraud (1 = Yes, 0 = No)"])  # Features
y = df["Fraud (1 = Yes, 0 = No)"]  # Target variable

# Split data into training and testing sets
_, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Load trained model
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("./outputmodel/xgboost_fraud_model.json")

# Make predictions
y_pred = xgb_model.predict(X_test)

# Save predictions
test_results = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
test_results.to_csv("./outputmodel/test_predictions.csv", index=False)
print("✅ Predictions saved in test_predictions.csv")
