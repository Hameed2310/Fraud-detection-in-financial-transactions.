import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os
import numpy as np

# Ensure the output directory exists
os.makedirs("./outputmodel", exist_ok=True)

# Load the preprocessed dataset
data_path = "./data/processed_fraud_detection_data.csv"
if not os.path.exists(data_path):
    raise FileNotFoundError(f"❌ Error: The dataset file '{data_path}' does not exist!")

df = pd.read_csv(data_path)

# Drop non-relevant columns if present
if "Transaction ID" in df.columns:
    df.drop(columns=["Transaction ID"], inplace=True)

# Handle missing values
for col in df.columns:
    if df[col].dtype == "object":
        df[col].fillna(df[col].mode()[0], inplace=True)  # Fill categorical missing values with mode
    else:
        df[col].fillna(df[col].median(), inplace=True)  # Fill numerical missing values with median

# Encode categorical features
categorical_cols = ["Location", "Transaction Type", "Device Used"]
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])  # Fit and transform column
    label_encoders[col] = le  # Store encoder for later use

# ✅ Save label encoders for API use
joblib.dump(label_encoders, "./outputmodel/label_encoders.pkl")

# Split features and target
X = df.drop(columns=["Fraud (1 = Yes, 0 = No)"])  # Features
y = df["Fraud (1 = Yes, 0 = No)"]  # Target variable

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Apply SMOTE to balance fraud cases
smote = SMOTE(sampling_strategy="auto", random_state=42)  # Auto mode balances fraud cases with non-fraud
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Print class distribution before and after SMOTE
print("✅ Before SMOTE:", y_train.value_counts().to_dict())
print("✅ After SMOTE:", y_train_balanced.value_counts().to_dict())

# Compute scale_pos_weight dynamically
fraud_count = sum(y_train_balanced == 1)
non_fraud_count = sum(y_train_balanced == 0)
scale_pos_weight = non_fraud_count / fraud_count  # Adjust weight dynamically

# Initialize XGBoost classifier with improved parameters
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="aucpr",  # Better for imbalanced datasets
    use_label_encoder=False,
    n_estimators=300,
    learning_rate=0.05,
    max_depth=8,
    scale_pos_weight=scale_pos_weight  # Dynamically calculated
)

# Train the model on the balanced dataset
xgb_model.fit(X_train_balanced, y_train_balanced)

# ✅ Save the trained model
xgb_model.save_model("./outputmodel/xgboost_fraud_model.json")  # Save in JSON format
joblib.dump(xgb_model, "./outputmodel/xgboost_fraud_model.pkl")  # Save as .pkl

print("✅ Model training completed with SMOTE & dynamic class weighting. Model saved in ./outputmodel/")
