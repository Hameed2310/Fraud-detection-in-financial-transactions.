import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score
import pickle

# Load dataset
df = pd.read_csv("dataset.csv")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()

# Rename columns to standard format for consistency
df.rename(columns={"Amount ($)": "Amount", "Time (Seconds Since Last Transaction)": "Time"}, inplace=True)

# Remove extra spaces from string values
df = df.applymap(lambda x: x.strip() if isinstance(x, str) else x)

# Check if "Fraud" column exists
if "Fraud" not in df.columns:
    raise KeyError("🚨 'Fraud' column not found! Please check your dataset.")

# Handle missing values
df = df.dropna()  # Removes rows with missing values

# Encode categorical variables
label_encoders = {}
for col in ["Location", "Transaction Type", "Device Used"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for later use

# Features & Target
X = df.drop(columns=["Transaction ID", "Fraud"])  # Drop ID column
y = df["Fraud"]

# Check dataset balance
fraud_ratio = y.mean()
print(f"⚖️ Fraud cases in dataset: {fraud_ratio * 100:.2f}%")

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train XGBoost Model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss", n_estimators=100, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate Model on test data
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
test_f1 = f1_score(y_test, y_pred)
print(f"✅ Test Accuracy: {test_accuracy * 100:.2f}% | F1-Score: {test_f1:.2f}")

# Evaluate Model on train data (to check for overfitting)
x_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, x_pred)
train_f1 = f1_score(y_train, x_pred)
print(f"✅ Train Accuracy: {train_accuracy * 100:.2f}% | F1-Score: {train_f1:.2f}")

# Save Model & Encoders
with open("fraud_model.pkl", "wb") as f:
    pickle.dump((model, label_encoders), f)

print("✅ Model & Encoders Saved as 'fraud_model.pkl'")
