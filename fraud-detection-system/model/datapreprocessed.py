import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

# Load dataset
file_path = "./data/fraud_detection_dataset.csv"  # Update with actual path
df = pd.read_csv(file_path)

# Handling missing values (if any)
imputer = SimpleImputer(strategy="most_frequent")
df[:] = imputer.fit_transform(df)

# Encode categorical variables
label_encoders = {}
categorical_cols = ["Location", "Transaction Type", "Device Used"]

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future use

# Scale numerical features
scaler = StandardScaler()
numerical_cols = ["Amount ($)", "Time (Seconds Since Last Transaction)"]
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save preprocessed data
processed_file_path = "./data/processed_fraud_detection_data.csv"
df.to_csv(processed_file_path, index=False)
print(f"Processed data saved to {processed_file_path}")
