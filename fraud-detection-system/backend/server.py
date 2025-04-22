from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import xgboost as xgb

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS to fix frontend connection issues

# Load trained model
model = joblib.load("../model/outputmodel/xgboost_fraud_model.pkl")

# Load saved label encoders
label_encoders = joblib.load("../model/outputmodel/label_encoders.pkl")

# Define categorical columns
categorical_cols = ["Location", "Transaction Type", "Device Used"]

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        df = pd.DataFrame([data])  # Convert input to DataFrame

        # Encode categorical features using the saved label encoders
        for col in categorical_cols:
            if col in df.columns and col in label_encoders:
                df[col] = df[col].apply(lambda x: label_encoders[col].transform([x])[0] 
                                        if x in label_encoders[col].classes_ else -1)

        # Drop Transaction ID if present
        df = df.drop(columns=["Transaction ID"], errors="ignore")

        # Predict fraud probability
        pred_prob = model.predict_proba(df)[:, 1]  # Probability of fraud
        threshold = 0.01  # Low threshold to always predict fraud
        fraud_prediction = int(pred_prob > threshold)

        # Return response
        return jsonify({"fraud_prediction": fraud_prediction})

    except Exception as e:
        return jsonify({"error": str(e)})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
