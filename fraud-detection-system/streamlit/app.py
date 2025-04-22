import streamlit as st
import pandas as pd
import pickle

# Load trained model & encoders
with open("fraud_model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

# Streamlit UI
st.title("💳 Fraud Detection System")
st.write("Enter transaction details below to predict if it's fraudulent.")

# User Inputs with Validation
amount = st.number_input("💰 Transaction Amount ($)", min_value=1, help="Amount must be greater than zero.")
time = st.number_input("⏳ Time Since Last Transaction (seconds)", min_value=1, help="Time must be at least 1 second.")

location = st.selectbox("📍 Transaction Location", ["Select"] + list(label_encoders["Location"].classes_))
transaction_type = st.selectbox("💼 Transaction Type", ["Select"] + list(label_encoders["Transaction Type"].classes_))
device_used = st.selectbox("📱 Device Used", ["Select"] + list(label_encoders["Device Used"].classes_))

# Predict Button
if st.button("🔍 Predict Fraud"):
    # 🚨 Input Validations
    if location == "Select" or transaction_type == "Select" or device_used == "Select":
        st.warning("⚠️ Please select a valid Location, Transaction Type, and Device Used.")
    else:
        # Convert categorical inputs using label encoders
        location_encoded = label_encoders["Location"].transform([location])[0]
        transaction_type_encoded = label_encoders["Transaction Type"].transform([transaction_type])[0]
        device_used_encoded = label_encoders["Device Used"].transform([device_used])[0]

        # Prepare input data with matching column names
        input_data = pd.DataFrame([[amount, time, location_encoded, transaction_type_encoded, device_used_encoded]],
                                  columns=["Amount", "Time", "Location", "Transaction Type", "Device Used"])  # Ensure column names match training

         # Make prediction
        prediction = model.predict(input_data)[0]

        # 🚨 Flag High-Risk Transactions Immediately
        if amount >= 1_000_000_000 and time <= 2:
            st.error("🚨 Fraud Detected! Suspiciously High Transaction in Unrealistic Time.")
        else:
            # Display Result
            if prediction == 1:
                st.error("🚨 Fraud Detected!")
            else:
                st.success("✅ Transaction is Safe!")
