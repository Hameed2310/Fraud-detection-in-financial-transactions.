import streamlit as st
import pandas as pd
import pickle
import random

# Load model and encoders
with open("fraud_model.pkl", "rb") as f:
    model, label_encoders = pickle.import streamlit as st
import pandas as pd
import pickle
import random

# Load model and encoders
with open("fraud_model.pkl", "rb") as f:
    model, label_encoders = pickle.load(f)

st.title("💳 Fraud Detection System")
st.write("Enter transaction details below to detect potential fraud.")

# User Inputs
amount = st.number_input("💰 Transaction Amount ($)", min_value=1)
time = st.number_input("⏳ Time Since Last Transaction (seconds)", min_value=1)

location = st.selectbox("📍 Transaction Location", ["Select"] + list(label_encoders["Location"].classes_))
transaction_type = st.selectbox("💼 Transaction Type", ["Select"] + list(label_encoders["Transaction Type"].classes_))
device_used = st.selectbox("📱 Device Used", ["Select"] + list(label_encoders["Device Used"].classes_))

# Simulate new device/location (could be user profile-based in future)
is_new_device = st.checkbox("🔧 Use of New/Unrecognized Device?")
is_new_location = st.checkbox("📍 New Location for This User?")

# Predict Button
if st.button("🔍 Predict Fraud"):
    if location == "Select" or transaction_type == "Select" or device_used == "Select":
        st.warning("⚠️ Please select all transaction fields.")
    else:
        # Encode categorical inputs
        loc_enc = label_encoders["Location"].transform([location])[0]
        type_enc = label_encoders["Transaction Type"].transform([transaction_type])[0]
        device_enc = label_encoders["Device Used"].transform([device_used])[0]

        input_data = pd.DataFrame([[amount, time, loc_enc, type_enc, device_enc]],
                                  columns=["Amount", "Time", "Location", "Transaction Type", "Device Used"])

        # Get model prediction and probability
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # Probability of fraud

        # 🔍 Real-World Fraud Logic
        fraud_flags = []

        # Rule: Very large amount
        if amount > 100_000:
            fraud_flags.append("💰 High transaction amount")

        # Rule: Unrealistically fast transaction
        if time < 2:
            fraud_flags.append("⏱️ Extremely fast transaction")

        # Rule: New device used
        if is_new_device:
            fraud_flags.append("📱 Unrecognized device")

        # Rule: New or risky location
        if is_new_location:
            fraud_flags.append("📍 Unusual transaction location")

        # Final fraud decision logic
        if prediction == 1 and (proba >= 0.7 or len(fraud_flags) > 0):
            st.error("🚨 Fraud Detected!")
            if fraud_flags:
                st.write("⚠️ Flags triggered:")
                for flag in fraud_flags:
                    st.write(f"- {flag}")
        elif len(fraud_flags) >= 2:
            st.warning("⚠️ Suspicious Transaction (Manual Review Suggested)")
            st.write("Triggered rules:")
            for flag in fraud_flags:
                st.write(f"- {flag}")
        else:
            st.success("✅ Transaction is Safe.")

        # Show probability
        st.info(f"🔎 Model Fraud Probability: **{proba:.2f}**")load(f)

st.title("💳 Fraud Detection System")
st.write("Enter transaction details below to detect potential fraud.")

# User Inputs
amount = st.number_input("💰 Transaction Amount ($)", min_value=1)
time = st.number_input("⏳ Time Since Last Transaction (seconds)", min_value=1)

location = st.selectbox("📍 Transaction Location", ["Select"] + list(label_encoders["Location"].classes_))
transaction_type = st.selectbox("💼 Transaction Type", ["Select"] + list(label_encoders["Transaction Type"].classes_))
device_used = st.selectbox("📱 Device Used", ["Select"] + list(label_encoders["Device Used"].classes_))

# Simulate new device/location (could be user profile-based in future)
is_new_device = st.checkbox("🔧 Use of New/Unrecognized Device?")
is_new_location = st.checkbox("📍 New Location for This User?")

# Predict Button
if st.button("🔍 Predict Fraud"):
    if location == "Select" or transaction_type == "Select" or device_used == "Select":
        st.warning("⚠️ Please select all transaction fields.")
    else:
        # Encode categorical inputs
        loc_enc = label_encoders["Location"].transform([location])[0]
        type_enc = label_encoders["Transaction Type"].transform([transaction_type])[0]
        device_enc = label_encoders["Device Used"].transform([device_used])[0]

        input_data = pd.DataFrame([[amount, time, loc_enc, type_enc, device_enc]],
                                  columns=["Amount", "Time", "Location", "Transaction Type", "Device Used"])

        # Get model prediction and probability
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]  # Probability of fraud

        # 🔍 Real-World Fraud Logic
        fraud_flags = []

        # Rule: Very large amount
        if amount > 100_000:
            fraud_flags.append("💰 High transaction amount")

        # Rule: Unrealistically fast transaction
        if time < 2:
            fraud_flags.append("⏱️ Extremely fast transaction")

        # Rule: New device used
        if is_new_device:
            fraud_flags.append("📱 Unrecognized device")

        # Rule: New or risky location
        if is_new_location:
            fraud_flags.append("📍 Unusual transaction location")

        # Final fraud decision logic
        if prediction == 1 and (proba >= 0.7 or len(fraud_flags) > 0):
            st.error("🚨 Fraud Detected!")
            if fraud_flags:
                st.write("⚠️ Flags triggered:")
                for flag in fraud_flags:
                    st.write(f"- {flag}")
        elif len(fraud_flags) >= 2:
            st.warning("⚠️ Suspicious Transaction (Manual Review Suggested)")
            st.write("Triggered rules:")
            for flag in fraud_flags:
                st.write(f"- {flag}")
        else:
            st.success("✅ Transaction is Safe.")

        # Show probability
        st.info(f"🔎 Model Fraud Probability: **{proba:.2f}**")
