import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the preprocessed dataset
df = pd.read_csv("./data/processed_fraud_detection_data.csv")  # Update with your file path if needed

# Set plot style
sns.set_style("whitegrid")

# Plot distribution of Amount ($)
plt.figure(figsize=(8, 4))
sns.histplot(df["Amount ($)"], bins=50, kde=True, color="blue")
plt.title("Distribution of Transaction Amounts")
plt.xlabel("Amount ($)")
plt.ylabel("Frequency")
plt.savefig("./image/amount_distribution.png")  # Save plot
plt.close()

# Plot distribution of Time (Seconds Since Last Transaction)
plt.figure(figsize=(8, 4))
sns.histplot(df["Time (Seconds Since Last Transaction)"], bins=50, kde=True, color="green")
plt.title("Distribution of Transaction Time Gaps")
plt.xlabel("Time (Seconds)")
plt.ylabel("Frequency")
plt.savefig("./image/time_distribution.png")  # Save plot
plt.close()

# Countplot of Fraud vs. Non-Fraud Transactions
plt.figure(figsize=(6, 4))
sns.countplot(x=df["Fraud (1 = Yes, 0 = No)"], palette="pastel")
plt.title("Count of Fraudulent and Non-Fraudulent Transactions")
plt.xlabel("Fraud (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.savefig("./image/fraud_count.png")  # Save plot
plt.close()

# Boxplot of Amount by Fraud Category
plt.figure(figsize=(8, 4))
sns.boxplot(x=df["Fraud (1 = Yes, 0 = No)"], y=df["Amount ($)"], palette="coolwarm")
plt.title("Transaction Amounts by Fraud Category")
plt.xlabel("Fraud (0 = No, 1 = Yes)")
plt.ylabel("Amount ($)")
plt.yscale("log")  # Log scale to better visualize outliers
plt.savefig("./image/fraud_amount_boxplot.png")  # Save plot
plt.close()
