import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load test predictions
test_results = pd.read_csv("./outputmodel/test_predictions.csv")

# Extract actual and predicted values
y_test = test_results["Actual"]
y_pred = test_results["Predicted"]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Print evaluation results
print(f"✅ Model Accuracy: {accuracy:.4f}")
print("📊 Confusion Matrix:\n", conf_matrix)
print("📈 Classification Report:\n", class_report)
