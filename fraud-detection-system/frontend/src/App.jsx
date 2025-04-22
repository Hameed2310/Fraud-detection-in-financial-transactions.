import { useState } from "react";
import axios from "axios";

function App() {
  const [formData, setFormData] = useState({
    Amount: "",
    Transaction_Type: "Online Purchase",
    Location: "Mumbai",
    Card_Used: "Credit",
    Previous_Fraud_Reports: "",
    Merchant_Risk_Score: "",
    User_Age: "",
    Transaction_Frequency: "",
  });

  const [result, setResult] = useState(null);

  // Handle input changes
  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData);
      setResult(response.data);
    } catch (error) {
      console.error("Error:", error);
      setResult({ error: "Prediction failed. Please try again." });
    }
  };

  return (
    <div style={{ maxWidth: "500px", margin: "auto", padding: "20px", textAlign: "center" }}>
      <h2>Fraud Detection System</h2>
      <form onSubmit={handleSubmit}>
        <input type="number" name="Amount" placeholder="Transaction Amount" value={formData.Amount} onChange={handleChange} required />
        
        <select name="Transaction_Type" value={formData.Transaction_Type} onChange={handleChange}>
          <option>Online Purchase</option>
          <option>ATM Withdrawal</option>
          <option>Bank Transfer</option>
          <option>POS Payment</option>
        </select>

        <select name="Location" value={formData.Location} onChange={handleChange}>
          <option>Mumbai</option>
          <option>Delhi</option>
          <option>Bangalore</option>
          <option>Hyderabad</option>
        </select>

        <select name="Card_Used" value={formData.Card_Used} onChange={handleChange}>
          <option>Credit</option>
          <option>Debit</option>
        </select>

        <input type="number" name="Previous_Fraud_Reports" placeholder="Previous Fraud Reports" value={formData.Previous_Fraud_Reports} onChange={handleChange} required />
        <input type="number" name="Merchant_Risk_Score" placeholder="Merchant Risk Score (0-100)" value={formData.Merchant_Risk_Score} onChange={handleChange} required />
        <input type="number" name="User_Age" placeholder="User Age" value={formData.User_Age} onChange={handleChange} required />
        <input type="number" name="Transaction_Frequency" placeholder="Transaction Frequency" value={formData.Transaction_Frequency} onChange={handleChange} required />

        <button type="submit">Check Fraud</button>
      </form>

      {result && (
        <div style={{ marginTop: "20px", padding: "10px", border: "1px solid #ddd" }}>
          <h3>Result:</h3>
          {result.error ? (
            <p style={{ color: "red" }}>{result.error}</p>
          ) : (
            <>
              <p><strong>Fraud Prediction:</strong> {result.Fraud_Prediction === 1 ? "Fraudulent" : "Legitimate"}</p>
              <p><strong>Fraud Probability:</strong> {result.Fraud_Probability * 100}%</p>
            </>
          )}
        </div>
      )}
    </div>
  );
}

export default App;
