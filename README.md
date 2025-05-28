# AI_agent
# 🧠 AI Logistics Optimization Agent

This is an AI-powered logistics decision-support tool that recommends the optimal delivery mode for shipments based on distance, weight, and urgency. Built using Streamlit and scikit-learn, the app combines machine learning, business rules, and real-time UI interaction to create a self-improving logistics agent.

---

## 🚀 Features

- ✅ Predicts the best delivery mode (LTL, TL, Drayage, Transload)
- ✅ Estimates cost and ETA based on inputs
- ✅ Shows model confidence with visual charts
- ✅ Interactive Streamlit interface
- ✅ Logs every decision for traceability
- ✅ Lets users view, filter, and download decision logs
- ✅ Supports model retraining from historical decisions

---

## 📊 Technologies Used

- `Python`
- `scikit-learn` (Random Forest, Label Encoding)
- `pandas` and `numpy`
- `matplotlib` / `seaborn`
- `Streamlit` (for the interactive UI)

---

## 📂 How to Run

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt


2. ** to Run APP **
streamlit run agent.py



.
├── agent.py                # Main app
├── shipments_full.csv      # Initial training data
├── agent_decision_log.csv  # Auto-generated log
├── README.md
├── requirements.txt

✅ How This Tool Works as an AI-Powered Logistics Agent
1. Inputs from Real-World Context
Users provide:

Distance (km)

Weight (kg)

Urgency level (Low, Medium, High)

🔁 These map to real shipment attributes you'd have in a supply chain.

2. AI-Based Mode Prediction
It uses a machine learning model (RandomForestClassifier) trained on historical data (shipments_full.csv) to learn:

Which delivery mode (LTL, TL, Drayage, Transload) works best for which combinations of inputs.

🧠 The model identifies non-obvious relationships, outperforming fixed business rules.

3. Business Rule Integration
Before ML kicks in, a fallback rule-based logic (recommend_mode) provides a reliable default — combining expert systems + machine learning.

⚖️ This hybrid approach ensures resilience in low-data situations.

4. Estimates Cost & ETA
The agent simulates realistic decision output with:

Estimated cost based on distance × mode-based rates

Estimated delivery time (ETA) per mode

'TL' → 'Truckload (TL)'

'LTL' → 'Less-Than-Truckload (LTL)'

'Drayage' → 'Drayage (short-distance)'

'Transload' → 'Transload (intermodal)'

📊 This mimics real decision-support tools used by logistics planners.

5. Confidence Scores & Explanation
It shows:

Model confidence using probability outputs

Bar chart of all mode probabilities, improving transparency

🎯 Users can assess how certain the model is, and override when confidence is low.

6. Continuous Learning (Retraining from Logs)
Logged decisions (agent_decision_log.csv) are used to:

Retrain the model using user interaction history

Make the agent smarter over time

♻️ This creates a self-learning feedback loop — a core element of AI agents.

7. User Interface via Streamlit
A user-friendly dashboard where users:

Interact with the agent

Log and filter decisions

Download logs

Retrain the model

🖥️ This UI turns raw ML logic into a usable AI system.

🤖 TL;DR – Why It’s an AI Agent
Capability	How It’s Achieved
Intelligent decisions	ML model trained on shipment data
Adaptation over time	Retraining from decision logs
User feedback integration	Logging and filtered viewing
Transparent decision-making	Confidence scores and explainability
Hybrid intelligence	Combines business rules + ML

