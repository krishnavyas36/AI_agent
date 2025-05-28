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

