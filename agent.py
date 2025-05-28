import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None


def recommend_mode(row):
    if row['Urgency'] == 'High' and row['Weight_kg'] > 700:
        return 'TL'
    elif row['Urgency'] == 'High':
        return 'Drayage'
    elif row['Urgency'] == 'Medium' and row['Distance_km'] > 1000:
        return 'Transload'
    elif row['Weight_kg'] <= 500:
        return 'LTL'
    else:
        return 'TL'


def train_ml_model(df):
    df = df.copy()
    le_urgency = LabelEncoder()
    df['Urgency_Code'] = le_urgency.fit_transform(df['Urgency'])
    le_mode = LabelEncoder()
    df['Mode_Code'] = le_mode.fit_transform(df['Mode'])
    X = df[['Distance_km', 'Weight_kg', 'Urgency_Code']]
    y = df['Mode_Code']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(class_weight='balanced', random_state=42)
    clf.fit(X_train, y_train)
    return clf, le_urgency, le_mode, X_test, y_test


def estimate_cost_eta(distance, weight, mode):
    base_cost_per_km = {'LTL': 0.5, 'TL': 0.8, 'Drayage': 0.9, 'Transload': 0.6}
    eta_by_mode = {'LTL': 4, 'TL': 2, 'Drayage': 3, 'Transload': 5}
    cost = distance * base_cost_per_km.get(mode, 0.7) + (weight * 0.1)
    eta = eta_by_mode.get(mode, 3)
    return round(cost, 2), eta


def load_log_data(log_path):
    if os.path.exists(log_path):
        return pd.read_csv(log_path)
    else:
        return pd.DataFrame()


def streamlit_agent_interface():
    st.set_page_config(page_title="AI Logistics Agent", layout="wide")
    st.title("AI Logistics Agent")

    tab1, tab2 = st.tabs(["Predict Shipment", "View Decision Log"])
    filepath = r"C:\\Users\\dkris\\Downloads\\shipments_full.csv"
    df = load_data(filepath)

    if df is not None:
        clf, le_urgency, le_mode, X_test, y_test = train_ml_model(df)

        with tab1:
            st.sidebar.header("Shipment Input")
            distance_miles = st.sidebar.number_input("Distance (miles)", min_value=0.0, value=500.0)
            distance = distance_miles * 1.60934  # convert to km
            weight_lbs = st.sidebar.number_input("Weight (lbs)", min_value=0.0, value=1000.0)
            weight = weight_lbs * 0.453592  # convert to kg
            urgency = st.sidebar.selectbox("Urgency", ['Low', 'Medium', 'High'])

            if st.sidebar.button("Get Recommendation"):
                urgency_code = le_urgency.transform([urgency])[0]
                features = pd.DataFrame([[distance, weight, urgency_code]], columns=['Distance_km', 'Weight_kg', 'Urgency_Code'])
                probas = clf.predict_proba(features)[0]
                class_probs = {label: round(probas[i] * 100, 2) for i, label in enumerate(le_mode.classes_)}
                urgency_code = le_urgency.transform([urgency])[0]
                features = pd.DataFrame([[distance, weight, urgency_code]], columns=['Distance_km', 'Weight_kg', 'Urgency_Code'])
                pred_code = clf.predict(features)[0]
                mode = le_mode.inverse_transform([pred_code])[0]
                cost, eta = estimate_cost_eta(distance, weight, mode)

                st.success(f"Recommended Mode: {mode} ({class_probs[mode]}% confidence)")
                st.info(f"Estimated Cost: ${cost}")
                st.info(f"Estimated ETA: {eta} days")

                st.markdown("### Prediction Confidence")
                st.bar_chart(pd.DataFrame([class_probs]))

                log_file = "agent_decision_log.csv"
                with open(log_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    if os.path.getsize(log_file) == 0:
                        writer.writerow(['Distance_km', 'Weight_kg', 'Urgency', 'Recommended_Mode', 'Estimated_Cost', 'Estimated_ETA_days'])
                    writer.writerow([distance, weight, urgency, mode, cost, eta])

        with tab2:
            st.subheader("Decision Log")
            log_df = load_log_data("agent_decision_log.csv")
            if not log_df.empty:
                selected_urgency = st.selectbox("Filter by Urgency", options=['All'] + sorted(log_df['Urgency'].unique()))
                filtered_df = log_df if selected_urgency == 'All' else log_df[log_df['Urgency'] == selected_urgency]

                st.dataframe(filtered_df)
                st.bar_chart(filtered_df['Recommended_Mode'].value_counts())

                csv_buffer = io.StringIO()
                filtered_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="Download Filtered Log as CSV",
                    data=csv_buffer.getvalue(),
                    file_name="filtered_decision_log.csv",
                    mime="text/csv"
                )

                #  Retraining trigger from user log
                if st.button("Retrain Model from Decision Log"):
                    try:
                        retrain_df = log_df.rename(columns={
                            'Recommended_Mode': 'Mode'
                        })[['Distance_km', 'Weight_kg', 'Urgency', 'Mode']]
                        clf, le_urgency, le_mode, X_test, y_test = train_ml_model(retrain_df)
                        st.success("Model successfully retrained using decision log!")
                    except Exception as e:
                        st.error(f"Retraining failed: {e}")
            else:
                st.info("No decisions logged yet.")


if __name__ == "__main__":
    streamlit_agent_interface()
