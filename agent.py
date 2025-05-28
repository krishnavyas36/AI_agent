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
    return clf, le_urgency, le_mode

def estimate_cost_eta(distance, weight, mode):
    base_cost_per_km = {'LTL': 0.5, 'TL': 0.8, 'Drayage': 0.9, 'Transload': 0.6}
    eta_by_mode = {'LTL': 4, 'TL': 2, 'Drayage': 3, 'Transload': 5}
    cost = distance * base_cost_per_km.get(mode, 0.7) + (weight * 0.1)
    eta = eta_by_mode.get(mode, 3)
    return round(cost, 2), eta

def streamlit_agent_interface():
    st.title("AI Logistics Agent")

    filepath = r"C:\Users\dkris\OneDrive\Desktop\New folder\shipments_full.csv"
    df = load_data(filepath)
    if df is not None:
        clf, le_urgency, le_mode = train_ml_model(df)

        st.sidebar.header("Shipment Input")
        distance = st.sidebar.number_input("Distance (km)", min_value=0.0, value=500.0)
        weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, value=1000.0)
        urgency = st.sidebar.selectbox("Urgency", ['Low', 'Medium', 'High'])

        if st.sidebar.button("Get Recommendation"):
            urgency_code = le_urgency.transform([urgency])[0]
            features = pd.DataFrame([[distance, weight, urgency_code]], columns=['Distance_km', 'Weight_kg', 'Urgency_Code'])
            pred_code = clf.predict(features)[0]
            mode = le_mode.inverse_transform([pred_code])[0]
            cost, eta = estimate_cost_eta(distance, weight, mode)

            st.success(f"Recommended Mode: {mode}")
            st.info(f"Estimated Cost: ${cost}")
            st.info(f"Estimated ETA: {eta} days")

if __name__ == "__main__":
    streamlit_agent_interface()