# logistics_agent/main.py (Step 10: Add basic data visualization of logs)

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import csv
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        print("\nSample Data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
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

def apply_recommendations(df):
    df['Recommended_Mode'] = df.apply(recommend_mode, axis=1)
    return df

def evaluate_recommendations(df):
    if 'Mode' in df.columns and 'Recommended_Mode' in df.columns:
        print("\nEvaluation Report:")
        accuracy = accuracy_score(df['Mode'], df['Recommended_Mode'])
        print(f"Accuracy: {accuracy:.2f}")
        print("\nClassification Report:")
        print(classification_report(df['Mode'], df['Recommended_Mode']))
    else:
        print("\nCannot evaluate: Required columns 'Mode' and/or 'Recommended_Mode' missing.")

def train_ml_model(df):
    if {'Distance_km', 'Weight_kg', 'Urgency', 'Mode'}.issubset(df.columns):
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
        y_pred = clf.predict(X_test)
        y_pred_labels = le_mode.inverse_transform(y_pred)
        df.loc[X_test.index, 'ML_Predicted_Mode'] = y_pred_labels
        print("\nML Model Evaluation:")
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print(classification_report(y_test, y_pred, target_names=le_mode.classes_))
        output_path = "output_with_predictions.csv"
        df.to_csv(output_path, index=False)
        print(f"\nResults exported to {output_path}")
        return clf, le_urgency, le_mode
    else:
        print("\nRequired columns for ML model are missing.")
        return None, None, None

def estimate_cost_eta(distance, weight, mode):
    base_cost_per_km = {
        'LTL': 0.5,
        'TL': 0.8,
        'Drayage': 0.9,
        'Transload': 0.6
    }
    eta_by_mode = {
        'LTL': 4,
        'TL': 2,
        'Drayage': 3,
        'Transload': 5
    }
    cost = distance * base_cost_per_km.get(mode, 0.7) + (weight * 0.1)
    eta = eta_by_mode.get(mode, 3)
    return round(cost, 2), eta

def log_decision(distance, weight, urgency, mode, cost, eta):
    log_file = "agent_decision_log.csv"
    file_exists = os.path.isfile(log_file)
    with open(log_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(['Distance_km', 'Weight_kg', 'Urgency', 'Recommended_Mode', 'Estimated_Cost', 'Estimated_ETA_days'])
        writer.writerow([distance, weight, urgency, mode, cost, eta])

def visualize_decision_log():
    log_file = "agent_decision_log.csv"
    if os.path.isfile(log_file):
        df = pd.read_csv(log_file)
        plt.figure(figsize=(10, 6))
        sns.countplot(data=df, x='Recommended_Mode')
        plt.title("Distribution of Recommended Modes")
        plt.ylabel("Count")
        plt.xlabel("Mode")
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Recommended_Mode', y='Estimated_Cost')
        plt.title("Cost Distribution by Mode")
        plt.ylabel("Estimated Cost")
        plt.xlabel("Mode")
        plt.tight_layout()
        plt.show()
    else:
        print("\nNo decision log found to visualize.")

def run_agent(model, le_urgency, le_mode):
    print("\n=== AI Logistics Agent ===")
    while True:
        try:
            distance = float(input("Enter distance (km): "))
            weight = float(input("Enter weight (kg): "))
            urgency = input("Enter urgency (Low / Medium / High): ").strip().title()
            urgency_code = le_urgency.transform([urgency])[0]
            features = pd.DataFrame([[distance, weight, urgency_code]], columns=['Distance_km', 'Weight_kg', 'Urgency_Code'])
            pred_code = model.predict(features)[0]
            mode = le_mode.inverse_transform([pred_code])[0]
            cost, eta = estimate_cost_eta(distance, weight, mode)
            print(f"\nRecommended Delivery Mode: {mode}")
            print(f"Estimated Cost: ${cost}")
            print(f"Estimated ETA: {eta} days\n")
            log_decision(distance, weight, urgency, mode, cost, eta)
            cont = input("Would you like to test another shipment? (yes/no): ").strip().lower()
            if cont != 'yes':
                break
        except Exception as e:
            print(f"Error: {e}\nTry again.\n")

if __name__ == "__main__":
    filepath = r"C:\Users\dkris\Downloads\shipments_full.csv"
    df = load_data(filepath)
    if df is not None:
        df = apply_recommendations(df)
        print("\nWith Recommended Modes:")
        if {'Origin', 'Destination', 'Urgency', 'Weight_kg', 'Distance_km', 'Mode', 'Recommended_Mode'}.issubset(df.columns):
            print(df[['Origin', 'Destination', 'Urgency', 'Weight_kg', 'Distance_km', 'Mode', 'Recommended_Mode']].head())
        else:
            print("\nSome expected columns are missing from the dataset.")
        evaluate_recommendations(df)
        clf, le_urgency, le_mode = train_ml_model(df)
        if clf is not None:
            run_agent(clf, le_urgency, le_mode)
            visualize_decision_log()
    else:
        print("\nFailed to load data.")