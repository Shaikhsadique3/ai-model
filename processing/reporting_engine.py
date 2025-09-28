import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from recommendation_engine import generate_recommendations, estimate_roi


def generate_dashboard_data(pipeline_path, raw_data_path):
    """
    Generates data points and visualizations for a founder-friendly dashboard.
    """
    print("Generating dashboard data...")

    # Load raw data
    df_raw = pd.read_csv(raw_data_path)

    # Load the pipeline to get the preprocessor for feature names if needed
    pipeline = joblib.load(pipeline_path)

    # Assuming 'churn' is the target column
    if 'churn' in df_raw.columns:
        churn_rate = df_raw['churn'].mean() * 100
    else:
        # If 'churn' is not in raw data, we might need to predict it first or handle differently
        print("Warning: 'churn' column not found in raw data for churn rate calculation.")
        churn_rate = 0.0 # Placeholder

    print(f"Overall Churn Rate: {churn_rate:.2f}%")

    # Top Churn Contributing Features (using recommendation_engine)
    # This will print to console for now, eventually integrate into a dashboard object
    print("\nTop Churn Contributing Features:")
    generate_recommendations(pipeline_path, raw_data_path) # Updated call

    # Placeholder for more dashboard data generation (e.g., customer segmentation, engagement trends)

    dashboard_data = {
        'churn_rate': churn_rate,
        # 'feature_importance': feature_importances # This would come from generate_recommendations
    }
    return dashboard_data

def generate_alerts(pipeline_path, raw_data_path, threshold=0.7):
    """
    Generates alerts for high-risk customers or significant churn trends.
    """
    print("Generating alerts...")

    # Load pipeline and raw data
    pipeline = joblib.load(pipeline_path)
    df_raw = pd.read_csv(raw_data_path)

    # Separate features and target (assuming 'churn' is the target column)
    if 'churn' in df_raw.columns:
        X_raw = df_raw.drop(columns=['churn'])
        y_raw = df_raw['churn']
    else:
        X_raw = df_raw.copy()
        y_raw = None # No target for alert generation if not present

    # Predict churn probabilities using the loaded pipeline
    churn_probabilities = pipeline.predict_proba(X_raw)[:, 1]

    # Identify high-risk customers
    high_risk_customers = df_raw[churn_probabilities >= threshold]

    if not high_risk_customers.empty:
        print(f"\nAlert: {len(high_risk_customers)} customers identified with churn probability >= {threshold:.2f}!")
        print(high_risk_customers.head())
    else:
        print("No high-risk customer alerts at this time.")

    # Placeholder for more alert types (e.g., sudden drops in engagement)

def visualize_dashboard(dashboard_data):
    """
    Visualizes the dashboard data (placeholder for actual dashboard generation).
    """
    print("Visualizing dashboard (placeholder)...")
    # In a real application, this would involve a web framework (e.g., Dash, Streamlit) or a BI tool.
    # For now, we'll just print some basic info.
    print(f"Dashboard Summary: Overall Churn Rate = {dashboard_data['churn_rate']:.2f}%")

if __name__ == "__main__":
    pipeline_file_path = r'c:\Users\Sadique\Desktop\ai model\models\churn_prediction_pipeline.joblib' # Updated to pipeline path
    raw_data_file_path = r'c:\Users\Sadique\Desktop\ai model\data\enhanced_saas_churn_data.csv' # Using raw data

    # Generate dashboard data
    dashboard_info = generate_dashboard_data(pipeline_file_path, raw_data_file_path)

    # Generate alerts
    generate_alerts(pipeline_file_path, raw_data_file_path)

    # Visualize dashboard
    visualize_dashboard(dashboard_info)