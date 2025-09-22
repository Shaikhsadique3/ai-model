import sys
import os

# Add the project root to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import joblib
import configparser
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from processing.preprocessing import preprocess_data
from processing.process_csv import process_csv

if __name__ == '__main__':
    print("Starting model training and saving...")

    # Load configuration
    config = configparser.ConfigParser()
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.ini')
    config.read(config_path)

    # Define paths
    MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'churnaizer_saas_model.pkl')
    PREPROCESSOR_PATH = os.path.join(MODEL_DIR, 'one_hot_encoder.pkl')
    
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv("processing/enhanced_saas_churn_data.csv")
        print("Data loaded successfully.")
    except FileNotFoundError:
        print("Error: enhanced_saas_churn_data.csv not found. Please ensure it's in the correct directory.")
        exit()

    # Define features and target from config
    numerical_features = [f.strip() for f in config['model']['numerical_features'].split(',')]
    categorical_features = [f.strip() for f in config['model']['categorical_features'].split(',')]
    target_column = config['model']['target_column']

    # Preprocess data
    print("Preprocessing data...")
    X_processed, y, preprocessor = preprocess_data(df.copy(), categorical_features, target_column)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)

    # Train model
    print("Training model...")
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train, y_train)
    print("Model training complete.")

    # Save model and preprocessor
    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    print(f"Model saved to {MODEL_PATH}")
    print(f"Preprocessor saved to {PREPROCESSOR_PATH}")

    print("\nProcessing complete.")

    try:
        processed_df = pd.read_csv("processed_with_predictions.csv")
        print("\nFirst 10 rows of processed_with_predictions.csv:")
        print(processed_df.head(10))
    except FileNotFoundError:
        print("Error: processed_with_predictions.csv not found.")

    print("\nSaved files:")
    print(" - processed_with_predictions.csv")
    print(" - stats_summary.json")
    print(" - log.txt")
    print(f" - {os.path.basename(MODEL_PATH)}")
    print(f" - {os.path.basename(PREPROCESSOR_PATH)}")