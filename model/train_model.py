import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
DATA_PATH = 'churn_training_data.csv'
MODEL_DIR = 'model'
XGB_MODEL_PATH = os.path.join(MODEL_DIR, 'churnaizer_model.pkl')
RF_MODEL_PATH = os.path.join(MODEL_DIR, 'churnaizer_saas_model.pkl')

def load_data(path):
    logger.info(f"Loading data from {path}")
    try:
        df = pd.read_csv(path)
        logger.info("Data loaded successfully.")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Data file not found at {path}")
        return None

def preprocess_data(df):
    logger.info("Preprocessing data...")
    # Handle categorical features
    df = pd.get_dummies(df, columns=['plan_type', 'payment_status'], drop_first=True)
    
    # Convert signup_date to numerical timestamp
    if 'signup_date' in df.columns:
        df['signup_date'] = pd.to_datetime(df['signup_date']).astype(int) / 10**9 # Convert to Unix timestamp
    
    # Define features (X) and target (y)
    X = df.drop('churned', axis=1)
    y = df['churned']
    
    logger.info("Data preprocessed successfully.")
    return X, y

def train_and_evaluate_model(model, X_train, y_train, X_test, y_test, model_name):
    logger.info(f"Training {model_name}...")
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    
    logger.info(f"{model_name} Performance:")
    logger.info(f"  Accuracy: {accuracy:.4f}")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1-Score: {f1:.4f}")
    logger.info(f"\nClassification Report for {model_name}:\n{classification_report(y_test, predictions)}")
    
    return accuracy, precision, recall, f1

def save_model(model, path, model_name):
    logger.info(f"Saving {model_name} to {path}")
    try:
        if model_name == "XGBoost Model":
            joblib.dump(model, path)
        else:
            joblib.dump(model, path)
        logger.info(f"{model_name} saved successfully.")
    except Exception as e:
        logger.error(f"Error saving {model_name}: {e}")

if __name__ == "__main__":
    # Ensure model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    if df is not None:
        X, y = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        logger.info("Data split into training and testing sets.")

        # Train and evaluate XGBoost
        xgb_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_accuracy, xgb_precision, xgb_recall, xgb_f1 = train_and_evaluate_model(xgb_model, X_train, y_train, X_test, y_test, "XGBoost Model")
        save_model(xgb_model, XGB_MODEL_PATH, "XGBoost Model")

        # Train and evaluate RandomForest
        rf_model = RandomForestClassifier(random_state=42)
        rf_accuracy, rf_precision, rf_recall, rf_f1 = train_and_evaluate_model(rf_model, X_train, y_train, X_test, y_test, "RandomForest Model")
        save_model(rf_model, RF_MODEL_PATH, "RandomForest Model")

        logger.info("\n--- Training Summary ---")
        logger.info(f"XGBoost - Accuracy: {xgb_accuracy:.4f}, Precision: {xgb_precision:.4f}, Recall: {xgb_recall:.4f}, F1: {xgb_f1:.4f}")
        logger.info(f"RandomForest - Accuracy: {rf_accuracy:.4f}, Precision: {rf_precision:.4f}, Recall: {rf_recall:.4f}, F1: {rf_f1:.4f}")