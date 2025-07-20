import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import sys
import os
import logging

# Add the parent directory to the sys.path to allow importing from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.preprocessing import preprocess_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model(
    data_path: str,
    model_path: str,
    preprocessor_path: str,
    categorical_features: list,
    target_column: str,
    output_report_path: str
):
    logging.info("Starting model evaluation...")

    # 1. Load the dataset and trained model
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Dataset loaded from {data_path}")
    except FileNotFoundError:
        logging.error(f"Dataset not found at {data_path}")
        return

    try:
        model = joblib.load(model_path)
        logging.info(f"Model loaded from {model_path}")
    except FileNotFoundError:
        logging.error(f"Model not found at {model_path}")
        return

    try:
        ohe = joblib.load(preprocessor_path)
        logging.info(f"Preprocessor (OneHotEncoder) loaded from {preprocessor_path}")
    except FileNotFoundError:
        logging.error(f"Preprocessor not found at {preprocessor_path}")
        return

    # 2. Preprocess the data (use same steps as training)
    logging.info("Preprocessing data...")
    X_processed, y, _ = preprocess_data(df.copy(), categorical_features, target_column)

    # Ensure columns match the training data after preprocessing
    # This is a crucial step if the test data has different columns or order
    # For simplicity, we'll assume the preprocess_data function handles this consistently
    # In a real-world scenario, you might need to align columns using the fitted OHE's feature names

    # 3. Predict on test split
    logging.info("Making predictions...")
    y_pred = model.predict(X_processed)
    y_proba = model.predict_proba(X_processed)[:, 1]

    # 4. Output evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred)
    conf_matrix = confusion_matrix(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)

    evaluation_results = f"""
Model Evaluation Report
-----------------------

Accuracy: {accuracy:.4f}

Classification Report:
{report}

Confusion Matrix:
{conf_matrix}

ROC-AUC Score: {roc_auc:.4f}
"""
    logging.info(evaluation_results)

    # 5. Save this evaluation report
    with open(output_report_path, 'w') as f:
        f.write(evaluation_results)
    logging.info(f"Evaluation report saved to {output_report_path}")

    # Highlight top 5 most important features (if model supports feature_importances_)
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.Series(model.feature_importances_, index=X_processed.columns)
        top_features = feature_importances.nlargest(5)
        logging.info(f"\nTop 5 Most Important Features:\n{top_features}")
        with open(output_report_path, 'a') as f:
            f.write(f"\nTop 5 Most Important Features:\n{top_features}\n")

    # Warning signs of overfitting or underfitting (basic check)
    # This is a simplified check; more robust analysis would involve train/validation metrics
    if accuracy < 0.6:
        logging.warning("Potential sign of underfitting: Accuracy is low.")
        with open(output_report_path, 'a') as f:
            f.write("\nWarning: Potential sign of underfitting (Accuracy is low).\n")
    elif accuracy > 0.95 and len(y.unique()) > 1: # Check for high accuracy with multiple classes
        logging.warning("Potential sign of overfitting: Accuracy is very high, consider cross-validation.")
        with open(output_report_path, 'a') as f:
            f.write("\nWarning: Potential sign of overfitting (Accuracy is very high).\n")

    logging.info("Model evaluation completed.")

if __name__ == "__main__":
    # Define paths and parameters
    DATA_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\data\\enhanced_saas_churn_data.csv'
    MODEL_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\models\\churnaizer_saas_model.pkl'
    PREPROCESSOR_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\models\\one_hot_encoder.pkl' # Assuming OHE is saved separately
    OUTPUT_REPORT_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\evaluation_report.txt'
    
    # These should match the features used during training
    CATEGORICAL_FEATURES = ['last_payment_status', 'subscription_plan']
    TARGET_COLUMN = 'churn'

    evaluate_model(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        preprocessor_path=PREPROCESSOR_PATH,
        categorical_features=CATEGORICAL_FEATURES,
        target_column=TARGET_COLUMN,
        output_report_path=OUTPUT_REPORT_PATH
    )