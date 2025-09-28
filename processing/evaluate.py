import pandas as pd
"""Module for evaluating the churn prediction model.

This module provides a function to load a trained model and preprocessor,
preprocess new data, make predictions, and generate an evaluation report.
"""

import joblib
import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# from churnaizer.src.preprocessing import preprocess_data # Removed as preprocessing is now part of the pipeline

def evaluate_model(
    data_path: str,
    model_path: str,
    # preprocessor_path: str, # Removed as preprocessor is part of the pipeline
    categorical_features: list,
    target_column: str,
    output_report_path: str
):
    logging.info("Starting model evaluation...")

    # 1. Load the dataset and trained model (which now includes the preprocessor)
    try:
        df = pd.read_csv(data_path)
        logging.info(f"Dataset loaded from {data_path}")
    except FileNotFoundError:
        logging.error(f"Error: Dataset not found at {data_path}")
        return
    except Exception as e:
        logging.error(f"Error loading dataset from {data_path}: {e}")
        return

    try:
        # Load the entire pipeline (preprocessor + model)
        pipeline = joblib.load(model_path)
        logging.info(f"Pipeline (model + preprocessor) loaded from {model_path}")
    except FileNotFoundError:
        logging.error(f"Pipeline not found at {model_path}")
        return
    except Exception as e:
        logging.error(f"Error loading pipeline from {model_path}: {e}")
        return

    # 2. Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # 3. Predict on test split using the loaded pipeline
    logging.info("Making predictions...")
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

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
    if hasattr(model, 'feature_importances_') and X_processed is not None and not X_processed.empty:
        try:
            feature_importances = pd.Series(model.feature_importances_, index=X_processed.columns)
            top_features = feature_importances.nlargest(5)
            logging.info(f"\nTop 5 Most Important Features:\n{top_features}")
            with open(output_report_path, 'a') as f:
                f.write(f"\nTop 5 Most Important Features:\n{top_features}\n")
        except Exception as e:
            logging.warning(f"Could not determine feature importances: {e}")

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
    MODEL_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\models\\churn_prediction_pipeline.joblib' # Updated to pipeline path
    # PREPROCESSOR_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\models\\one_hot_encoder.pkl' # Removed
    OUTPUT_REPORT_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\evaluation_report.txt'
    
    # These should match the features used during training
    CATEGORICAL_FEATURES = ['last_payment_status', 'subscription_plan']
    TARGET_COLUMN = 'churn'

    evaluate_model(
        data_path=DATA_PATH,
        model_path=MODEL_PATH,
        # preprocessor_path=PREPROCESSOR_PATH, # Removed
        categorical_features=CATEGORICAL_FEATURES,
        target_column=TARGET_COLUMN,
        output_report_path=OUTPUT_REPORT_PATH
    )