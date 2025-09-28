import pandas as pd
"""Module for evaluating the churn prediction model.

This module provides a function to load a trained model and preprocessor,
preprocess new data, make predictions, and generate an evaluation report.
"""

import joblib
import os
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

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

    # Transform X using the pipeline's preprocessor to get feature names for plotting
    X_processed = pipeline.named_steps['preprocessor'].transform(X)
    # If the preprocessor outputs a sparse matrix, convert to dense for DataFrame
    if hasattr(X_processed, 'toarray'):
        X_processed = X_processed.toarray()
    
    # Get feature names after preprocessing
    feature_names = []
    preprocessor = pipeline.named_steps['preprocessor']
    for name, transformer, features in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(features))
        elif name == 'num': # For numeric features, just use the original names
            feature_names.extend(features)
    
    X_processed_df = pd.DataFrame(X_processed, columns=feature_names)

    # 3. Predict on test split using the loaded pipeline
    logging.info("Making predictions...")
    y_pred = pipeline.predict(X)
    y_proba = pipeline.predict_proba(X)[:, 1]

    # 4. Output evaluation metrics
    accuracy = accuracy_score(y, y_pred)
    report = classification_report(y, y_pred, output_dict=True)
    conf_matrix = confusion_matrix(y, y_pred)
    roc_auc = roc_auc_score(y, y_proba)

    evaluation_results = f"""
Model Evaluation Report
-----------------------

Accuracy: {accuracy:.4f}

Classification Report:
{classification_report(y, y_pred, zero_division=0)}

Confusion Matrix:
{conf_matrix}

ROC-AUC Score: {roc_auc:.4f}
"""
    logging.info(evaluation_results)

    # 5. Save this evaluation report
    with open(output_report_path, 'w') as f:
        f.write(evaluation_results)
    logging.info(f"Evaluation report saved to {output_report_path}")

    # Generate and save Confusion Matrix plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    conf_matrix_path = output_report_path.replace('.txt', '_confusion_matrix.png')
    plt.savefig(conf_matrix_path)
    logging.info(f"Confusion Matrix plot saved to {conf_matrix_path}")
    plt.close()

    # Generate and save ROC Curve plot
    fpr, tpr, _ = roc_curve(y, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    roc_curve_path = output_report_path.replace('.txt', '_roc_curve.png')
    plt.savefig(roc_curve_path)
    logging.info(f"ROC Curve plot saved to {roc_curve_path}")
    plt.close()

    # Highlight top 5 most important features (if model supports feature_importances_ or coef_)
    model = pipeline.named_steps['model']
    if hasattr(model, 'feature_importances_'):
        try:
            feature_importances = pd.Series(model.feature_importances_, index=X_processed_df.columns)
            top_features = feature_importances.nlargest(5)
            logging.info(f"\nTop 5 Most Important Features:\n{top_features}")
            with open(output_report_path, 'a') as f:
                f.write(f"\nTop 5 Most Important Features:\n{top_features}\n")
        except Exception as e:
            logging.warning(f"Could not determine feature importances: {e}")
    elif hasattr(model, 'coef_'):
        try:
            # For linear models, coef_ can be used as importance
            feature_importances = pd.Series(model.coef_[0], index=X_processed_df.columns)
            top_features = feature_importances.abs().nlargest(5)
            logging.info(f"\nTop 5 Most Important Features (from coefficients):\n{top_features}")
            with open(output_report_path, 'a') as f:
                f.write(f"\nTop 5 Most Important Features (from coefficients):\n{top_features}\n")
        except Exception as e:
            logging.warning(f"Could not determine feature importances from coefficients: {e}")

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
    DATA_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\data\\client_data_raw.csv'
    MODEL_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\model\\model.pkl'
    OUTPUT_REPORT_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\report\\performance_report.txt'
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