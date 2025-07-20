import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from src.preprocessing import preprocess_data
from xgboost import XGBClassifier
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def evaluate_model_performance(config_path: str = 'config/config.json'):
    """
    Evaluates the model performance using cross-validation.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        dataset_path = config.get('dataset_path')
        selected_features = config.get('selected_features', [])
    except FileNotFoundError:
        logging.error(f"Configuration file not found at {config_path}")
        return
    except json.JSONDecodeError:
        logging.error(f"Error decoding JSON from {config_path}. Check file format.")
        return
    except Exception as e:
        logging.error(f"An unexpected error occurred while loading config: {e}")
        return

    try:
        df = pd.read_csv(dataset_path)
        logging.info(f"Dataset loaded successfully from {dataset_path}")
    except FileNotFoundError:
        logging.error(f"Dataset not found at {dataset_path}. Please check the path in config.json.")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading the dataset: {e}")
        return

    target_column = 'churn'
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in the dataset.")
        return

    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Preprocess data using the shared function, but without SMOTE for evaluation
    # We pass a dummy target column to preprocess_data as SMOTE is not needed here
    # and we only care about X_processed and ohe for feature selection.
    X_processed, _, ohe = preprocess_data(df, categorical_features, target_column)

    # Re-align y after preprocessing
    y = df[target_column]

    # Filter for selected features if specified in config
    if selected_features:
        # Ensure all selected features exist in the processed DataFrame
        missing_features = [f for f in selected_features if f not in X_processed.columns]
        if missing_features:
            logging.warning(f"The following selected features are not found in the dataset: {missing_features}. "
                            "Proceeding with available selected features.")
            selected_features = [f for f in selected_features if f in X_processed.columns]
        
        if selected_features:
            X_processed = X_processed[selected_features]
        else:
            logging.warning("No valid selected features found after filtering. Using all processed features.")

    # Initialize XGBClassifier
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

    # Perform cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    try:
        f1_scores = cross_val_score(model, X_processed, y, cv=cv, scoring='f1')
        logging.info(f"Cross-validated F1 Scores: {f1_scores}")
        logging.info(f"Average F1 Score: {np.mean(f1_scores):.2f}")
    except Exception as e:
        logging.error(f"An error occurred during cross-validation: {e}")

if __name__ == "__main__":
    evaluate_model_performance()