import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import logging
import os
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def select_features(config_path: str = 'config/config.json'):
    """
    Performs feature selection using RandomForestClassifier and SelectFromModel.
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        dataset_path = config.get('dataset_path')
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

    # Define features and target
    target_column = 'churn'
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in the dataset.")
        return

    # Identify categorical features
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Preprocess data using the shared function
    X_processed, y, ohe = preprocess_data(df, categorical_features, target_column)

    # Feature selection using RandomForestClassifier
    logging.info("Performing feature selection...")
    sfm = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42))
    sfm.fit(X_processed, y)

    # Get selected features
    selected_feature_indices = sfm.get_support()
    all_feature_names = X_processed.columns
    selected_features = all_feature_names[selected_feature_indices].tolist()

    logging.info(f"Selected features: {selected_features}")

if __name__ == "__main__":
    select_features()