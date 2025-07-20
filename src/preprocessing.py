import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def preprocess_data(df: pd.DataFrame, categorical_features: list, target_column: str) -> tuple[pd.DataFrame, pd.Series, OneHotEncoder]:
    """
    Preprocesses the input DataFrame by handling missing values, converting data types,
    performing one-hot encoding, and applying SMOTE for class balancing.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): A list of column names to be treated as categorical.
        target_column (str): The name of the target column.

    Returns:
        tuple[pd.DataFrame, pd.Series, OneHotEncoder]: A tuple containing the preprocessed features (X),
                                                      the target variable (y), and the fitted OneHotEncoder.
    """
    logging.info("Starting data preprocessing...")

    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object' or df[col].dtype == 'category':
            imputer = SimpleImputer(strategy='most_frequent')
        else:
            imputer = SimpleImputer(strategy='mean')
        df[col] = imputer.fit_transform(df[[col]])

    # Convert relevant columns to appropriate types
    for col in ['total_usage_minutes', 'monthly_avg_bill', 'customer_service_interactions']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # One-hot encode categorical features
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_categorical = ohe.fit_transform(X[categorical_features])
    ohe_feature_names = ohe.get_feature_names_out(categorical_features)
    X_categorical_df = pd.DataFrame(X_categorical, columns=ohe_feature_names, index=X.index)

    X_processed = pd.concat([X.drop(columns=categorical_features), X_categorical_df], axis=1)

    # Apply SMOTE for class balancing
    logging.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    logging.info("Data preprocessing completed.")
    return X_resampled, y_resampled, ohe