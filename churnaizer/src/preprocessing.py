import numpy as np
"""Module for data preprocessing in the churn prediction project.

This module provides functions for handling missing values, converting data types,
performing one-hot encoding, and applying SMOTE for class balancing.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import logging

def preprocess_data(df: pd.DataFrame, categorical_features: list, target_column: str, preprocessor: OneHotEncoder = None) -> tuple[pd.DataFrame, pd.Series, OneHotEncoder]:
    """
    Preprocesses the input DataFrame for model training by handling missing values,
    converting data types, performing one-hot encoding, and applying SMOTE for class balancing.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): A list of column names to be treated as categorical.
        target_column (str): The name of the target column.
        preprocessor (OneHotEncoder, optional): An optional pre-fitted OneHotEncoder. If None, a new one is fitted.

    Returns:
        tuple[pd.DataFrame, pd.Series, OneHotEncoder]: A tuple containing the preprocessed features (X),
                                                      the target variable (y), and the fitted OneHotEncoder.
    """
    logging.info("Starting data preprocessing for training...")

    # Make a copy to avoid modifying the original DataFrame
    df_processed = df.copy()

    # Convert specified columns to numeric, coercing errors
    for col in ['days_since_signup', 'monthly_revenue', 'number_of_logins_last30days', 'active_features_used', 'support_tickets_opened', 'avg_session_duration', 'last_login_days_ago', 'email_opens_last30days', 'billing_issue_count', 'trial_conversion_flag']:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

    # Explicitly convert known categorical features to string type
    for col in ['last_payment_status', 'subscription_plan']:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)

    X = df_processed.drop(columns=[target_column])
    y = df_processed[target_column]

    # Identify numerical and categorical columns in X
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    all_categorical_cols = [col for col in X.columns if col in categorical_features or X[col].dtype == 'object']
    all_categorical_cols = list(dict.fromkeys(all_categorical_cols))

    # Handle missing values in numerical features
    for col in numerical_cols:
        if col in X.columns and X[col].isnull().any():
            imputer = SimpleImputer(strategy='mean')
            X[col] = imputer.fit_transform(X[[col]])

    # Handle missing values in categorical features and convert to category dtype
    for col in all_categorical_cols:
        if col in X.columns:
            X[col] = X[col].fillna('Missing').astype('category')

    # One-hot encode categorical features
    cols_to_encode = [col for col in all_categorical_cols if col in X.columns]
    if preprocessor is None:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        X_categorical_encoded = pd.DataFrame()
        if cols_to_encode:
            X_categorical = ohe.fit_transform(X[cols_to_encode])
            ohe_feature_names = ohe.get_feature_names_out(cols_to_encode)
            X_categorical_encoded = pd.DataFrame(X_categorical, columns=ohe_feature_names, index=X.index)
    else:
        ohe = preprocessor
        fitted_features = list(ohe.feature_names_in_)
        X_to_transform = X[fitted_features]
        cols_to_encode = fitted_features

        X_categorical_encoded = pd.DataFrame()
        if not X_to_transform.empty:
            X_categorical = ohe.transform(X_to_transform)
            ohe_feature_names = ohe.get_feature_names_out(fitted_features)
            X_categorical_encoded = pd.DataFrame(X_categorical, columns=ohe_feature_names, index=X.index)

    X_numerical = X.drop(columns=cols_to_encode, errors='ignore')
    X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1)

    # Apply SMOTE for class balancing
    logging.info("Applying SMOTE for class balancing...")
    nan_counts = X_processed.isnull().sum()
    if nan_counts.sum() > 0:
        logging.warning(f"NaN values found in X_processed before SMOTE:\n{nan_counts[nan_counts > 0]}")
    smote = SMOTE(random_state=42, k_neighbors=1)
    logging.info(f"Shape of X before SMOTE: {X_processed.shape}")
    logging.info(f"Value counts of y before SMOTE:\n{y.value_counts()}")
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    logging.info(f"Shape of X after SMOTE: {X_resampled.shape}")
    logging.info(f"Value counts of y after SMOTE:\n{y_resampled.value_counts()}")
    logging.info("Data preprocessing for training completed.")
    return X_resampled, y_resampled, ohe

def transform_data_for_prediction(df: pd.DataFrame, numerical_features: list, categorical_features: list, preprocessor: OneHotEncoder) -> pd.DataFrame:
    """
    Transforms the input DataFrame for prediction using a pre-fitted OneHotEncoder.
    Handles missing values and applies one-hot encoding without fitting or SMOTE.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_features (list): A list of column names to be treated as numerical.
        categorical_features (list): A list of column names to be treated as categorical.
        preprocessor (OneHotEncoder): A pre-fitted OneHotEncoder.

    Returns:
        pd.DataFrame: The transformed DataFrame ready for prediction.
    """
    logging.info("Starting data transformation for prediction...")

    df_transformed = df.copy()

    # Convert specified columns to numeric, coercing errors
    for col in numerical_features:
        if col in df_transformed.columns:
            df_transformed[col] = pd.to_numeric(df_transformed[col], errors='coerce')

    # Explicitly convert known categorical features to string type
    for col in categorical_features:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].astype(str)

    # Handle missing values in numerical features (mean imputation)
    for col in numerical_features:
        if col in df_transformed.columns and df_transformed[col].isnull().any():
            imputer = SimpleImputer(strategy='mean')
            df_transformed[col] = imputer.fit_transform(df_transformed[[col]])

    # Handle missing values in categorical features (fill with 'Missing')
    for col in categorical_features:
        if col in df_transformed.columns:
            df_transformed[col] = df_transformed[col].fillna('Missing')

    # Apply the pre-fitted OneHotEncoder
    # Ensure that the columns passed to the preprocessor are exactly what it expects
    # based on its fitted feature names.
    fitted_categorical_features = list(preprocessor.feature_names_in_)
    
    # Filter df_transformed to only include the features the OHE expects, in the correct order
    X_to_transform_categorical = df_transformed[fitted_categorical_features]

    X_categorical_encoded = pd.DataFrame()
    if not X_to_transform_categorical.empty:
        X_categorical = preprocessor.transform(X_to_transform_categorical)
        ohe_feature_names = preprocessor.get_feature_names_out(fitted_categorical_features)
        X_categorical_encoded = pd.DataFrame(X_categorical, columns=ohe_feature_names, index=df_transformed.index)

    # Drop original categorical columns from df_transformed before concatenating
    df_numerical = df_transformed.drop(columns=categorical_features, errors='ignore')

    X_processed = pd.concat([df_numerical[numerical_features], X_categorical_encoded], axis=1)
    logging.info("Data transformation for prediction completed.")
    return X_processed