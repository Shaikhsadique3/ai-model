import numpy as np
import pandas as pd
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



    # Convert relevant columns to appropriate types first
    for col in ['total_usage_minutes', 'monthly_avg_bill', 'customer_service_interactions']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Identify numerical and categorical columns in X
    numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
    # Ensure all categorical_features are treated as such, even if they were initially object
    # And also include any other object columns that might be present and not explicitly in categorical_features
    all_categorical_cols = [col for col in X.columns if col in categorical_features or X[col].dtype == 'object']
    # Remove duplicates and ensure order
    all_categorical_cols = list(dict.fromkeys(all_categorical_cols))

    # Handle missing values in numerical features
    for col in numerical_cols:
        if X[col].isnull().any():
            imputer = SimpleImputer(strategy='mean')
            X[col] = imputer.fit_transform(X[[col]])

    # Handle missing values in categorical features and convert to category dtype
    for col in all_categorical_cols:
        if col in X.columns:
            # Fill NaN with a placeholder string, then convert to category
            X[col] = X[col].fillna('Missing').astype('category')

    # One-hot encode categorical features
    ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Only encode columns that are actually categorical and present in X
    cols_to_encode = [col for col in all_categorical_cols if col in X.columns]

    X_categorical_encoded = pd.DataFrame()
    if cols_to_encode:
        X_categorical = ohe.fit_transform(X[cols_to_encode])
        ohe_feature_names = ohe.get_feature_names_out(cols_to_encode)
        X_categorical_encoded = pd.DataFrame(X_categorical, columns=ohe_feature_names, index=X.index)

    # Drop original categorical columns from X before concatenating
    X_numerical = X.drop(columns=cols_to_encode, errors='ignore')

    X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1)

    # Apply SMOTE for class balancing
    logging.info("Applying SMOTE for class balancing...")
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_processed, y)
    logging.info("Data preprocessing completed.")
    return X_resampled, y_resampled, ohe