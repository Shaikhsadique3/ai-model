import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def preprocess_data(df: pd.DataFrame, categorical_features: list, target_column: str):
    """
    Preprocesses the input DataFrame by performing one-hot encoding on categorical features
    and separating the target column.

    Args:
        df (pd.DataFrame): The input DataFrame.
        categorical_features (list): A list of column names to be treated as categorical.
        target_column (str): The name of the target column.

    Returns:
        tuple: A tuple containing:
            - X_processed (pd.DataFrame): The DataFrame with processed features.
            - y (pd.Series): The target variable.
            - preprocessor (ColumnTransformer): The fitted preprocessor object.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Create a column transformer for one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Keep other columns as they are
    )

    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    
    # Get remaining feature names (numerical/passthrough)
    remaining_features = [col for col in X.columns if col not in categorical_features]
    
    all_feature_names = list(ohe_feature_names) + remaining_features
    
    X_processed = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)

    return X_processed, y, preprocessor