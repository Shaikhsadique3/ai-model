import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def feature_engineer_and_preprocess(file_path):
    """
    Loads the Synthetic SaaS Churn Dataset, engineers new features, and preprocesses the data.

    Args:
        file_path (str): The path to the synthetic_saas_churn_100k.csv file.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with new features.
    """
    df = pd.read_csv(file_path)

    # Feature Engineering
    # 1. Engagement Score: A composite score based on login frequency and active features used.
    #    Assuming higher login frequency and more active features indicate higher engagement.
    df['engagement_score'] = df['number_of_logins_last30days'] * df['active_features_used']

    # 2. Payment Reliability: A categorical feature derived from 'payment_status'.
    #    This can be directly used from the existing 'payment_status' column after encoding.
    #    For now, we'll keep it as is and handle it during one-hot encoding.

    # 3. Satisfaction Trend: If NPS_score is available, we can use it directly.
    #    If there were historical NPS scores, we could calculate a trend.
    #    For this dataset, we'll use NPS_score as a direct indicator of satisfaction.
    df['satisfaction_trend'] = df['NPS_score']

    # Handle Missing Values (if any, though synthetic data might be clean)
    # For numerical features, we'll fill with the median.
    for col in ['monthly_revenue', 'days_since_signup', 'last_login_days_ago', 
                'number_of_logins_last30days', 'active_features_used', 'tickets_opened', 
                'NPS_score', 'engagement_score', 'satisfaction_trend']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # For categorical features, we'll fill with the mode.
    for col in ['plan_type', 'payment_status']:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    # Define categorical and numerical features for preprocessing
    categorical_features = ['plan_type', 'payment_status']
    numerical_features = ['monthly_revenue', 'days_since_signup', 'last_login_days_ago',
                          'logins_last30days', 'active_features_used', 'tickets_opened',
                          'NPS_score', 'engagement_score', 'satisfaction_trend']

    # Create preprocessing pipelines for numerical and categorical features
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # Create a pipeline that first preprocesses and then scales
    # We will fit and transform the preprocessor later when we have the full dataset
    # For now, let's just apply the transformations
    # Separate target variable
    X = df.drop('churned', axis=1)
    y = df['churned']

    # Fit and transform the features
    X_processed = preprocessor.fit_transform(X)

    # Get feature names after one-hot encoding
    onehot_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numerical_features + list(onehot_feature_names)

    X_processed_df = pd.DataFrame(X_processed, columns=all_feature_names)
    X_processed_df['churned'] = y.values

    return X_processed_df

# This function will now be called from train_model.py
def feature_engineer_and_preprocess(df: pd.DataFrame):
    """
    Loads the Synthetic SaaS Churn Dataset, engineers new features, and preprocesses the data.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with new features.
    """

    # Feature Engineering
    # 1. Engagement Score: A composite score based on login frequency and active features used.
    #    Assuming higher login frequency and more active features indicate higher engagement.
    df['engagement_score'] = df['logins_last30days'] * df['active_features_used']

    # 2. Payment Reliability: A categorical feature derived from 'payment_status'.
    #    This can be directly used from the existing 'payment_status' column after encoding.
    #    For now, we'll keep it as is and handle it during one-hot encoding.

    # 3. Satisfaction Trend: If NPS_score is available, we can use it directly.
    #    If there were historical NPS scores, we could calculate a trend.
    #    For this dataset, we'll use NPS_score as a direct indicator of satisfaction.
    df['satisfaction_trend'] = df['NPS_score']

    # Handle Missing Values (if any, though synthetic data might be clean)
    # For numerical features, we'll fill with the median.
    for col in ['monthly_revenue', 'days_since_signup', 'last_login_days_ago', 
                'logins_last30days', 'active_features_used', 'tickets_opened', 
                'NPS_score', 'engagement_score', 'satisfaction_trend']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())

    # For categorical features, we'll fill with the mode.
    for col in ['plan_type', 'payment_status']:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])

    return df

if __name__ == "__main__":
    # This block is for testing purposes or standalone execution of feature engineering
    # It will load raw data, apply feature engineering, and save it to a new CSV.
    # The main training pipeline will load the raw data directly and apply feature engineering
    # and preprocessing within the pipeline.
    saas_churn_file_path = r'c:\Users\Sadique\Desktop\ai model\synthetic_saas_churn_100k.csv'
    raw_df = pd.read_csv(saas_churn_file_path)
    engineered_df = feature_engineer_and_preprocess(raw_df)
    
    # Save the engineered data (optional, for inspection or other uses)
    engineered_output_path = r'c:\Users\Sadique\Desktop\ai model\data\engineered_saas_churn.csv'
    engineered_df.to_csv(engineered_output_path, index=False)
    print(f"Engineered data saved to {engineered_output_path}")
    print(engineered_df.head())