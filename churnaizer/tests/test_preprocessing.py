import numpy as np
import pandas as pd
import numpy as np
from src.preprocessing import preprocess_data

def test_null_handling():
    # Test case with null values
    data = {
        'total_usage_minutes': [1.0, 2.0, np.nan, 4.0],
        'subscription_plan': ['A', 'B', 'A', np.nan],
        'churn': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)

    categorical_features = ['subscription_plan']
    target_column = 'churn'

    X_resampled, y_resampled, ohe = preprocess_data(df.copy(), categorical_features, target_column)

    # Assert no nulls in the processed DataFrame
    assert not X_resampled.isnull().any().any(), "Processed DataFrame should not contain null values."

def test_encoding_correctness():
    # Test case for one-hot encoding
    data = {
        'subscription_plan': ['A', 'B', 'A', 'C'],
        'total_usage_minutes': [10, 20, 30, 40],
        'churn': [0, 1, 0, 1]
    }
    df = pd.DataFrame(data)
    categorical_features = ['subscription_plan']
    target_column = 'churn'

    X_resampled, y_resampled, ohe = preprocess_data(df.copy(), categorical_features, target_column)

    # Check if new columns for one-hot encoded features exist
    assert 'subscription_plan_A' in X_resampled.columns
    assert 'subscription_plan_B' in X_resampled.columns
    assert 'subscription_plan_C' in X_resampled.columns

    # Check encoding correctness for a specific row
    # Original row: {'subscription_plan': 'A', 'total_usage_minutes': 10, 'churn': 0}
    # Find rows where total_usage_minutes is 10 (or close to it due to SMOTE interpolation)
    encoded_A_rows = X_resampled[X_resampled['subscription_plan_A'] == 1]
    assert not encoded_A_rows.empty
    assert (encoded_A_rows['subscription_plan_B'] == 0).all()
    assert (encoded_A_rows['subscription_plan_C'] == 0).all()

    encoded_B_rows = X_resampled[X_resampled['subscription_plan_B'] == 1]
    assert not encoded_B_rows.empty
    assert (encoded_B_rows['subscription_plan_A'] == 0).all()
    assert (encoded_B_rows['subscription_plan_C'] == 0).all()

    encoded_C_rows = X_resampled[X_resampled['subscription_plan_C'] == 1]
    assert not encoded_C_rows.empty
    assert (encoded_C_rows['subscription_plan_A'] == 0).all()
    assert (encoded_C_rows['subscription_plan_B'] == 0).all()

    # Check that original categorical column is removed
    assert 'subscription_plan' not in X_resampled.columns

def test_smote_application():
    # Test if SMOTE increases the number of samples for the minority class
    data = {
        'total_usage_minutes': list(range(1, 46)),
        'churn': [0]*20 + [1]*25 # More balanced for SMOTE
    }
    df = pd.DataFrame(data)
    categorical_features = [] # No categorical features in this test
    target_column = 'churn'

    X_resampled, y_resampled, ohe = preprocess_data(df.copy(), categorical_features, target_column)

    # Check if the number of samples for the minority class (1) has increased


    resampled_minority_count = y_resampled.value_counts()[1]


    # With default SMOTE, minority class count should become equal to majority class count
    assert y_resampled.value_counts()[0] == y_resampled.value_counts()[1]