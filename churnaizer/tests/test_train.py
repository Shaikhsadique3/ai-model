import pandas as pd
import numpy as np
import pytest
import os
import json
from unittest.mock import patch, MagicMock

from src.preprocessing import preprocess_data
# Adjust the import path based on your project structure
# Assuming churnaizer/src/train.py and churnaizer/config/config.json
from src.train import ChurnPredictor

# Mock config.json for testing
@pytest.fixture(scope='module')
def mock_config_file(tmp_path_factory):
    config_dir = tmp_path_factory.mktemp("config")
    config_path = config_dir / "config.json"
    test_data_path = tmp_path_factory.mktemp("data") / "test_data.csv"
    test_model_path = tmp_path_factory.mktemp("models") / "test_model.pkl"

    config_content = {
        "dataset_path": str(test_data_path),
        "model_path": str(test_model_path),
        "selected_features": []
    }
    with open(config_path, 'w') as f:
        json.dump(config_content, f)
    return str(config_path), str(test_data_path), str(test_model_path)

# Mock dataset for testing
@pytest.fixture(scope='module')
def mock_dataset_file(mock_config_file):
    _, test_data_path, _ = mock_config_file
    data = {
        'customer_id': range(100),
        'feature_numeric_1': np.random.rand(100),
        'feature_numeric_2': np.random.rand(100),
        'feature_categorical_1': np.random.choice(['A', 'B', 'C'], 100),
        'feature_categorical_2': np.random.choice(['X', 'Y'], 100),
        'total_usage_minutes': np.random.randint(100, 1000, 100),
        'monthly_avg_bill': np.random.uniform(20, 100, 100),
        'customer_service_interactions': np.random.randint(0, 5, 100),
        'churn': np.random.choice([0, 1], 100, p=[0.8, 0.2]) # Imbalanced for SMOTE
    }
    df = pd.DataFrame(data)
    df.to_csv(test_data_path, index=False)
    return df

def test_model_training_accuracy(mock_config_file, mock_dataset_file):
    config_path, _, _ = mock_config_file

    # Temporarily change the current working directory to the churnaizer directory
    # so that the config.json path is resolved correctly by ChurnPredictor
    original_cwd = os.getcwd()
    churnaizer_root = os.path.dirname(os.path.dirname(config_path)) # Adjust based on actual structure
    os.chdir(churnaizer_root)

    try:
        predictor = ChurnPredictor(config_path=config_path)
        predictor.run()

        # After running the pipeline, the model should be trained and evaluated
        # We can't directly assert on evaluate_model's output as it logs, but we can check
        # if the model was saved and if its internal score is above a threshold.
        # For a true unit test, we'd mock the evaluation part or capture logs.

        # For simplicity, let's re-evaluate the model on the test set used internally by run()
        # This requires some internal knowledge of ChurnPredictor's run method
        # A better approach for unit testing would be to test train_model and evaluate_model separately
        # and pass in preprocessed data.

        # Since run() orchestrates everything, we'll check the saved model's performance
        # This is more of an integration test, but fits the request for accuracy check.

        # Load the saved model
        with open(predictor.model_path, 'rb') as f:
            loaded_model = pickle.load(f)

        # Re-load and preprocess data to get X_test, y_test
        df = predictor.load_data()
        categorical_features = ['feature_categorical_1', 'feature_categorical_2']
        target_column = 'churn'

        X_processed, y_resampled, _ = preprocess_data(df, categorical_features, target_column)

        # Split data again to get the test set used by run()
        # This assumes run() uses the same random_state and test_size
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled)

        # Predict and evaluate
        y_pred = loaded_model.predict(X_test)
        accuracy = loaded_model.score(X_test, y_test)

        # Assert accuracy is above threshold
        assert accuracy > 0.75 # Lower threshold for mock data, adjust as needed

    finally:
        os.chdir(original_cwd)

# To run these tests, you would typically install pytest and run `pytest` from the churnaizer directory.
# Make sure your PYTHONPATH includes the directory containing 'src' or run pytest from the parent directory of 'src'.