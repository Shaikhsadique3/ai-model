import pytest
import pandas as pd
import os
import numpy as np
from churnaizer.src.predict import ChurnPredictorService
# Define paths for test models and preprocessors (these should be dummy/mocked for unit tests)
# For a real test, you'd train a small model and save it, or mock the loading.
TEST_MODEL_PATH = "./test_model.joblib"
TEST_PREPROCESSOR_PATH = "./test_preprocessor.joblib"
TEST_CONFIG_PATH = "./test_config.ini"

@pytest.fixture(scope="module")
def setup_test_environment():
    # Create dummy model, preprocessor, and config for testing
    # In a real scenario, you'd have actual small, pre-trained artifacts
    
    # Dummy model (e.g., a simple LogisticRegression)
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import OneHotEncoder
    import joblib
    import configparser
    
    # Create a dummy model
    model = LogisticRegression()
    joblib.dump(model, TEST_MODEL_PATH)
    
    # Create a dummy preprocessor
    preprocessor = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Fit it on some dummy data to get feature names
    dummy_data = pd.DataFrame({
        'subscription_plan': ['basic', 'premium', 'basic'],
        'last_payment_status': ['success', 'failed', 'success'],
        'days_since_signup': [100, 200, 50],
        'monthly_revenue': [50.0, 100.0, 25.0]
    })
    preprocessor.fit(dummy_data[['subscription_plan', 'last_payment_status']])
    joblib.dump(preprocessor, TEST_PREPROCESSOR_PATH)
    
    # Create a dummy config.ini
    config = configparser.ConfigParser()
    config['paths'] = {
        'model_path': TEST_MODEL_PATH,
        'preprocessor_path': TEST_PREPROCESSOR_PATH
    }
    config['model'] = {
        'numerical_features': 'days_since_signup, monthly_revenue',
        'categorical_features': 'subscription_plan, last_payment_status',
        'target_column': 'churn'
    }
    config['thresholds'] = {
        'high': '0.7',
        'medium': '0.4'
    }
    with open(TEST_CONFIG_PATH, 'w') as configfile:
        config.write(configfile)
        
    yield # This is where the test runs
    
    # Teardown: remove dummy files
    os.remove(TEST_MODEL_PATH)
    os.remove(TEST_PREPROCESSOR_PATH)
    os.remove(TEST_CONFIG_PATH)

@pytest.fixture
def churn_predictor_service(setup_test_environment):
    return ChurnPredictorService(TEST_MODEL_PATH, TEST_PREPROCESSOR_PATH, TEST_CONFIG_PATH)

def test_predict_batch_output_format(churn_predictor_service):
    # Create a sample CSV for testing
    sample_data = {
        'user_id': ['user_1', 'user_2', 'user_3'],
        'subscription_plan': ['basic', 'premium', 'basic'],
        'email': ['a@example.com', 'b@example.com', 'c@example.com'],
        'days_since_signup': [100, 200, 50],
        'monthly_revenue': [50.0, 100.0, 25.0],
        'number_of_logins_last30days': [10, 20, 5],
        'active_features_used': [3, 5, 2],
        'support_tickets_opened': [1, 0, 2],
        'last_login_days_ago': [5, 1, 10],
        'email_opens_last30days': [8, 15, 3],
        'billing_issue_count': [0, 1, 0],
        'last_payment_status': ['success', 'failed', 'success']
    }
    sample_df = pd.DataFrame(sample_data)

    predictions = churn_predictor_service.predict_batch(sample_df)

    assert isinstance(predictions, list)
    assert len(predictions) == 3

    for pred in predictions:
        assert isinstance(pred, dict)
        assert "user_id" in pred
        assert "churn_probability" in pred
        assert "risk_level" in pred
        assert "top_reasons" in pred
        assert isinstance(pred["churn_probability"], float)
        assert isinstance(pred["risk_level"], str)
        assert isinstance(pred["top_reasons"], list)
        assert len(pred["top_reasons"]) <= 2 # Ensure top 2 reasons are extracted

def test_predict_batch_risk_levels(churn_predictor_service):
    # Test different churn probabilities to verify risk level assignment
    
    # Mock X_train_for_shap for LinearExplainer
    dummy_X_train = pd.DataFrame([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]],
                                 columns=['days_since_signup', 'monthly_revenue', 'number_of_logins_last30days',
                                          'active_features_used', 'support_tickets_opened', 'last_login_days_ago',
                                          'email_opens_last30days', 'billing_issue_count',
                                          'subscription_plan_basic', 'subscription_plan_premium'])
    churn_predictor_service.X_train_for_shap = dummy_X_train
    
    sample_data = {
        'user_id': ['user_high', 'user_medium', 'user_low'],
        'subscription_plan': ['basic', 'premium', 'basic'],
        'email': ['h@example.com', 'm@example.com', 'l@example.com'],
        'days_since_signup': [10, 10, 10],
        'monthly_revenue': [10, 10, 10],
        'number_of_logins_last30days': [10, 10, 10],
        'active_features_used': [3, 3, 3],
        'support_tickets_opened': [1, 1, 1],
        'last_login_days_ago': [5, 5, 5],
        'email_opens_last30days': [8, 8, 8],
        'billing_issue_count': [0, 0, 0],
        'last_payment_status': ['success', 'success', 'success']
    }
    sample_df = pd.DataFrame(sample_data)

    # Mock the model's predict_proba to return specific probabilities
    # This is a simplified mock; in a real test, you might mock the entire model
    # Using a real LogisticRegression instance but overriding predict_proba for specific test values
    from sklearn.linear_model import LogisticRegression

    class MockLogisticRegression(LogisticRegression):
        def __init__(self):
            super().__init__()
            # Dummy fit to initialize internal attributes like classes_
            # A minimal fit is required for LogisticRegression to be fully initialized
            # and for predict_proba to work correctly in some contexts (e.g., SHAP)
            # Fit with 10 features as per dummy_X_train in this test
            self.fit(np.array([[0.0] * 10]), np.array([0]))
            self.classes_ = np.array([0, 1]) # Ensure classes_ is set for predict_proba

        def predict_proba(self, X):
            # Return probabilities that map to High, Medium, Low
            # X will have 3 rows for the 3 users
            return np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
        
        def predict(self, X):
            # Based on the probabilities above, predict 1, 1, 0
            return np.array([1, 1, 0])

    churn_predictor_service.model = MockLogisticRegression()
    
    # Ensure the mocked model has coef_ and intercept_ attributes
    # These are typically set after fitting a LogisticRegression model
    if not hasattr(churn_predictor_service.model, 'coef_'):
        churn_predictor_service.model.coef_ = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])
    if not hasattr(churn_predictor_service.model, 'intercept_'):
        churn_predictor_service.model.intercept_ = np.array([0.0])
    
    # Mock the explainer to avoid complex SHAP calculations for this test
    class MockExplainer:
        def shap_values(self, X):
            # Return dummy shap values for 3 users, 2 classes, N features
            # Ensure positive values for 'top_reasons' test
            return [np.array([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]),
                    np.array([[0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1], [0.4, 0.3, 0.2, 0.1]])]

    churn_predictor_service.explainer = MockExplainer()
    
    # Re-set feature names to match the dummy data for SHAP to work
    churn_predictor_service.numerical_features = ['days_since_signup', 'monthly_revenue']
    churn_predictor_service.categorical_features = ['subscription_plan', 'last_payment_status']
    churn_predictor_service._set_feature_names() # This will update self.feature_names

    predictions = churn_predictor_service.predict_batch(sample_df)

    # Assert risk levels based on mocked probabilities
    assert predictions[0]['risk_level'] == 'High' # 0.9 churn prob
    assert predictions[1]['risk_level'] == 'Medium' # 0.5 churn prob
    assert predictions[2]['risk_level'] == 'Low' # 0.2 churn prob

    # Assert top reasons are extracted (based on dummy shap values and feature names)
    # The dummy shap values are positive for all features, so it will pick the first two
    assert len(predictions[0]['top_reasons']) == 2
    assert len(predictions[1]['top_reasons']) == 2
    assert len(predictions[2]['top_reasons']) == 2
    
    # Check if the top reasons are from the feature names
    expected_feature_names = ['days_since_signup', 'monthly_revenue', 'subscription_plan_basic', 'subscription_plan_premium', 'last_payment_status_failed', 'last_payment_status_success']
    for pred in predictions:
        for reason in pred['top_reasons']:
            assert reason in churn_predictor_service.feature_names

# You would add more tests here, e.g., for edge cases, performance, etc.