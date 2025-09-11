import pandas as pd
import joblib
import os
import logging
import shap
import numpy as np
import configparser
import pandas as pd

from .preprocessing import preprocess_data

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictorService:
    def __init__(self, model_path=None, preprocessor_path=None, config_path=None):
        self.config = self._load_config(config_path)
        self.numerical_features = [f.strip() for f in self.config['model']['numerical_features'].split(',')]
        self.categorical_features = [f.strip() for f in self.config['model']['categorical_features'].split(',')]
        self.target_column = self.config['model']['target_column']
        self.high_churn_threshold = float(self.config['thresholds']['high'])
        self.medium_churn_threshold = float(self.config['thresholds']['medium'])
        self.model = self._load_model(model_path)
        self.preprocessor = self._load_preprocessor(preprocessor_path)
        self._set_feature_names()

        # For LinearExplainer, a background dataset is needed.
        # In a real application, this would be a small, representative sample of the training data.
        # For now, we'll create a dummy one with zeros and the correct feature names.
        if self.feature_names:
            self.X_train_for_shap = pd.DataFrame(np.zeros((1, len(self.feature_names))), columns=self.feature_names)
        else:
            # Fallback if feature_names are not yet set (shouldn't happen if _set_feature_names is called first)
            # This might happen if the model is not yet loaded or feature names cannot be determined.
            # In a robust system, this would be handled by ensuring feature_names are always available.
            # For now, we'll assume a default number of features if feature_names is empty.
            # This assumes the model is already loaded and has a coef_ attribute to infer feature count.
            if hasattr(self.model, 'coef_') and self.model.coef_.size > 0:
                self.X_train_for_shap = pd.DataFrame(np.zeros((1, self.model.coef_.shape[1])))
            else:
                # If model is not loaded or has no coef_, create a minimal dummy DataFrame
                self.X_train_for_shap = pd.DataFrame(np.zeros((1, 1))) # Fallback to a single feature

        # Initialize SHAP explainer
        # Use LinearExplainer for linear models like LogisticRegression
        self.explainer = shap.LinearExplainer(self.model, self.X_train_for_shap)

    def _load_config(self, config_path=None) -> dict:
        config = configparser.ConfigParser()
        try:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.ini')
            config.read(config_path)
            logging.info("Configuration loaded successfully.")
            return config
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    def _load_model(self, model_path):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'churnaizer_saas_model.pkl')
        try:
            model = joblib.load(model_path)
            logging.info(f"Model loaded from {model_path}")
            return model
        except Exception as e:
            logging.error(f"Error loading model from {model_path}: {e}")
            raise

    def _load_preprocessor(self, preprocessor_path):
        if preprocessor_path is None:
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'one_hot_encoder.pkl')
        try:
            preprocessor = joblib.load(preprocessor_path)
            logging.info(f"Preprocessor loaded from {preprocessor_path}")
            return preprocessor
        except Exception as e:
            logging.error(f"Error loading preprocessor from {preprocessor_path}: {e}")
            raise

    def _get_risk_level(self, probability: float) -> str:
        high_threshold = float(os.getenv('CHURN_HIGH_THRESHOLD', self.config['thresholds']['high']))
        medium_threshold = float(os.getenv('CHURN_MEDIUM_THRESHOLD', self.config['thresholds']['medium']))

        if probability > high_threshold:
            return "High"
        elif probability >= medium_threshold:
            return "Medium"
        else:
            return "Low"

    def _get_top_reasons(self, shap_values, feature_names, top_n=2):
        # For a single prediction, shap_values will be a 1D array
        # For multi-class, it might be 2D, we take the values for the positive class (index 1)
        if len(shap_values.shape) > 1:
            shap_values = shap_values[1] # Assuming binary classification, get values for positive class

        # Get absolute SHAP values and sort them to find the most impactful features
        abs_shap_values = np.abs(shap_values)
        sorted_indices = np.argsort(abs_shap_values)[::-1]

        top_reasons = []
        for i in sorted_indices[:top_n]:
            feature_name = feature_names[i]
            # Determine if the feature positively or negatively contributes to churn
            # A positive SHAP value for the positive class (churn) means the feature increases churn probability
            if shap_values[i] > 0:
                top_reasons.append(f"{feature_name}") # Removed (positive impact) as it's implied by being a top reason for churn
            else:
                # If a feature has a negative SHAP value for churn, it means it decreases churn probability.
                # We might not want to list these as 'reasons for churn' unless specifically requested.
                # For now, we'll only include features that positively contribute to churn risk.
                pass
        return top_reasons

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model or preprocessor not loaded. Call load_model_and_preprocessor first.")

        logging.info(f"Starting batch prediction for {len(df)} users.")
        
        # Ensure user_id is handled correctly and not used for prediction
        if 'user_id' not in df.columns:
            raise ValueError("Input DataFrame must contain a 'user_id' column.")
            
        user_ids = df['user_id']
        df_for_prediction = df.drop(columns=['user_id'])

        # Transform the DataFrame for prediction using the new function
        X_processed = transform_data_for_prediction(
            df=df_for_prediction,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            preprocessor=self.preprocessor
        )

        # Ensure the columns of X_processed match the order and presence of self.feature_names
        # Add any missing columns with 0 and reorder
        missing_cols = set(self.feature_names) - set(X_processed.columns)
        for c in missing_cols:
            X_processed[c] = 0
        X_final = X_processed[self.feature_names] # Reorder columns

        # Predict churn probabilities
        churn_probabilities = self.model.predict_proba(X_final)[:, 1]

        # Calculate SHAP values
        if self.explainer is None:
            self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(X_final)

        results = []
        for i, prob in enumerate(churn_probabilities):
            user_id = user_ids.iloc[i]
            risk_level = self._get_risk_level(prob)
            # SHAP values for the current user
            user_shap_values = shap_values[1][i] if isinstance(shap_values, list) else shap_values[i]
            top_reasons = self._get_top_reasons(user_shap_values, self.feature_names)

            results.append({
                "user_id": user_id,
                "churn_probability": round(float(prob), 4),
                "risk_level": risk_level,
                "top_reasons": top_reasons
            })

        logging.info("Batch prediction completed.")
        return pd.DataFrame(results)

    def _set_feature_names(self):
        # This method should be called after preprocessor is loaded
        # It constructs the full list of feature names that the model expects
        if self.preprocessor is None:
            raise RuntimeError("Preprocessor not loaded. Cannot set feature names.")

        # Get numerical features from config or infer from a dummy dataframe
        # For now, let's assume we know the numerical features or can get them from the model's training data
        # A more robust solution would involve saving the numerical feature names during training.
        numerical_features_str = self.config['model'].get('numerical_features', '')
        numerical_features = [f.strip() for f in numerical_features_str.split(',') if f.strip()]

        categorical_features_str = self.config['model'].get('categorical_features', '')
        categorical_features = [f.strip() for f in categorical_features_str.split(',') if f.strip()]

        # Get feature names from the one-hot encoder
        ohe_feature_names = []
        if self.preprocessor and hasattr(self.preprocessor, 'get_feature_names_out'):
            ohe_feature_names = self.preprocessor.get_feature_names_out(categorical_features).tolist()
        elif self.preprocessor and hasattr(self.preprocessor, 'categories_'): # For older versions or different encoders
            for i, cat_col in enumerate(categorical_features):
                for cat in self.preprocessor.categories_[i]:
                    ohe_feature_names.append(f"{cat_col}_{cat}")

        self.feature_names = numerical_features + ohe_feature_names

# Example usage (for testing purposes)
if __name__ == "__main__":
    # Create dummy data for testing
    data = {
        'user_id': [f'user_{i}' for i in range(1, 11)],
        'subscription_plan': ['premium', 'basic', 'premium', 'basic', 'premium', 'basic', 'premium', 'basic', 'premium', 'basic'],
        'days_since_signup': np.random.randint(10, 500, 10),
        'monthly_revenue': np.random.rand(10) * 100,
        'number_of_logins_last30days': np.random.randint(0, 30, 10),
        'active_features_used': np.random.randint(1, 10, 10),
        'support_tickets_opened': np.random.randint(0, 5, 10),
        'last_login_days_ago': np.random.randint(0, 60, 10),
        'email_opens_last30days': np.random.randint(0, 15, 10),
        'billing_issue_count': np.random.randint(0, 3, 10),
        'last_payment_status': ['success', 'failed', 'success', 'success', 'failed', 'success', 'success', 'failed', 'success', 'success']
    }
    dummy_df = pd.DataFrame(data)

    # Save dummy data to a CSV file
    dummy_csv_path = "dummy_users.csv"
    dummy_df.to_csv(dummy_csv_path, index=False)
    logging.info(f"Dummy CSV created at {dummy_csv_path}")

    # Instantiate the service
    # Ensure model and preprocessor exist in churnaizer/models/
    logging.info(f"Feature names set: {self.feature_names}")
    # You might need to run churnaizer/src/train.py first to generate them
    try:
        predictor_service = ChurnPredictorService()
        # Perform batch prediction
        predictions_df = predictor_service.predict_batch(dummy_df)
        print("\nBatch Predictions:")
        print(predictions_df.to_string())
    except Exception as e:
        logging.error(f"Error during example usage: {e}")

    # Clean up dummy CSV
    if os.path.exists(dummy_csv_path):
        os.remove(dummy_csv_path)
        logging.info(f"Cleaned up {dummy_csv_path}")