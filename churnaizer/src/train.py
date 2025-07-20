import pandas as pd
import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
from typing import Tuple
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from churnaizer.src.preprocessing import preprocess_data

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

# Configure logging
log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'churnaizer.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])

class ChurnPredictor:
    """A class to handle data loading, preprocessing, model training, evaluation, and saving."""

    def __init__(self, config_path=None):
        """Initializes the ChurnPredictor with model and preprocessor as None."""
        self.model = None
        self.preprocessor = None
        self.config = self._load_config(config_path)

    def _load_config(self, config_path=None) -> dict:
        """Loads configuration from config.json.
        Args:
            config_path (str, optional): Path to the config file. Defaults to None.
        """
        try:
            if config_path is None:
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info("Configuration loaded successfully.")
            return config
        except FileNotFoundError:
            logging.error(f"Error: config.json not found at {config_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """Loads the dataset from the configured path and handles missing values.

        Returns:
            pd.DataFrame: The loaded and cleaned DataFrame.
        """
        try:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config['dataset_path'])
            df = pd.read_csv(dataset_path)
            logging.info("Dataset loaded successfully.")

            # Handle missing values
            for col in df.columns:
                if df[col].dtype == 'object':
                    df[col].fillna(df[col].mode()[0], inplace=True)
                else:
                    df[col].fillna(df[col].mean(), inplace=True)
            return df
        except FileNotFoundError:
            logging.error(f"Error: Dataset not found at {dataset_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading dataset: {e}")
            raise



    def train_model(self, X: np.ndarray, y: pd.Series) -> Tuple[np.ndarray, pd.Series]:
        """Trains the XGBoost model using GridSearchCV and SMOTE for class balancing.

        Args:
            X (np.ndarray): Processed features.
            y (pd.Series): Target variable.

        Returns:
            Tuple[np.ndarray, pd.Series]: Test features and target variable.
        """
        try:
            # Balance classes using SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            logging.info("Class balancing with SMOTE done.")

            # Train-test split
            X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

            # XGBoost Classifier with GridSearchCV
            xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [3, 5],
                'learning_rate': [0.01, 0.1]
            }
            grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=0) # Set verbose to 0 for cleaner logs
            grid_search.fit(X_train, y_train)

            self.model = grid_search.best_estimator_
            logging.info("Model trained successfully.")
            return X_test, y_test
        except Exception as e:
            logging.error(f"Error during model training: {e}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: pd.Series, original_columns: pd.Index):
        """Evaluates the trained model and logs performance metrics and feature importances.

        Args:
            X_test (np.ndarray): Test features.
            y_test (pd.Series): Test target variable.
            original_columns (pd.Index): Original column names before preprocessing.
        """
        try:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)

            logging.info(f"Model Accuracy: {accuracy:.2f}")
            logging.info(f"Model F1 Score: {f1:.2f}")
            logging.info("Classification Report:\n" + classification_report(y_test, y_pred))
            logging.info("Confusion Matrix:\n" + str(confusion_matrix(y_test, y_pred)))

            # Feature Importance
            feature_importances = self.model.feature_importances_
            # Get feature names after one-hot encoding
            # Use the stored preprocessor to get feature names
            all_feature_names = X_test.columns # X_test already has the processed column names


            importance_df = pd.DataFrame({
                'Feature': all_feature_names,
                'Importance': feature_importances
            }).sort_values(by='Importance', ascending=False)
            importance_df['Percentage'] = importance_df['Importance'] / importance_df['Importance'].sum() * 100
            logging.info("Top Features:\n" + importance_df.to_string())
        except Exception as e:
            logging.error(f"Error during model evaluation: {e}")
            raise

    def save_model(self):
        """Saves the trained model to the configured path."""
        try:
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config['model_path'])
            joblib.dump(self.model, model_path)
            logging.info(f"Model saved at {model_path}")
        except Exception as e:
            logging.error(f"Error saving model: {e}")
            raise

    def run(self):
        """Runs the complete churn prediction pipeline: data loading, preprocessing, training, evaluation, and saving."""
        logging.info("Starting churn prediction pipeline...")
        try:
            df = self.load_data()
            X_processed, y, ohe = preprocess_data(df.copy(), self.config['categorical_features'], self.config['target_column'])
            self.preprocessor = ohe # Store the fitted OHE
            X_test, y_test = self.train_model(X_processed, y)
            self.evaluate_model(X_test, y_test, X_processed.columns) # Pass X_processed.columns for feature names
            self.save_model()
            self.save_preprocessor(ohe) # Save the preprocessor
            logging.info("Churn prediction pipeline completed successfully.")
        except Exception as e:
            logging.critical(f"Pipeline failed: {e}")

    def save_preprocessor(self, preprocessor):
        """Saves the fitted preprocessor (OneHotEncoder) to the configured path."""
        try:
            preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config['preprocessor_path'])
            joblib.dump(preprocessor, preprocessor_path)
            logging.info(f"Preprocessor saved at {preprocessor_path}")
        except Exception as e:
            logging.error(f"Error saving preprocessor: {e}")
            raise


if __name__ == "__main__":
    try:
        predictor = ChurnPredictor()
        predictor.run()
    except Exception as e:
        logging.error(f"An error occurred in the main execution block: {e}")