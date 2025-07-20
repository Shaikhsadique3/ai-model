import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from src.preprocessing import preprocess_data
import pickle
import logging
import os
import json
from typing import Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChurnPredictor:
    """
    A class to encapsulate the churn prediction model training and evaluation process.
    """
    def __init__(self, config_path: str = 'config/config.json'):
        """
        Initializes the ChurnPredictor with configuration settings.

        Args:
            config_path (str): Path to the configuration JSON file.
        """
        self.config = self._load_config(config_path)
        self.dataset_path = self.config.get('dataset_path')
        self.model_path = self.config.get('model_path')
        self.selected_features = self.config.get('selected_features', [])
        self.model = None
        self.one_hot_encoder = None

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads configuration from a JSON file.

        Args:
            config_path (str): Path to the configuration JSON file.
\        Returns:
            Dict[str, Any]: Loaded configuration dictionary.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info(f"Configuration loaded successfully from {config_path}")
            return config
        except FileNotFoundError:
            logging.error(f"Configuration file not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error decoding JSON from {config_path}. Check file format.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred while loading config: {e}")
            raise

    def load_data(self) -> pd.DataFrame:
        """
        Loads the dataset from the configured path.

        Returns:
            pd.DataFrame: Loaded DataFrame.
        """
        try:
            df = pd.read_csv(self.dataset_path)
            logging.info(f"Dataset loaded successfully from {self.dataset_path}")
            return df
        except FileNotFoundError:
            logging.error(f"Dataset not found at {self.dataset_path}. Please check the path in config.json.")
            raise
        except Exception as e:
            logging.error(f"An error occurred while loading the dataset: {e}")
            raise



    def train_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the XGBoost classifier model.

        Args:
            X (pd.DataFrame): Features for training.
            y (pd.Series): Target variable for training.
        """
        logging.info("Starting model training...")
        try:
            # Define the parameter grid for GridSearchCV
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'subsample': [0.7, 0.9],
                'colsample_bytree': [0.7, 0.9]
            }

            # Initialize XGBClassifier
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)

            # Initialize GridSearchCV
            grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)

            # Fit GridSearchCV
            grid_search.fit(X, y)

            self.model = grid_search.best_estimator_
            logging.info("Model training completed. Best parameters found:")
            logging.info(grid_search.best_params_)
        except Exception as e:
            logging.error(f"An error occurred during model training: {e}")
            raise

    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Evaluates the trained model and logs performance metrics.

        Args:
            X (pd.DataFrame): Features for evaluation.
            y (pd.Series): Target variable for evaluation.
        """
        logging.info("Starting model evaluation...")
        if self.model is None:
            logging.error("Model has not been trained yet. Please train the model first.")
            raise ValueError("Model not trained.")

        try:
            y_pred = self.model.predict(X)
            accuracy = self.model.score(X, y)
            f1 = f1_score(y, y_pred)

            logging.info(f"Model Accuracy: {accuracy:.2f}")
            logging.info(f"Model F1 Score: {f1:.2f}")
            logging.info("Classification Report:\n" + classification_report(y, y_pred))
            logging.info("Confusion Matrix:\n" + str(confusion_matrix(y, y_pred)))

            # Feature Importance
            if hasattr(self.model, 'feature_importances_'):
                feature_importances = pd.Series(self.model.feature_importances_, index=X.columns)
                logging.info("Feature Importances:\n" + str(feature_importances.sort_values(ascending=False)))
            logging.info("Model evaluation completed.")
        except Exception as e:
            logging.error(f"An error occurred during model evaluation: {e}")
            raise

    def save_model(self) -> None:
        """
        Saves the trained model to the configured path.
        """
        if self.model is None:
            logging.error("No model to save. Please train the model first.")
            raise ValueError("No model to save.")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(self.model, f)
            logging.info(f"Model saved successfully to {self.model_path}")
        except Exception as e:
            logging.error(f"An error occurred while saving the model: {e}")
            raise

    def run(self):
        """
        Orchestrates the entire churn prediction pipeline.
        """
        try:
            df = self.load_data()
            
            target_column = 'churn'
            if target_column not in df.columns:
                logging.error(f"Target column '{target_column}' not found in the dataset.")
                raise ValueError(f"Target column '{target_column}' not found.")

            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
            if target_column in categorical_features:
                categorical_features.remove(target_column)

            X, y, self.one_hot_encoder = preprocess_data(df, categorical_features, target_column)
            
            # Split data into training and testing sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            self.train_model(X_train, y_train)
            self.evaluate_model(X_test, y_test)
            self.save_model()
            logging.info("Churn prediction pipeline completed successfully.")
        except Exception as e:
            logging.critical(f"Churn prediction pipeline failed: {e}")

if __name__ == "__main__":
    predictor = ChurnPredictor(config_path='config/config.json')
    predictor.run()