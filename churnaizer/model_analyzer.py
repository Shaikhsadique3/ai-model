"""Module for analyzing a trained churn prediction model.

This module defines the `ModelAnalyzer` class, which loads a model and preprocessor,
processes data, makes predictions, and generates a model summary report with various metrics.
"""

import pandas as pd
import numpy as np
import joblib
import logging
import os
import json
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

from churnaizer.src.preprocessing import preprocess_data

class ModelAnalyzer:
    """A class to analyze a trained churn prediction model."""

    def __init__(self, model_path, preprocessor_path, config_path):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        self.config_path = config_path
        self.model = None
        self.preprocessor = None
        self.config = self._load_config(config_path)

    def _load_config(self, config_path) -> dict:
        """Loads configuration from config.json."""
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logging.info("Configuration loaded successfully.")
            return config
        except FileNotFoundError:
            logging.error(f"Error: config.json not found at {config_path}")
            raise
        except json.JSONDecodeError:
            logging.error(f"Error: Could not decode JSON from {config_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading configuration: {e}")
            raise

    def _load_model_and_preprocessor(self):
        """Loads the trained model and preprocessor."""
        try:
            self.model = joblib.load(self.model_path)
            logging.info(f"Model loaded from {self.model_path}")
        except FileNotFoundError:
            logging.error(f"Error: Model not found at {self.model_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading model: {e}")
            raise

        try:
            self.preprocessor = joblib.load(self.preprocessor_path)
            logging.info(f"Preprocessor loaded from {self.preprocessor_path}")
        except FileNotFoundError:
            logging.error(f"Error: Preprocessor not found at {self.preprocessor_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading preprocessor: {e}")
            raise

    def _load_data_and_preprocess(self) -> tuple[pd.DataFrame, pd.Series]:
        """Loads the dataset and preprocesses it using the loaded preprocessor."""
        try:
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config['dataset_path'])
            df = pd.read_csv(dataset_path)
            logging.info("Data loaded successfully.")

            X_processed, y, _ = preprocess_data(df.copy(), self.config['categorical_features'], self.config['target_column'])
            logging.info("Data preprocessed successfully.")
            return X_processed, y
        except FileNotFoundError:
            logging.error(f"Error: Dataset not found at {dataset_path}")
            raise
        except Exception as e:
            logging.error(f"Error loading or preprocessing data: {e}")
            raise

    def run_analysis(self):
        """Runs the complete model analysis pipeline."""
        logging.info("Starting model analysis...")
        try:
            self._load_model_and_preprocessor()
            X_processed, y = self._load_data_and_preprocess()

            # ==== Model predictions ====
            y_pred = self.model.predict(X_processed)
            if hasattr(self.model, "predict_proba"):
                y_proba = self.model.predict_proba(X_processed)[:, 1]
            else:
                y_proba = None

            # Get hyperparameters safely
            model_hyperparameters = "Unavailable"
            try:
                if hasattr(self.model, 'get_params') and callable(self.model.get_params):
                    model_hyperparameters = self.model.get_params()
            except AttributeError:
                model_hyperparameters = "Unavailable"
            except Exception as e:
                model_hyperparameters = f"Error retrieving: {e}"

            # ==== Generate report ====
            report = {
                "Model Type": str(type(self.model)),
                "Target Variable": self.config['target_column'],
                "Input Features Used": list(X_processed.columns),
                "Model Hyperparameters": model_hyperparameters,
                "Accuracy": accuracy_score(y, y_pred),
                "F1 Score": classification_report(y, y_pred, output_dict=True)["weighted avg"]["f1-score"],
                "Confusion Matrix": confusion_matrix(y, y_pred).tolist(),
                "ROC-AUC Score": roc_auc_score(y, y_proba) if y_proba is not None else "Not available",
                "Feature Importances": dict(zip(X_processed.columns, self.model.feature_importances_)) if hasattr(self.model, "feature_importances_") else "N/A",
            }

            # ==== Check for Overfitting ====
            if report["Accuracy"] > 0.95:
                report["Overfitting Risk"] = "Accuracy is very high. Consider cross-validation."
            else:
                report["Overfitting Risk"] = "Looks reasonable."

            logging.info("\nModel Summary Report:\n")
            logging.info(json.dumps(report, indent=2))
            logging.info("Model analysis finished.")
        except Exception as e:
            logging.critical(f"An error occurred during model analysis: {e}")

if __name__ == "__main__":
    try:
        # Define paths and parameters
        CONFIG_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\config\\config.json'
        
        # Load config to get model and preprocessor paths
        with open(CONFIG_PATH, 'r') as f:
            config_data = json.load(f)

        MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_data['model_path'])
        PREPROCESSOR_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), config_data['preprocessor_path'])

        analyzer = ModelAnalyzer(MODEL_PATH, PREPROCESSOR_PATH, CONFIG_PATH)
        analyzer.run_analysis()
    except Exception as e:
        logging.critical(f"An error occurred in the main execution block: {e}")

    # ==== Check for Overfitting ====
    if report["\ud83d\udcc8 Accuracy"] > 0.95:
        report["\u26a0\ufe0f Overfitting Risk"] = "\u26a0\ufe0f Accuracy is very high. Consider cross-validation."
    else:
        report["\u26a0\ufe0f Overfitting Risk"] = "\u2705 Looks reasonable."

    # ==== Output JSON summary ====
    logging.info("\nModel Summary Report:\n")
    report = convert_numpy_types(report)

    logging.info(json.dumps(report, indent=2))
    logging.info("Model analysis finished.")