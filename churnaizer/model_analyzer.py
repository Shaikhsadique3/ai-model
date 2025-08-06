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
import configparser
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score

from churnaizer.src.preprocessing import preprocess_data

class ModelAnalyzer:
    """A class to analyze a trained churn prediction model."""

    def __init__(self, model_path, preprocessor_path, config_path):
        self.config_path = config_path
        self.model = None
        self.preprocessor = None
        self.config = self._load_config(config_path)
        self.model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config['paths']['model_path'])
        self.preprocessor_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config['paths']['preprocessor_path'])
        print(f"Constructed Model Path: {self.model_path}")
        print(f"Constructed Preprocessor Path: {self.preprocessor_path}")

    @staticmethod
    def convert_numpy_types(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: ModelAnalyzer.convert_numpy_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [ModelAnalyzer.convert_numpy_types(elem) for elem in obj]
        else:
            return obj

    def _load_config(self, config_path) -> dict:
        """Loads configuration from config.ini."""
        config = configparser.ConfigParser()
        try:
            config.read(config_path)
            logging.info("Configuration loaded successfully.")
            # Convert configparser object to a dictionary for easier access
            config_dict = {section: dict(config[section]) for section in config.sections()}
            config_dict['DEFAULT'] = dict(config.defaults())
            return config_dict
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
            dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), self.config['paths']['dataset_path'])
            df = pd.read_csv(dataset_path)
            logging.info("Data loaded successfully.")

            X_processed, y, _ = preprocess_data(df.copy(), self.config['features']['categorical_features'].split(','), self.config['model']['target_column'], preprocessor=self.preprocessor)
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
                    params = self.model.get_params()
                    # Convert numpy types to standard Python types for JSON serialization
                    model_hyperparameters = ModelAnalyzer.convert_numpy_types(params)
            except AttributeError:
                model_hyperparameters = "Unavailable"
            except Exception as e:
                model_hyperparameters = f"Error retrieving: {e}"

            # ==== Generate report ====
            report = {
                "Model Type": str(type(self.model)),
                "Target Variable": self.config['model']['target_column'],
                "Input Features Used": list(X_processed.columns),
                "Model Hyperparameters": model_hyperparameters,
                "Accuracy": float(accuracy_score(y, y_pred)),
                "F1 Score": float(classification_report(y, y_pred, output_dict=True)["weighted avg"]["f1-score"]),
                "Confusion Matrix": confusion_matrix(y, y_pred).tolist(),
                "ROC-AUC Score": roc_auc_score(y, y_proba) if y_proba is not None else "Not available",
                "Feature Importances": dict(zip(X_processed.columns, self.model.feature_importances_)) if hasattr(self.model, "feature_importances_") else "N/A",
            }

            # ==== Check for Overfitting ====
            if report["Accuracy"] > 0.95:
                report["Overfitting Risk"] = "Accuracy is very high. Consider cross-validation."
            else:
                report["Overfitting Risk"] = "Looks reasonable."

            logging.info("Model analysis finished.")
            return report
        except Exception as e:
            logging.critical(f"An error occurred during model analysis: {e}")
            raise

if __name__ == "__main__":
    try:
        # Define paths and parameters
        CONFIG_PATH = 'c:\\Users\\Sadique\\Desktop\\ai model\\churnaizer\\config\\config.ini'
        
        # Load config to get model and preprocessor paths




        report = None
        analyzer = ModelAnalyzer(None, None, CONFIG_PATH)
        report = analyzer.run_analysis()

        # ==== Check for Overfitting ====
        if report["Accuracy"] > 0.95:
            report["Overfitting Risk"] = "Accuracy is very high. Consider cross-validation."
        else:
            report["Overfitting Risk"] = "Looks reasonable."

        # ==== Output JSON summary ====
        logging.info("\nModel Summary Report:\n")

        clean_report = ModelAnalyzer.convert_numpy_types(report)
        logging.info(json.dumps(clean_report, indent=2))
        logging.info("Model analysis finished.")

    except Exception as e:
        logging.critical(f"An error occurred in the main execution block: {e}")