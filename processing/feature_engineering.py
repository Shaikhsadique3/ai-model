# rebalance_features.py 
"""Module for feature selection using RandomForestClassifier and SelectFromModel."""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureSelector:
    """A class to perform feature selection on a dataset."""

    def __init__(self, categorical_features: list, target_column: str, threshold: str = "mean"):
        """Initializes the FeatureSelector with categorical features, target column, and selection threshold.

        Args:
            categorical_features (list): List of categorical feature names.
            target_column (str): Name of the target column.
            threshold (str): Threshold for feature selection (e.g., "mean", "median").
        """
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.threshold = threshold
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ],
            remainder='passthrough'
        )
        self.selector = None
        self.selected_feature_names = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Fits the feature selector to the data.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target Series.
        """
        logging.info("Starting feature selection fit...")
        try:
            X_processed = self.preprocessor.fit_transform(X)
            
            # Ensure X_processed is a DataFrame for consistent column handling
            if not isinstance(X_processed, pd.DataFrame):
                # Get feature names after one-hot encoding
                ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
                # Get remaining feature names (numerical/passthrough)
                remaining_features = [col for col in X.columns if col not in self.categorical_features]
                all_feature_names = np.concatenate((ohe_feature_names, remaining_features))
                X_processed = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)

            model = RandomForestClassifier(random_state=42)
            model.fit(X_processed, y)

            self.selector = SelectFromModel(model, threshold=self.threshold)
            self.selector.fit(X_processed, y)

            # Get selected feature names
            self.selected_feature_names = X_processed.columns[self.selector.get_support()].tolist()
            logging.info(f"Selected Features: {self.selected_feature_names}")
            logging.info("Feature selection fit completed.")
        except Exception as e:
            logging.error(f"Error during feature selection fit: {e}")
            raise

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transforms the input DataFrame to include only selected features.

        Args:
            X (pd.DataFrame): Feature DataFrame to transform.

        Returns:
            pd.DataFrame: DataFrame with only selected features.
        """
        logging.info("Transforming data with selected features...")
        if self.selector is None:
            raise RuntimeError("FeatureSelector has not been fitted yet. Call .fit() first.")
        try:
            X_processed = self.preprocessor.transform(X)
            
            # Ensure X_processed is a DataFrame for consistent column handling
            if not isinstance(X_processed, pd.DataFrame):
                ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(self.categorical_features)
                remaining_features = [col for col in X.columns if col not in self.categorical_features]
                all_feature_names = np.concatenate((ohe_feature_names, remaining_features))
                X_processed = pd.DataFrame(X_processed, columns=all_feature_names, index=X.index)

            X_reduced = self.selector.transform(X_processed)
            
            # Convert X_reduced back to DataFrame with selected column names
            if self.selected_feature_names is None:
                raise RuntimeError("Selected feature names not available. Call .fit() first.")
            
            X_reduced_df = pd.DataFrame(X_reduced, columns=self.selected_feature_names, index=X.index)
            logging.info("Data transformation completed.")
            return X_reduced_df
        except Exception as e:
            logging.error(f"Error during feature selection transform: {e}")
            raise

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Fits the feature selector and then transforms the data.

        Args:
            X (pd.DataFrame): Feature DataFrame.
            y (pd.Series): Target Series.

        Returns:
            pd.DataFrame: Transformed DataFrame with selected features.
        """
        self.fit(X, y)
        return self.transform(X)

    def get_selected_feature_names(self) -> list:
        """Returns the list of selected feature names.

        Returns:
            list: List of selected feature names.
        """
        if self.selected_feature_names is None:
            raise RuntimeError("FeatureSelector has not been fitted yet. Call .fit() first.")
        return self.selected_feature_names

if __name__ == "__main__":
    # Example Usage (for testing purposes)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Create dummy data
    data = {
        'feature1': np.random.rand(100),
        'feature2': np.random.rand(100) * 10,
        'feature3': np.random.rand(100) * 100,
        'subscription_plan': np.random.choice(['A', 'B', 'C'], 100),
        'last_payment_status': np.random.choice(['Success', 'Failed'], 100),
        'churn': np.random.randint(0, 2, 100)
    }
    df = pd.DataFrame(data)

    categorical_features = ['subscription_plan', 'last_payment_status']
    target_column = 'churn'

    selector = FeatureSelector(categorical_features=categorical_features, target_column=target_column)
    
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_selected = selector.fit_transform(X, y)
    logging.info(f"Shape of original X: {X.shape}")
    logging.info(f"Shape of selected X: {X_selected.shape}")
    logging.info(f"Selected features: {selector.get_selected_feature_names()}")