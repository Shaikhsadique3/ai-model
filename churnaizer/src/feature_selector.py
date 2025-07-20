# rebalance_features.py 
 
import pandas as pd
import numpy as np
import logging
import os
import json
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer 

# Configure logging
log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'churnaizer.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])

# Load configuration
try:
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config', 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    logging.info("Configuration loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: config.json not found at {config_path}")
    exit()
except Exception as e:
    logging.error(f"Error loading configuration: {e}")
    exit()

# Load your cleaned dataset 
try:
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), config['dataset_path'])
    df = pd.read_csv(dataset_path)
    logging.info("Dataset loaded successfully.")
except FileNotFoundError:
    logging.error(f"Error: Dataset not found at {dataset_path}")
    exit()
except Exception as e:
    logging.error(f"Error loading dataset: {e}")
    exit()

# Preprocessing steps (similar to train_model.py)
categorical_features = ['subscription_plan', 'last_payment_status']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ], remainder='passthrough'
)

X = df.drop("churn", axis=1)
y = df["churn"]

X_processed = preprocessor.fit_transform(X)

model = RandomForestClassifier(random_state=42) 
model.fit(X_processed, y) 
 
sfm = SelectFromModel(model, threshold="mean")  # Remove overly dominant features 
X_reduced = sfm.transform(X_processed) 
 
# Optional: Print selected feature names 
# Get feature names after one-hot encoding
ohe_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
all_feature_names = np.concatenate((ohe_feature_names, X.drop(columns=categorical_features).columns.values))

selected_features = all_feature_names[sfm.get_support()] 
logging.info(f"Selected Features: {list(selected_features)}")