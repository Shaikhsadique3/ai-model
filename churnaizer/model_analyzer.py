import joblib
import pandas as pd
import configparser
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import json
import os

# ==== Load model and data ====
model_path = "churnaizer/models/churnaizer_saas_model.pkl"  # \ud83d\udd02 Adjust path
data_path = "churnaizer/data/enhanced_saas_churn_data.csv"  # \ud83d\udd02 Adjust path

# Load configuration
config = configparser.ConfigParser()
config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.ini')
config.read(config_path)

categorical_features = config['preprocessing']['categorical_features'].split(',')
target_column = config['model']['target_column']

model = joblib.load(model_path)
preprocessor_path = os.path.join(os.path.dirname(__file__), 'models', 'one_hot_encoder.pkl')
preprocessor = joblib.load(preprocessor_path)
data = pd.read_csv(data_path)

# Replicate preprocessing steps from src/preprocessing.py
# Convert relevant columns to appropriate types first
for col in ['total_usage_minutes', 'monthly_avg_bill', 'customer_service_interactions']:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')

# ==== Detect target variable ====
# target_col is now loaded from config.ini

# ==== Separate features and target ====
X = data.drop(columns=[target_column])
y = data[target_column]

# Identify numerical and categorical columns in X
numerical_cols = X.select_dtypes(include=np.number).columns.tolist()
all_categorical_cols = [col for col in X.columns if col in categorical_features or X[col].dtype == 'object']
all_categorical_cols = list(dict.fromkeys(all_categorical_cols))

# Handle missing values in numerical features
for col in numerical_cols:
    if X[col].isnull().any():
        imputer = SimpleImputer(strategy='mean')
        X[col] = imputer.fit_transform(X[[col]])

# Handle missing values in categorical features and convert to category dtype
for col in all_categorical_cols:
    if col in X.columns:
        X[col] = X[col].fillna('Missing').astype('category')

# Only encode columns that are actually categorical and present in X
cols_to_encode = [col for col in all_categorical_cols if col in X.columns]

X_categorical_encoded = pd.DataFrame()
if cols_to_encode:
    X_categorical_encoded = pd.DataFrame(preprocessor.transform(X[cols_to_encode]),
                                         columns=preprocessor.get_feature_names_out(cols_to_encode),
                                         index=X.index)

# Drop original categorical columns from X before concatenating
X_numerical = X.drop(columns=cols_to_encode, errors='ignore')

X_processed = pd.concat([X_numerical, X_categorical_encoded], axis=1)

# ==== Model predictions ====
y_pred = model.predict(X_processed)
if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_processed)[:, 1]
else:
    y_proba = None

# Get hyperparameters safely
model_hyperparameters = "Unavailable"
try:
    if hasattr(model, 'get_params') and callable(model.get_params):
        model_hyperparameters = model.get_params()
except AttributeError:
    model_hyperparameters = "Unavailable"
except Exception as e:
    model_hyperparameters = f"Error retrieving: {e}"

# ==== Generate report ====
report = {
    "\ud83e\udde0 Model Type": str(type(model)),
    "\ud83c\udfaf Target Variable": target_column,
    "\ud83d\udcca Input Features Used": list(X.columns),
    "\ud83d\uddd1\ufe0f Ignored/Dropped Features": list(set(data.columns) - set(X.columns) - {target_column}),
    "\u2699\ufe0f Model Hyperparameters": model_hyperparameters,
    "\ud83d\udcc8 Accuracy": accuracy_score(y, y_pred),
     "\ud83d\udcc9 F1 Score": classification_report(y, y_pred, output_dict=True)["weighted avg"]["f1-score"],
     "\ud83d\udcca Confusion Matrix": confusion_matrix(y, y_pred).tolist(),
     "\ud83d\udcc9 ROC-AUC Score": roc_auc_score(y, y_proba) if y_proba is not None else "Not available",
     "\u2b50 Feature Importances": dict(zip(X.columns, model.feature_importances_)) if hasattr(model, "feature_importances_") else "N/A",
 }

# ==== Check for Overfitting ====
if report["\ud83d\udcc8 Accuracy"] > 0.95:
    report["\u26a0\ufe0f Overfitting Risk"] = "\u26a0\ufe0f Accuracy is very high. Consider cross-validation."
else:
    report["\u26a0\ufe0f Overfitting Risk"] = "\u2705 Looks reasonable."

# ==== Output JSON summary ====
print("\nModel Summary Report:\n")
def convert_numpy_types(obj):
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(elem) for elem in obj]
    else:
        return obj

report = convert_numpy_types(report)

print(json.dumps(report, indent=2))
# Bonus: Add cross-validation section (optional)
# You can extend this script to include cross_val_score() from sklearn.model_selection to further validate generalization.