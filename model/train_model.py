"""
train.py
Reproducible retraining script for Churnaizer (scikit-learn 1.6.1).
- Loads client CSV (client_data_raw.csv)
- Creates churn label if missing (clear heuristic)
- Feature engineering + preprocessing pipeline
- Trains a model (HistGradientBoostingClassifier)
- Evaluates (cv ROC AUC + test metrics)
- Saves artifacts: model.pkl, train_metrics.json, model_card.txt, predictions CSV
- Optionally attempts ONNX export (if skl2onnx available)
"""

import os, json, uuid
from datetime import datetime
import numpy as np
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import requests
import matplotlib.pyplot as plt
from processing.feature_engineering import feature_engineer_and_preprocess

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
)

# Optional ONNX libs (import inside try)
# from skl2onnx import convert_sklearn
# from skl2onnx.common.data_types import FloatTensorType

# -------------------- CONFIG --------------------
DATA_CSV = os.getenv("CLIENT_DATA_CSV", "client_data_raw.csv")
USP_CSV = os.getenv("USP_CSV", "product_usp.csv")
MODEL_OUT = os.getenv("MODEL_OUT", "model.pkl")
PIPELINE_OUT = os.getenv("PIPELINE_OUT", "pipeline.pkl")
METRICS_OUT = os.getenv("METRICS_OUT", "train_metrics.json")
MODEL_CARD = os.getenv("MODEL_CARD", "model_card.txt")
PRED_OUT = os.getenv("PRED_OUT", "churn_predictions.csv")
AI_EMAILS_OUT = os.getenv("AI_EMAILS_OUT", "ai_emails_sample.csv")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Thresholds used when label is not present:
INACTIVITY_DAYS_THRESHOLD = 30  # no login for 30+ days => candidate churn
BILLING_FAIL_STATUSES = {"cancelled", "failed", "past_due"}

# -------------------- HELPERS --------------------
def download_model_placeholder():
    """(Optional) If you have a pre-trained model URL, place a download routine here."""
    pass

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found in workspace. Please upload dataset.")
    return pd.read_csv(path)

# -------------------- LABEL CREATION (if missing) --------------------
def create_churn_label(df: pd.DataFrame) -> pd.Series:
    """
    Heuristic churn label:
    - If billing_status (or last_payment_status) in failed/cancelled -> churn=1
    - Else if last_login exists and days since last_login > INACTIVITY_DAYS_THRESHOLD -> churn=1
    - Else churn=0
    """
    df = df.copy()
    # unify possible billing columns
    billing_cols = [c for c in df.columns if c.lower() in ("billing_status", "last_payment_status", "payment_status", "status")]
    last_login_cols = [c for c in df.columns if c.lower() in ("last_login", "last_active", "last_seen", "last_login_date")]
    # create normalized columns
    billing_col = billing_cols[0] if billing_cols else None
    last_login_col = last_login_cols[0] if last_login_cols else None

    churn = pd.Series(0, index=df.index)

    if billing_col:
        billing_vals = df[billing_col].astype(str).str.lower().fillna("")
        churn |= billing_vals.isin([s.lower() for s in BILLING_FAIL_STATUSES])

    if last_login_col:
        try:
            last_login_ts = pd.to_datetime(df[last_login_col], errors="coerce")
            days_since_last = (pd.Timestamp.now() - last_login_ts).dt.days.fillna(9999).astype(int)
            churn |= days_since_last > INACTIVITY_DAYS_THRESHOLD
        except Exception:
            # ignore parse errors
            pass

    # If nothing flagged, fallback: customers with 0 monthly_revenue and trial flag = True -> risk of churn
    if "monthly_revenue" in df.columns and "trial_not_converted" in df.columns:
        churn |= ((df["monthly_revenue"].fillna(0) == 0) & (df["trial_not_converted"].fillna(True) == True))

    return churn.astype(int)

# -------------------- PREP + TRAIN --------------------
def prepare_features_and_target(df: pd.DataFrame):
    # Features chosen (explainable & robust)
    feature_cols = [
        "days_since_signup",        # numeric (if missing we'll fill with median)
        "monthly_revenue",         # numeric
        "number_of_logins_last30days",  # numeric
        "active_features_used",    # numeric
        "support_tickets_opened",  # numeric (optional)
        "billing_issue_count",     # numeric (optional)
        "subscription_plan",       # categorical
        "last_login_days_ago",     # numeric (from feature engineering)
        "engagement_score",        # numeric (from feature engineering)
        "satisfaction_trend"       # numeric (from feature engineering)
    ]
    # Keep only existing ones; missing numeric will be imputed later
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols].copy()
    # Target:
    if "churned" in df.columns:
        y = df["churned"].astype(int)
    elif "is_churn" in df.columns:
        y = df["is_churn"].astype(int)
    else:
        y = create_churn_label(df)

    return X, y, feature_cols

def build_preprocessing_pipeline(X):
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    numeric_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    cat_transform = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transform, numeric_features),
        ("cat", cat_transform, categorical_features)
    ], remainder="drop")
    return preprocessor, numeric_features, categorical_features

def train_and_evaluate(X, y, feature_cols):
    # split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)

    # Build preprocessing pipeline based on X_train
    preprocessor, numeric_features, categorical_features = build_preprocessing_pipeline(X_train)

    model = HistGradientBoostingClassifier(random_state=RANDOM_STATE, max_iter=300)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # cross-val ROC AUC
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    cv_mean = float(np.mean(cv_scores))
    cv_std = float(np.std(cv_scores))

    # fit on train and evaluate on test
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "cv_roc_auc_mean": cv_mean,
        "cv_roc_auc_std": cv_std,
        "test_roc_auc": float(roc_auc_score(y_test, y_proba)),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred, zero_division=0))
    }
    # confusion matrix and classification report text
    metrics["confusion_matrix"] = confusion_matrix(y_test, y_pred).tolist()
    metrics["classification_report"] = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Save model and pipeline
    joblib.dump(pipeline, MODEL_OUT, compress=3)
    joblib.dump(pipeline, PIPELINE_OUT, compress=3)

    # Save predictions for later use
    test_results = X_test.copy()
    test_results["y_true"] = y_test.values
    test_results["y_pred"] = y_pred
    test_results["y_proba"] = y_proba
    test_results.to_csv(PRED_OUT, index=False)

    return pipeline, metrics, test_results

# -------------------- MAIN --------------------
def main():
    print("1) Loading data...")
    df = safe_read_csv(DATA_CSV)

    # Apply feature engineering before splitting
    df = feature_engineer_and_preprocess(df)

    print("2) Preparing features & target...")
    X, y, feature_cols = prepare_features_and_target(df)

    print(f"> Features used: {feature_cols}; samples={len(X)}; positive_rate={float(y.mean()):.3f}")

    print("3) Training and evaluating model...")
    pipeline, metrics, test_results = train_and_evaluate(X, y, feature_cols)

    print("4) Saving metrics & model card...")
    now = datetime.utcnow().isoformat()
    meta = {
        "created_at": now,
        "sklearn_version": __import__("sklearn").__version__,
        "n_samples": int(len(X)),
        "feature_columns": feature_cols
    }
    train_summary = {"meta": meta, "metrics": metrics}
    with open(METRICS_OUT, "w") as f:
        json.dump(train_summary, f, indent=2)

    with open(MODEL_CARD, "w") as f:
        f.write("Churnaizer Model Card\n")
        f.write("=====================\n")
        f.write(json.dumps(train_summary, indent=2))

    print("5) Attempting optional ONNX export (skl2onnx must be installed)...\n")
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        import onnx
        # need a numeric input shape, build sample input
        # The pipeline now includes preprocessing, so we need to get the number of features
        # after preprocessing for ONNX export. This is a bit tricky as ONNX doesn't directly
        # support ColumnTransformer in all cases. For simplicity, we'll use the number of
        # features from the preprocessed X_test, which is available in test_results.
        # A more robust solution would involve inspecting the preprocessor's output shape.
        n_features = pipeline.named_steps['preprocessor'].transform(X_test.head(1)).shape[1]
        initial_type = [("input", FloatTensorType([None, n_features]))]
        onx = convert_sklearn(pipeline, initial_types=initial_type)
        onnx_path = MODEL_OUT.replace(".pkl", ".onnx")
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())
        print(f"ONNX model exported: {onnx_path}")
    except Exception as e:
        print(f"ONNX export skipped or failed: {e}")

    print("Done. Artifacts:")
    print(f" - Pipeline/Model (joblib): {MODEL_OUT}")
    print(f" - Predictions CSV: {PRED_OUT}")
    print(f" - Metrics JSON: {METRICS_OUT}")
    print(f" - Model card: {MODEL_CARD}")

def train_churn_model(file_path):
    """
    Loads preprocessed data, trains a churn prediction model, and saves the model.

    Args:
        file_path (str): The path to the preprocessed_saas_churn.csv file.
    """
    # This function is now deprecated as preprocessing is handled within the main pipeline.
    # It will be removed or refactored to load raw data and use the saved pipeline.
    print("train_churn_model is deprecated and will be removed or refactored.")
    pass

if __name__ == "__main__":
    # The main function now handles loading raw data and performing feature engineering
    # and preprocessing within the pipeline.
    # The preprocessed_saas_churn.csv is no longer directly used for training here.
    # The original raw data CSV should be used.
    # For now, we'll assume DATA_CSV points to the raw data.
    # preprocessed_file_path = r'c:\Users\Sadique\Desktop\ai model\data\preprocessed_saas_churn.csv'
    # train_churn_model(preprocessed_file_path) # This call is now effectively a no-op
    main()