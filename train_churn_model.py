import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import joblib
import pandas as pd

# Step 1: Load and Merge Datasets
print("Loading datasets...")
telco_df = pd.read_csv('telco_churn.csv')
print(f"Shape of telco_df after loading: {telco_df.shape}")
synthetic_df = pd.read_csv('synthetic_churn_data_7000.csv')
print(f"Shape of synthetic_df after loading: {synthetic_df.shape}")

# Drop customerID early as it's not a feature
if 'customerID' in telco_df.columns:
    telco_df.drop('customerID', axis=1, inplace=True)
if 'customerID' in synthetic_df.columns:
    synthetic_df.drop('customerID', axis=1, inplace=True)

# Standardize column names for merging
telco_df.rename(columns={'tenure': 'days_since_signup', 'MonthlyCharges': 'monthly_revenue', 'Churn': 'churn'}, inplace=True)

# Convert 'TotalCharges' to numeric in telco_df, coercing errors to NaN
telco_df['TotalCharges'] = pd.to_numeric(telco_df['TotalCharges'], errors='coerce')

# Drop rows with NaN values that resulted from 'TotalCharges' conversion
telco_df.dropna(subset=['TotalCharges'], inplace=True)
print(f"Shape of telco_df after dropping TotalCharges NaNs: {telco_df.shape}")

print(f"Shape of telco_df after TotalCharges handling: {telco_df.shape}")
print(f"Shape of synthetic_df after TotalCharges handling: {synthetic_df.shape}")

# Ensure 'churn' column is numeric (1 for churn, 0 for no churn)
print(f"Unique values in telco_df['churn'] before conversion: {telco_df['churn'].unique()}")
print(f"Value counts in telco_df['churn'] before conversion:\n{telco_df['churn'].value_counts()}")
telco_df['churn'] = telco_df['churn'].apply(lambda x: 1 if x == 'Yes' else 0)
print(f"Unique values in telco_df['churn'] after conversion: {telco_df['churn'].unique()}")
print(f"Value counts in telco_df['churn'] after conversion:\n{telco_df['churn'].value_counts()}")
synthetic_df['churn'] = synthetic_df['churn'].astype(int)
print(f"Unique values in synthetic_df['churn'] after conversion: {synthetic_df['churn'].unique()}")
print(f"Value counts in synthetic_df['churn'] after conversion:\n{synthetic_df['churn'].value_counts()}")

print(f"Shape of telco_df after churn conversion: {telco_df.shape}")
print(f"Shape of synthetic_df after churn conversion: {synthetic_df.shape}")

# Align columns before concatenation
# Identify common columns and columns unique to each dataframe
telco_cols = set(telco_df.columns)
synthetic_cols = set(synthetic_df.columns)

common_cols = list(telco_cols.intersection(synthetic_cols))

# For columns unique to telco_df, add them to synthetic_df with default values (e.g., 0 or appropriate placeholder)
for col in telco_cols - synthetic_cols:
    if col not in ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod', 'TotalCharges']:
        if telco_df[col].dtype == 'object':
            synthetic_df[col] = 'No'
        else:
            synthetic_df[col] = 0

# For columns unique to synthetic_df, add them to telco_df with default values
for col in synthetic_cols - telco_cols:
    if synthetic_df[col].dtype == 'object':
        telco_df[col] = 'No'
    else:
        telco_df[col] = 0

# Ensure both dataframes have the same order of columns
telco_df = telco_df[list(common_cols) + list(telco_cols - synthetic_cols)]
synthetic_df = synthetic_df[list(common_cols) + list(synthetic_cols - telco_cols)]

print(f"Shape of telco_df after column alignment: {telco_df.shape}")
print(f"Shape of synthetic_df after column alignment: {synthetic_df.shape}")



# Concatenate them into a single DataFrame
combined_df = pd.concat([telco_df, synthetic_df], ignore_index=True)
print(f"Shape of combined_df after concatenation: {combined_df.shape}")

# Drop customerID after concatenation if it still exists
if 'customerID' in combined_df.columns:
    combined_df.drop('customerID', axis=1, inplace=True)
print(f"Shape of combined_df after dropping customerID: {combined_df.shape}")

# Remove duplicates and nulls
combined_df.drop_duplicates(inplace=True)
print(f"Shape of combined_df after dropping duplicates: {combined_df.shape}")

# Check for nulls before final dropna
print("Null values before final dropna:")
print(combined_df.isnull().sum()[combined_df.isnull().sum() > 0])

# Inspect nulls before dropping
print("\nNulls before final dropna:")
print(combined_df.isnull().sum()[combined_df.isnull().sum() > 0])

# Fill missing values with appropriate defaults instead of dropping rows
for col in combined_df.columns:
    if combined_df[col].dtype == 'object':
        combined_df[col].fillna('Unknown', inplace=True)
    else:
        combined_df[col].fillna(0, inplace=True)
print(f"Shape of combined_df after filling NaNs: {combined_df.shape}")

print(f"Shape of combined_df before feature engineering: {combined_df.shape}")



# Step 2: Feature Engineering
# Add new features
combined_df['days_inactive'] = 30 - combined_df['number_of_logins_last30days']
combined_df['engagement_score'] = combined_df['active_features_used'] / 10
combined_df['support_tickets_ratio'] = combined_df['support_tickets_opened'] / (combined_df['days_since_signup'] + 1)

# Drop irrelevant columns
irrelevant_cols = [
    'customerID', 'SeniorCitizen', 'Contract', 'PaymentMethod', 'TotalCharges',
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'PaperlessBilling'
]

# Filter out columns that don't exist in combined_df
irrelevant_cols = [col for col in irrelevant_cols if col in combined_df.columns]
combined_df.drop(columns=irrelevant_cols, inplace=True)

print(f"Shape of combined_df after dropping irrelevant columns: {combined_df.shape}")

print(f"Shape of combined_df before one-hot encoding: {combined_df.shape}")

# One-hot encode categorical variables
# Identify categorical columns excluding 'churn'
categorical_cols = combined_df.select_dtypes(include=['object']).columns.tolist()
combined_df = pd.get_dummies(combined_df, columns=categorical_cols, drop_first=True)

# Separate features (X) and target (y)
X = combined_df.drop('churn', axis=1)
y = combined_df['churn']

print(f"Shape of X before SMOTE: {X.shape}")
print(f"Shape of y before SMOTE: {y.shape}")

# Step 3: Handle Class Imbalance with SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X, y)

print(f"Churn ratio after SMOTE: {y_res.mean():.2%}")

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)

# Step 4: Train and Evaluate Multiple Models

# Calculate scale_pos_weight for imbalanced datasets
scale_pos_weight_value = len(y_res[y_res == 0]) / len(y_res[y_res == 1])

models = {
    'Logistic Regression': LogisticRegression(random_state=42, solver='liblinear'),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'XGBoost': XGBClassifier(
        scale_pos_weight=scale_pos_weight_value,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    ),
    'LightGBM': LGBMClassifier(random_state=42, scale_pos_weight=scale_pos_weight_value),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

param_grids = {
    'Random Forest': {
        'n_estimators': [100, 150],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    'XGBoost': {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1],
        'n_estimators': [100, 200],
        'subsample': [0.8, 0.9]
    },
    'LightGBM': {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1],
        'num_leaves': [31, 40],
        'max_depth': [-1, 10]
    }
}

results = []
best_model = None
best_f1 = -1
best_accuracy = -1

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    if name in param_grids:
        print(f"Performing GridSearchCV for {name}...")
        grid_search = GridSearchCV(model, param_grids[name], cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        best_model_for_eval = grid_search.best_estimator_
        print(f"Best parameters for {name}: {grid_search.best_params_}")
    else:
        model.fit(X_train, y_train)
        best_model_for_eval = model

    y_pred = best_model_for_eval.predict(X_test)
    y_pred_proba = best_model_for_eval.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    f1_churned = report['1']['f1-score']
    precision_churned = report['1']['precision']
    recall_churned = report['1']['recall']
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision (Churn): {precision_churned:.4f}")
    print(f"Recall (Churn): {recall_churned:.4f}")
    print(f"F1 Score (Churn): {f1_churned:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")
    print("Confusion Matrix:\n", cm)

    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision (Churn)': precision_churned,
        'Recall (Churn)': recall_churned,
        'F1 Score (Churn)': f1_churned,
        'ROC-AUC Score': roc_auc
    })

    # Check for best model
    if f1_churned > best_f1 and accuracy > 0.85: # Target > 85% accuracy
        best_f1 = f1_churned
        best_accuracy = accuracy
        best_model = best_model_for_eval

# Print comparison table
results_df = pd.DataFrame(results)
print("\n--- Model Comparison ---")
print(results_df.to_markdown(index=False))

# Save the best model
if best_model:
    model_filename = 'churnaizer_model_best.pkl'
    joblib.dump(best_model, model_filename)
    print(f"\nBest model saved as {model_filename}")
    print(f"Best Model: {best_model.__class__.__name__} with F1 Score (Churn): {best_f1:.4f} and Accuracy: {best_accuracy:.4f}")
else:
    print("\nNo model met the criteria for best model (F1 > previous best and Accuracy > 85%).")

# Step 6: Save Final Model
joblib.dump(model, 'churnaizer_model_v4.pkl')
print("Model saved as churnaizer_model_v4.pkl")

# Feature Importance
feature_importances = model.feature_importances_
feature_names = X.columns

# Create a DataFrame for feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Calculate percentage importance
total_importance = importance_df['Importance'].sum()
importance_df['Percentage'] = (importance_df['Importance'] / total_importance) * 100

print("\nTop 10 Feature Importances:")
print(importance_df.head(10).to_string(index=False))