import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib

# Step 1: Load datasets
print("Loading datasets...")

# Load the datasets
telco_df = pd.read_csv('telco_churn.csv')
synthetic_df = pd.read_csv('synthetic_churn_data_7000.csv')

# Data Cleaning: Remove irrelevant features
# Remove 'customerID' as it's an identifier and not a feature
telco_df = telco_df.drop('customerID', axis=1)
synthetic_df = synthetic_df.drop('customerID', axis=1)

# Remove 'SeniorCitizen', 'subscription_plan_Unknown', and telecom-specific fields not relevant to SaaS
# First, identify telecom-specific columns that are not relevant to a general SaaS context.
# Based on common telecom datasets, these might include 'PhoneService', 'MultipleLines', 'OnlineSecurity',
# 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
# 'PaperlessBilling', 'PaymentMethod'.
# 'InternetService' was previously dropped due to XGBoost compatibility, but it's also telecom-specific.
# 'TotalCharges' and 'MonthlyCharges' are general and can be kept.

# Columns to drop based on the prompt and previous issues
columns_to_drop = ['SeniorCitizen', 'PhoneService', 'MultipleLines', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                   'Contract', 'PaperlessBilling', 'PaymentMethod']

# Check if columns exist before dropping to avoid errors
for col in columns_to_drop:
    if col in telco_df.columns:
        telco_df = telco_df.drop(col, axis=1)
    if col in synthetic_df.columns:
        synthetic_df = synthetic_df.drop(col, axis=1)

# Handle 'subscription_plan_Unknown' if it exists after one-hot encoding, which will happen later.
# For now, we focus on initial raw data cleaning.

# Drop duplicate rows
telco_df.drop_duplicates(inplace=True)
synthetic_df.drop_duplicates(inplace=True)

# Drop rows with any null values
telco_df.dropna(inplace=True)
synthetic_df.dropna(inplace=True)

# Drop columns with >40% missing values if any
def drop_sparse_columns(df, threshold=0.4):
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().sum() / len(df)
    # Identify columns to drop
    columns_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
    if columns_to_drop:
        print(f"Dropping columns with >{threshold*100}% missing values: {columns_to_drop}")
        df = df.drop(columns=columns_to_drop)
    else:
        print(f"No columns with >{threshold*100}% missing values found.")
    return df

telco_df = drop_sparse_columns(telco_df)
synthetic_df = drop_sparse_columns(synthetic_df)

# Rename columns for consistency
telco_df = telco_df.rename(columns={'customerID': 'customer_id', 'MonthlyCharges': 'monthly_revenue', 'TotalCharges': 'total_revenue', 'tenure': 'days_since_signup'})
synthetic_df = synthetic_df.rename(columns={'CustomerID': 'customer_id', 'MonthlyRevenue': 'monthly_revenue', 'DaysSinceSignup': 'days_since_signup'})

# Convert 'TotalCharges' to numeric, coercing errors to NaN
telco_df['total_revenue'] = pd.to_numeric(telco_df['total_revenue'], errors='coerce')
synthetic_df['total_revenue'] = pd.to_numeric(synthetic_df['total_revenue'], errors='coerce')

# Feature Engineering
# days_inactive = 30 - number_of_logins_last30days
# Assuming 'number_of_logins_last30days' is a new feature that needs to be created or simulated.
# For now, let's create a placeholder or a simple simulation if not available in the dataset.
# In a real scenario, this would come from actual user activity data.

# Add a placeholder for number_of_logins_last30days if it doesn't exist
# For demonstration, let's assume a random distribution for 'number_of_logins_last30days' for now.
if 'number_of_logins_last30days' not in telco_df.columns:
    telco_df['number_of_logins_last30days'] = np.random.randint(0, 31, size=len(telco_df))
if 'number_of_logins_last30days' not in synthetic_df.columns:
    synthetic_df['number_of_logins_last30days'] = np.random.randint(0, 31, size=len(synthetic_df))

telco_df['days_inactive'] = 30 - telco_df['number_of_logins_last30days']
synthetic_df['days_inactive'] = 30 - synthetic_df['number_of_logins_last30days']

# engagement_score = active_features_used / max_possible_features
# Assume max_possible_features = 10
# Add a placeholder for active_features_used if it doesn't exist
if 'active_features_used' not in telco_df.columns:
    telco_df['active_features_used'] = np.random.randint(0, 11, size=len(telco_df))
if 'active_features_used' not in synthetic_df.columns:
    synthetic_df['active_features_used'] = np.random.randint(0, 11, size=len(synthetic_df))

MAX_POSSIBLE_FEATURES = 10
telco_df['engagement_score'] = telco_df['active_features_used'] / MAX_POSSIBLE_FEATURES
synthetic_df['engagement_score'] = synthetic_df['active_features_used'] / MAX_POSSIBLE_FEATURES

# support_tickets_ratio = support_tickets_opened / (days_since_signup + 1)
# Add a placeholder for support_tickets_opened if it doesn't exist
if 'support_tickets_opened' not in telco_df.columns:
    telco_df['support_tickets_opened'] = np.random.randint(0, 10, size=len(telco_df))
if 'support_tickets_opened' not in synthetic_df.columns:
    synthetic_df['support_tickets_opened'] = np.random.randint(0, 10, size=len(synthetic_df))

telco_df['support_tickets_ratio'] = telco_df['support_tickets_opened'] / (telco_df['days_since_signup'] + 1)
synthetic_df['support_tickets_ratio'] = synthetic_df['support_tickets_opened'] / (synthetic_df['days_since_signup'] + 1)

# Drop unnecessary columns that are not relevant to SaaS or have been replaced by engineered features
# Ensure 'total_revenue' is not dropped here as it's a valid feature.
# 'InternetService' was dropped earlier due to XGBoost compatibility, and is also telecom-specific.

# Columns to drop from telco_df that are not relevant to SaaS or are redundant
telco_cols_to_drop_final = ['gender', 'Partner', 'Dependents']
# Add the original 'number_of_logins_last30days', 'active_features_used', 'support_tickets_opened' if they were placeholders
# and we want to keep only the engineered features.
telco_cols_to_drop_final.extend(['number_of_logins_last30days', 'active_features_used', 'support_tickets_opened'])

# Columns to drop from synthetic_df that are not relevant to SaaS or are redundant
synthetic_cols_to_drop_final = []
synthetic_cols_to_drop_final.extend(['number_of_logins_last30days', 'active_features_used', 'support_tickets_opened'])

# Drop columns from telco_df
for col in telco_cols_to_drop_final:
    if col in telco_df.columns:
        telco_df = telco_df.drop(columns=[col], errors='ignore')

# Drop columns from synthetic_df
for col in synthetic_cols_to_drop_final:
    if col in synthetic_df.columns:
        synthetic_df = synthetic_df.drop(columns=[col], errors='ignore')

# Ensure 'churn' column is consistent (True/False)
telco_df['churn'] = telco_df['Churn'].apply(lambda x: True if x == 'Yes' else False)
synthetic_df['churn'] = synthetic_df['churn'].astype(bool)
telco_df = telco_df.drop(columns=['Churn'], errors='ignore')

# Calculate and log the churn rate (% of churned users)
initial_churn_rate_telco = telco_df['churn'].mean() * 100
initial_churn_rate_synthetic = synthetic_df['churn'].mean() * 100
print(f"Initial Churn Rate (Telco): {initial_churn_rate_telco:.2f}%")
print(f"Initial Churn Rate (Synthetic): {initial_churn_rate_synthetic:.2f}%")
# synthetic_df = synthetic_df.drop(columns=['churn'], errors='ignore') # This line should be removed as 'churn' is the target

# Add missing columns to synthetic_df if they exist in telco_df and vice-versa
# For simplicity, let's assume the relevant columns are already aligned or handled by one-hot encoding

# Merge datasets
merged_df = pd.concat([telco_df, synthetic_df], ignore_index=True)

# Drop customer_id as it's not a feature for prediction
merged_df = merged_df.drop(columns=['customer_id'], errors='ignore')

# Handle missing values (if any) - fill with 0 or mean/median depending on the column
# For simplicity, let's fill numerical NaNs with 0 and categorical NaNs with 'Unknown'
for col in merged_df.columns:
    if merged_df[col].dtype == 'object':
        merged_df[col] = merged_df[col].fillna('Unknown')
    else:
        merged_df[col] = merged_df[col].fillna(0)

# One-hot encode categorical variables
merged_df = pd.get_dummies(merged_df, columns=['subscription_plan', 'last_payment_status'], drop_first=True)

# Removed print("Merged DataFrame dtypes before splitting X and y:")
# Removed print(merged_df.dtypes)

# Separate features (X) and target (y)
X = merged_df.drop('churn', axis=1)
y = merged_df['churn']

# Removed print(f"Unique values in y: {y.unique()}")
# Removed print(f"Dtype of y: {y.dtype}")

# Use train_test_split with stratification
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Apply SMOTE for class balancing
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Train the model using XGBoost
from xgboost import XGBClassifier

# Calculate scale_pos_weight for class imbalance handling
# Ensure 'churn' is boolean (True/False) for this calculation
churned_count = y_train_balanced.sum() # Count of True (churned users)
not_churned_count = len(y_train_balanced) - churned_count # Count of False (not churned users)

# Avoid division by zero if there are no churned users (highly unlikely after SMOTE)
scale_pos_weight_value = not_churned_count / churned_count if churned_count > 0 else 1

model = XGBClassifier(
    scale_pos_weight = scale_pos_weight_value,
    use_label_encoder=False,
    eval_metric='logloss',
    max_depth=6,
    learning_rate=0.05,
    n_estimators=300,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
model.fit(X_train_balanced, y_train_balanced) # Train on balanced data

# Evaluate on the test set only
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report, roc_auc_score

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", class_report)
print("\nConfusion Matrix:\n", conf_matrix)
print(f"ROC-AUC Score: {roc_auc:.4f}")

# Print top 10 feature importances with their importance %
feature_importances = pd.Series(model.feature_importances_, index=X.columns)
top_10_features = feature_importances.nlargest(10)
print("\nTop 10 Feature Importances (with percentage):\n")
total_importance = feature_importances.sum()
for feature, importance in top_10_features.items():
    print(f"{feature}: {importance:.4f} ({importance/total_importance:.2%})")

# Save the trained model as churnaizer_model_v4.pkl
import joblib
joblib.dump(model, 'churnaizer_model_v4.pkl')
print("\nModel saved as churnaizer_model_v4.pkl")

print("\nPerformance for churned users (True class):")
# Calculate metrics for the 'True' class specifically
precision_churn, recall_churn, f1_churn, _ = precision_recall_fscore_support(y_test, y_pred, labels=[True], average=None)
print(f"Precision (Churned): {precision_churn[0]:.4f}")
print(f"Recall (Churned): {recall_churn[0]:.4f}")
print(f"F1-score (Churned): {f1_churn[0]:.4f}")

# Sample prediction function (for demonstration, not part of the main training)
def predict_churn(customer_data: dict):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([customer_data])
    
    # One-hot encode categorical variables, ensuring consistency with training data
    input_encoded = pd.get_dummies(input_df, columns=['subscription_plan', 'last_payment_status'], drop_first=True)
    
    # Align columns - add missing columns with 0 and reorder
    missing_cols = set(X.columns) - set(input_encoded.columns)
    for c in missing_cols:
        input_encoded[c] = 0
    input_encoded = input_encoded[X.columns]
    
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    return {
        'churn_prediction': bool(prediction),
        'churn_probability': float(probability),
        'message': 'High risk of churn' if prediction == 1 else 'Low risk of churn'
    }

# Example usage of the sample prediction function
# sample_customer = {
#     'days_since_signup': 300,
#     'monthly_revenue': 50.0,
#     'subscription_plan': 'Premium',
#     'number_of_logins_last30days': 10,
#     'active_features_used': 5,
#     'support_tickets_opened': 1,
#     'last_payment_status': 'Paid'
# }
# prediction_result = predict_churn(sample_customer)
# print("\nSample Prediction:", prediction_result)