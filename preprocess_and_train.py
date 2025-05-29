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
telco_df = pd.read_csv('telco_churn.csv')
synthetic_df = pd.read_csv('synthetic_churn_data_7000.csv')

# Print initial information
print(f"Telco dataset shape: {telco_df.shape}")
print(f"Synthetic dataset shape: {synthetic_df.shape}")

# Step 2: Preprocess Telco dataset
print("\nPreprocessing Telco dataset...")

# Convert 'Churn' to binary (1/0)
telco_df['churn'] = telco_df['Churn'].map({'Yes': 1, 'No': 0})

# Rename columns for consistency
telco_df = telco_df.rename(columns={
    'tenure': 'days_since_signup',
    'MonthlyCharges': 'monthly_revenue',
    'Contract': 'subscription_plan',
    'PaymentMethod': 'last_payment_status'
})

# Map subscription_plan values
subscription_map = {
    'Month-to-month': 'Basic',
    'One year': 'Pro',
    'Two year': 'Enterprise'
}
telco_df['subscription_plan'] = telco_df['subscription_plan'].map(subscription_map)

# Map payment status
payment_map = {
    'Electronic check': 'Success',
    'Mailed check': 'Success',
    'Bank transfer (automatic)': 'Success',
    'Credit card (automatic)': 'Success'
}
telco_df['last_payment_status'] = telco_df['last_payment_status'].map(payment_map)

# Add missing columns with reasonable defaults based on other features
telco_df['number_of_logins_last30days'] = np.random.randint(1, 30, size=len(telco_df))
telco_df['active_features_used'] = np.random.randint(1, 10, size=len(telco_df))
telco_df['support_tickets_opened'] = np.random.randint(0, 5, size=len(telco_df))

# Select only the required columns
telco_processed = telco_df[[
    'days_since_signup',
    'monthly_revenue',
    'subscription_plan',
    'number_of_logins_last30days',
    'active_features_used',
    'support_tickets_opened',
    'last_payment_status',
    'churn'
]]

# Step 3: Preprocess Synthetic dataset
print("\nPreprocessing Synthetic dataset...")

# Ensure synthetic dataset has the same columns
synthetic_processed = synthetic_df[[
    'days_since_signup',
    'monthly_revenue',
    'subscription_plan',
    'number_of_logins_last30days',
    'active_features_used',
    'support_tickets_opened',
    'last_payment_status',
    'churn'
]]

# Step 4: Merge datasets
print("\nMerging datasets...")
merged_df = pd.concat([telco_processed, synthetic_processed], axis=0)
print(f"Merged dataset shape: {merged_df.shape}")

# Save merged dataset
merged_df.to_csv('merged_churn_data.csv', index=False)
print("Merged dataset saved as: merged_churn_data.csv")

# Step 5: Handle categorical variables
print("\nHandling categorical variables...")
df_encoded = pd.get_dummies(merged_df, columns=['subscription_plan', 'last_payment_status'], drop_first=True)
print(f"Encoded dataset shape: {df_encoded.shape}")

# Step 6: Split features and target
X = df_encoded.drop('churn', axis=1)
y = df_encoded['churn']

# Step 7: Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Step 8: Check class balance
print("\nClass distribution before balancing:")
print(y_train.value_counts())

# Step 9: Apply SMOTE for class balancing
print("\nBalancing dataset with SMOTE...")
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
print("Class distribution after balancing:")
print(pd.Series(y_train_balanced).value_counts())

# Step 10: Train the model
print("\nTraining Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_balanced, y_train_balanced)

# Step 11: Evaluate the model
print("\nEvaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 12: Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance.head(10))

# Step 13: Save the model
print("\nSaving model...")
model_filename = 'churnaizer_model_v2.pkl'
joblib.dump(model, model_filename)
print(f"Model saved as: {model_filename}")

# Step 14: Test prediction function
print("\nTesting prediction function...")

def predict_churn(customer_data):
    # Convert input data to DataFrame
    input_data = pd.DataFrame([customer_data])
    
    # Encode categorical variables
    input_encoded = pd.get_dummies(input_data, columns=['subscription_plan', 'last_payment_status'], drop_first=True)
    
    # Ensure all columns from training are present
    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[X.columns]
    
    # Make prediction
    prediction = model.predict(input_encoded)[0]
    probability = model.predict_proba(input_encoded)[0][1]
    
    return {
        "churn_prediction": bool(prediction),
        "churn_probability": float(probability),
        "message": "High risk of churn" if prediction == 1 else "Low risk of churn"
    }

# Test with a sample customer
sample_customer = {
    'days_since_signup': 30,
    'monthly_revenue': 99,
    'subscription_plan': 'Basic',
    'number_of_logins_last30days': 2,
    'active_features_used': 1,
    'support_tickets_opened': 3,
    'last_payment_status': 'Failed'
}

prediction_result = predict_churn(sample_customer)
print("Sample prediction result:")
print(prediction_result)

print("\nModel training and evaluation complete!")