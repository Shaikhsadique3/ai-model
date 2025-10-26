import streamlit as st
import pandas as pd
import joblib
import logging

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import os

# Define file paths for the models
MODEL_XGB_PATH = "model/churnaizer_model.pkl"
MODEL_RF_PATH = "model/churnaizer_saas_model.pkl"

# Load the models
@st.cache_data
def load_models():
    try:
        model_xgb = joblib.load(MODEL_XGB_PATH)
        model_rf = joblib.load(MODEL_RF_PATH)
        logger.info("Models loaded successfully.")
        return model_xgb, model_rf
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        st.error(f"Error loading models: {e}. Please ensure the models are compatible with the installed joblib and scikit-learn versions. You might need to retrain the models or save them in a compatible format.")
        return None, None

model_xgb, model_rf = load_models()

st.title("Churn Prediction App")
st.write("Upload a CSV file to get churn predictions. The output will include the original data with 'churn_prediction' (0 or 1) and 'churn_probability' columns added.")

uploaded_file = st.file_uploader("Upload CSV for Churn Prediction", type=["csv"])

st.subheader("Or use Sample Data")
use_sample_data = st.checkbox("Use high-risk sample data instead of uploading")

df = None # Initialize df to None

if use_sample_data:
    with open("high_risk_sample_data.csv", "rb") as f:
        st.download_button(
            label="Download High-Risk Sample Data CSV",
            data=f,
            file_name="high_risk_sample_data.csv",
            mime="text/csv",
        )
    df = pd.read_csv("high_risk_sample_data.csv")
    st.write("Using High-Risk Sample Data Preview:")
    st.dataframe(df.head())
    st.session_state['original_df'] = df.copy() # Store original DataFrame
elif uploaded_file is not None and model_xgb is not None and model_rf is not None:
    logger.info("CSV file uploaded.")
    df = pd.read_csv(uploaded_file)
    st.session_state['original_df'] = df.copy() # Store original DataFrame
    st.write("Uploaded CSV Preview:")
    st.dataframe(df.head())

if df is not None:
    categorical_features = ['plan_type', 'payment_status']
    
    # Check if categorical features exist in the DataFrame
    missing_features = [feature for feature in categorical_features if feature not in df.columns]
    if missing_features:
        st.error(f"Error: Missing expected categorical features in the uploaded CSV: {', '.join(missing_features)}. Please ensure your CSV contains these columns for prediction. Alternatively, you can use the 'High-Risk Sample Data' provided below.")
        logger.error(f"Missing categorical features: {', '.join(missing_features)}")
        st.stop() # Stop execution if critical features are missing

    df_encoded = pd.get_dummies(df, columns=categorical_features, drop_first=True)

    # Ensure all expected columns from training are present, add missing ones with 0
    expected_columns = ['monthly_revenue', 'days_since_signup', 'last_login_days_ago',
                        'logins_last30days', 'active_features_used', 'tickets_opened',
                        'NPS_score', 'plan_type_Enterprise', 'plan_type_Free',
                        'plan_type_Pro', 'payment_status_Late', 'payment_status_On-time']

    for col in expected_columns:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
    
    # Reorder columns to match the training order
    df_processed = df_encoded[expected_columns]

    if st.button("Make Predictions"):
        st.write("Making predictions...")
        # Make predictions with XGBoost model
        predictions_xgb = model_xgb.predict(df_processed)
        prediction_proba_xgb = model_xgb.predict_proba(df_processed)[:, 1] # Probability of churn

        # Make predictions with RandomForest model
        predictions_rf = model_rf.predict(df_processed)
        prediction_proba_rf = model_rf.predict_proba(df_processed)[:, 1] # Probability of churn

        # Add predictions to the original DataFrame
        df["churn_prediction_xgb"] = predictions_xgb
        df["churn_probability_xgb"] = prediction_proba_xgb
        df["churn_prediction_rf"] = predictions_rf
        df["churn_probability_rf"] = prediction_proba_rf
        logger.info("Predictions made and displayed.")

        st.write("Prediction Results:")
        st.dataframe(df)

        st.session_state['processed_df'] = df

        # Add download button for the processed data
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Processed Data as CSV",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv',
        )

        st.subheader("Feature Importance (XGBoost Model)")
        # Get feature importances from the XGBoost model
        feature_importances = model_xgb.feature_importances_
        feature_names = df_processed.columns

        # Create a DataFrame for feature importances
        importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)

        st.bar_chart(importance_df.set_index('Feature'))
        logger.info("Feature importances displayed.")