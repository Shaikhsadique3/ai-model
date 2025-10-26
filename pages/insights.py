import streamlit as st
import pandas as pd
import joblib
import logging
import plotly.express as px
import shap
import numpy as np

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

st.title("Insights")
st.write("Explore key metrics and visualizations related to customer churn.")

# Load models
@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)

try:
    xgb_model = load_model('model/churnaizer_model.pkl')
    rf_model = load_model('model/churnaizer_saas_model.pkl')
except FileNotFoundError:
    st.error("Error: Model files (xgb_model.joblib or rf_model.joblib) not found. Please ensure they are in the root directory.")
    st.stop()

# Check if processed data is available in session state
if 'processed_df' not in st.session_state:
    st.info("Please go to the 'Predict' page, upload your data, and make predictions first.")
    st.stop()

# Initialize original_df if not already in session state
if 'original_df' not in st.session_state:
    st.session_state['original_df'] = pd.DataFrame() # Initialize as empty DataFrame or with appropriate default

df = st.session_state['processed_df']

# Overall KPIs
st.subheader("Overall Key Performance Indicators (KPIs)")

# Calculate KPIs
total_users = len(df)
active_users = total_users # Assuming all users in the uploaded data are active for this context

churn_percentage = df['churn_prediction_xgb'].mean() * 100 if 'churn_prediction_xgb' in df.columns else 0
predicted_churn_percentage = df['churn_prediction_rf'].mean() * 100 if 'churn_prediction_rf' in df.columns else 0

# Revenue at Risk: Sum of monthly_revenue for predicted churners
revenue_at_risk = df[df['churn_prediction_xgb'] == 1]['monthly_revenue'].sum() if 'churn_prediction_xgb' in df.columns and 'monthly_revenue' in df.columns else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Active Users", f"{active_users}")
col2.metric("Churn % (XGBoost)", f"{churn_percentage:.2f}%")
col3.metric("Predicted Churn % (Random Forest)", f"{predicted_churn_percentage:.2f}%")
col4.metric("Revenue at Risk", f"${revenue_at_risk:,.2f}")

# Placeholder for Breakdowns
st.subheader("Churn Breakdowns")

# Churn probability by plan type
st.write("### Churn Probability by Plan Type")
if 'plan_type' in df.columns and 'churn_prediction_xgb' in df.columns:
    churn_by_plan = df.groupby('plan_type')['churn_prediction_xgb'].mean().reset_index()
    churn_by_plan['churn_probability'] = churn_by_plan['churn_prediction_xgb'] * 100
    fig = px.bar(churn_by_plan, x='plan_type', y='churn_probability', title='Churn Probability by Plan Type')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("'plan_type' or 'churn_prediction_xgb' column not found in data for this breakdown.")

# Feature adoption vs churn risk
st.write("### Feature Adoption vs Churn Risk")
if 'active_features_used' in df.columns and 'churn_prediction_xgb' in df.columns:
    churn_by_features = df.groupby('active_features_used')['churn_prediction_xgb'].mean().reset_index()
    churn_by_features['churn_probability'] = churn_by_features['churn_prediction_xgb'] * 100
    fig = px.bar(churn_by_features, x='active_features_used', y='churn_probability', title='Churn Probability by Active Features Used')
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("'active_features_used' or 'churn_prediction_xgb' column not found in data for this breakdown.")

# Top 10 churn drivers (SHAP output)
st.write("### Top 10 Churn Drivers (SHAP Output)")
if xgb_model and 'original_df' in st.session_state:
    # Assuming 'original_df' in session_state is the preprocessed dataframe used for prediction
    shap_df = st.session_state['original_df'].copy()
    # Ensure categorical features are one-hot encoded for SHAP, matching model training
    if 'plan_type' in shap_df.columns:
        shap_df = pd.get_dummies(shap_df, columns=['plan_type'], drop_first=True)
    if 'payment_status' in shap_df.columns:
        shap_df = pd.get_dummies(shap_df, columns=['payment_status'], drop_first=True)
    # Convert signup_date to numerical (Unix timestamp) if it exists for SHAP
    if 'signup_date' in shap_df.columns:
        shap_df['signup_date'] = pd.to_datetime(shap_df['signup_date']).astype(int) / 10**9

    # Define the exact feature columns and their order as used in model training
    feature_columns = xgb_model.feature_names_in_

    # Ensure all feature columns exist in shap_df, adding missing ones with 0
    for col in feature_columns:
        if col not in shap_df.columns:
            shap_df[col] = 0

    # Select and reorder features for SHAP explanation
    X = shap_df[feature_columns]
    logger.info(f"Data types of X before SHAP: {X.dtypes}") # Debug statement
    
    # Create a SHAP explainer
    explainer = shap.TreeExplainer(xgb_model)
    try:
        shap_values = explainer.shap_values(X)
    except shap.utils._exceptions.ExplainerError as e:
        st.warning(f"SHAP Additivity Check Failed: {e}. Displaying model's default feature importances instead.")
        logger.warning(f"SHAP Additivity Check Failed: {e}. Displaying model's default feature importances instead.")
        # Fallback to model's feature importances if SHAP fails
        if hasattr(xgb_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({'feature': X.columns, 'importance': xgb_model.feature_importances_})
            feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(10)

            fig = px.bar(feature_importance, x='feature', y='importance', title='Top 10 Churn Drivers (Model Feature Importance)')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("SHAP values could not be generated, and model feature importances are not available.")
        shap_values = None # Ensure shap_values is None if an error occurs
    if shap_values is not None:
        # Summarize the SHAP values to get feature importance
        # For multi-class, shap_values is a list of arrays, so we take the mean absolute SHAP value across all classes
        shap_sum = np.abs(shap_values).mean(axis=0)

        feature_importance = pd.DataFrame({'feature': X.columns, 'importance': shap_sum})
        feature_importance = feature_importance.sort_values(by='importance', ascending=False).head(10)

        fig = px.bar(feature_importance, x='feature', y='importance', title='Top 10 Churn Drivers (SHAP)')
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("SHAP values could not be generated. Please check the logs for more details.")
else:
    st.info("XGBoost model or original data not available for SHAP analysis. Please ensure predictions are made.")

# Cohort chart: Retention trend by signup month
st.write("### Cohort Chart: Retention Trend by Signup Month")
if 'signup_date' in st.session_state['original_df'].columns and 'churn_prediction_xgb' in df.columns:
    cohort_df = st.session_state['original_df'].copy()
    cohort_df['signup_date'] = pd.to_datetime(cohort_df['signup_date'])
    cohort_df['signup_month'] = cohort_df['signup_date'].dt.to_period('M')
    cohort_df['churned'] = df['churn_prediction_xgb']

    # Calculate monthly retention (inverse of churn)
    monthly_churn = cohort_df.groupby('signup_month')['churned'].mean().reset_index()
    monthly_churn['retention'] = (1 - monthly_churn['churned']) * 100
    monthly_churn['signup_month'] = monthly_churn['signup_month'].astype(str)

    fig = px.line(monthly_churn, x='signup_month', y='retention', title='Retention Trend by Signup Month')
    st.plotly_chart(fig, use_container_width=True)
else:
    missing_cols = []
    if 'signup_date' not in st.session_state['original_df'].columns:
        missing_cols.append('signup_date')
    if 'churn_prediction_xgb' not in df.columns:
        missing_cols.append('churn_prediction_xgb')
    st.info(f"Cohort analysis requires the following columns: {', '.join(missing_cols)}. Please ensure they are present in your uploaded data and predictions have been made.")