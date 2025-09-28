import pandas as pd
import joblib
import shap
import numpy as np

def generate_recommendations(pipeline_path, raw_data_path, customer_id=None):
    """
    Generates actionable recommendations based on SHAP values for churn prediction.

    Args:
        pipeline_path (str): Path to the trained pipeline (.joblib file).
        raw_data_path (str): Path to the raw data (.csv file).
        customer_id (int, optional): Specific customer ID to analyze. If None, analyzes a sample.
    """
    # Load the trained pipeline
    pipeline = joblib.load(pipeline_path)
    model = pipeline.named_steps['model'] # Extract the model from the pipeline

    # Load raw data
    df_raw = pd.read_csv(raw_data_path)

    # Separate features and target (assuming 'churn' is the target column)
    # The target column should not be present in the data passed to the preprocessor
    if 'churn' in df_raw.columns:
        X_raw = df_raw.drop(columns=['churn'])
    else:
        X_raw = df_raw.copy()

    # Preprocess the data using the pipeline's preprocessor for SHAP explanation
    # We need the preprocessed data to get feature names for SHAP
    preprocessor = pipeline.named_steps['preprocessor']
    X_processed = preprocessor.transform(X_raw)
    
    # Get feature names after preprocessing
    # This assumes the preprocessor is a ColumnTransformer with named transformers
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols) # For passthrough or other transformers without get_feature_names_out

    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_raw.index)

    # Create a SHAP explainer
    explainer = shap.TreeExplainer(model)

    if customer_id is not None:
        # Analyze a specific customer
        if customer_id not in X_processed_df.index:
            print(f"Customer ID {customer_id} not found in the dataset.")
            return
        customer_data = X_processed_df.loc[[customer_id]]
        shap_values = explainer.shap_values(customer_data)
        
        # Get SHAP values for the positive class (churned)
        if isinstance(shap_values, list):
            shap_values_for_positive_class = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_for_positive_class = shap_values[:, :, 1]
        else:
            shap_values_for_positive_class = shap_values

        # Identify features pushing towards churn for this specific customer
        churn_contributing_features = pd.DataFrame({
            'feature': X_processed_df.columns,
            'shap_value': shap_values_for_positive_class[0]
        }).sort_values(by='shap_value', ascending=False)

        print(f"\nRecommendations for Customer ID {customer_id}:")
        print("Features contributing to churn (positive SHAP values):")
        for index, row in churn_contributing_features[churn_contributing_features['shap_value'] > 0].iterrows():
            print(f"- {row['feature']}: Consider addressing this factor. (SHAP value: {row['shap_value']:.2f})")

    else:
        # Analyze a sample of customers
        sample_data = X_processed_df.sample(n=100, random_state=42)
        sample_data_np = sample_data.values
        shap_values = explainer.shap_values(sample_data_np)

        # Get SHAP values for the positive class (churned)
        if isinstance(shap_values, list):
            shap_values_for_positive_class = shap_values[1]
        elif len(shap_values.shape) == 3:
            shap_values_for_positive_class = shap_values[:, :, 1]
        else:
            shap_values_for_positive_class = shap_values

        # Get feature importances from SHAP values
        feature_importances = pd.DataFrame({
            'feature': X_processed_df.columns,
            'importance': np.abs(shap_values_for_positive_class).mean(axis=0)
        }).sort_values(by='importance', ascending=False)

        print("\nTop features contributing to churn (overall):")
        print(feature_importances)

        print("\nGeneral recommendations based on top features:")
        for index, row in feature_importances.head(5).iterrows(): # Top 5 features
            print(f"- For {row['feature']}: Implement strategies to optimize this area to reduce churn.")

def estimate_roi(recommendations):
    """
    Estimates the Return on Investment (ROI) for a given set of recommendations.

    Args:
        recommendations (list): A list of recommendations.

    Returns:
        float: Estimated ROI.
    """
    # TODO: Implement ROI estimation logic
    print("ROI estimation: This is a placeholder. Actual ROI would depend on specific intervention costs and churn reduction.")
    return 0.0

if __name__ == "__main__":
    pipeline_file_path = r'c:\Users\Sadique\Desktop\ai model\models\churn_prediction_pipeline.joblib' # Updated to pipeline path
    raw_data_file_path = r'c:\Users\Sadique\Desktop\ai model\data\enhanced_saas_churn_data.csv' # Using raw data

    # Example usage: Generate recommendations for a sample of customers
    generate_recommendations(pipeline_file_path, raw_data_file_path)

    # Example usage: Generate recommendations for a specific customer (replace 0 with an actual customer ID from your dataset)
    # For demonstration, let's pick a customer ID from the preprocessed data, e.g., the first one.
    # You might need to adjust this based on your actual data's index.
    # customer_to_analyze = pd.read_csv(preprocessed_file_path).index[0]
    # generate_recommendations(model_file_path, preprocessed_file_path, customer_id=customer_to_analyze)

    # Example usage: Estimate ROI (dummy call)
    # roi = estimate_roi([])
    # print(f"Estimated ROI: {roi}")