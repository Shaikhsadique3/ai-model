import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

def analyze_churn_causes(pipeline_path, raw_data_path, customer_id=None):
    """
    Loads the trained pipeline and raw data, then uses SHAP to explain churn predictions.

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
    if 'churn' in df_raw.columns:
        X_raw = df_raw.drop(columns=['churn'])
    else:
        X_raw = df_raw.copy()

    # Preprocess the data using the pipeline's preprocessor for SHAP explanation
    preprocessor = pipeline.named_steps['preprocessor']
    X_processed = preprocessor.transform(X_raw)
    
    # Get feature names after preprocessing
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if hasattr(transformer, 'get_feature_names_out'):
            feature_names.extend(transformer.get_feature_names_out(cols))
        else:
            feature_names.extend(cols) # For passthrough or other transformers without get_feature_names_out

    X_processed_df = pd.DataFrame(X_processed, columns=feature_names, index=X_raw.index)

    # Create a SHAP explainer (default: model_output="raw")
    explainer = shap.TreeExplainer(model)

    if customer_id is not None:
        # Analyze a specific customer
        if customer_id not in X_processed_df.index:
            print(f"Customer ID {customer_id} not found in the dataset.")
            return
        customer_data = X_processed_df.loc[[customer_id]]
        shap_values = explainer.shap_values(customer_data)
        print(f"\nSHAP values for Customer ID {customer_id}:")
        shap.initjs()
        shap.force_plot(explainer.expected_value[1], shap_values[1], customer_data)
        plt.show()
    else:
        # Analyze a sample of customers
        sample_data = X_processed_df.sample(n=100, random_state=42)
        sample_data_np = sample_data.values
        shap_values = explainer.shap_values(sample_data_np)

        # Print type and shape for debugging
        print(f"Type of shap_values: {type(shap_values)}")
        if isinstance(shap_values, list):
            print(f"Length of shap_values: {len(shap_values)}")
            for i, sv in enumerate(shap_values):
                print(f"Shape of shap_values[{i}]: {sv.shape}")
            # Use the positive class (usually index 1)
            shap_values_for_positive_class = shap_values[1]
        elif len(shap_values.shape) == 3:
            print(f"Shape of shap_values: {shap_values.shape}")
            # Select positive class (last dimension index 1)
            shap_values_for_positive_class = shap_values[:, :, 1]
        else:
            print(f"Shape of shap_values: {shap_values.shape}")
            shap_values_for_positive_class = shap_values

        print("\nSHAP values for a sample of customers:")
        print(f"Shape of shap_values_for_positive_class: {shap_values_for_positive_class.shape}")
        print(f"Shape of sample_data: {sample_data.shape}")

        shap.summary_plot(shap_values_for_positive_class, sample_data)
        plt.show()

if __name__ == "__main__":
    pipeline_file_path = r'c:\Users\Sadique\Desktop\ai model\models\churn_prediction_pipeline.joblib' # Updated to pipeline path
    raw_data_file_path = r'c:\Users\Sadique\Desktop\ai model\data\enhanced_saas_churn_data.csv' # Using raw data

    # Example usage: Analyze a sample of customers
    analyze_churn_causes(pipeline_file_path, raw_data_file_path)

    # Example usage: Analyze a specific customer (replace 0 with an actual customer ID from your dataset)
    # analyze_churn_causes(pipeline_file_path, raw_data_file_path, customer_id=0)