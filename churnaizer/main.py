import logging
"""Flask application for churn prediction.

This module sets up a Flask API to serve churn predictions. It loads a pre-trained
model and preprocessor, and provides an endpoint for making predictions.
"""

import os
import logging
import pickle
import pandas as pd
from flask import Flask, request, jsonify


# Configure logging
log_file_path = os.path.join(os.path.dirname(__file__), 'logs', 'churnaizer.log')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
    logging.FileHandler(log_file_path),
    logging.StreamHandler()
])

app = Flask(__name__)

model = None
preprocessor = None

def load_model_and_preprocessor():
    """Loads the pre-trained churn prediction model and preprocessor.

    The model and preprocessor are loaded from the 'models' directory.
    Logs an error if the files are not found.
    """
    global model, preprocessor
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'churnaizer_saas_model.pkl')
    preprocessor_path = os.path.join(os.path.dirname(__file__), 'models', 'one_hot_encoder.pkl')
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        with open(preprocessor_path, 'rb') as f:
            preprocessor = pickle.load(f)
        logging.info("Model and preprocessor loaded successfully.")
    except FileNotFoundError as e:
        logging.error(f"Error loading model or preprocessor: {e}. Please ensure they are trained and saved.")
        model = None
        preprocessor = None

@app.route('/predict', methods=['POST'])
def predict():
    """Handles prediction requests.

    Expects a JSON payload containing customer data. Preprocesses the data,
    makes predictions using the loaded model, and returns the predictions
    and probabilities.
    """
    if model is None or preprocessor is None:
        return jsonify({"error": "Model or preprocessor not loaded. Please train the model first."}), 500

    try:
        data = request.get_json(force=True)
        if not isinstance(data, list):
            data = [data] # Ensure data is a list of dictionaries
        
        df = pd.DataFrame(data)
        
        # Preprocess the data using the loaded preprocessor
        # Assuming the preprocessor expects the same columns as during training
        # and handles categorical features correctly.
        # You might need to adjust this part based on how your preprocessor works.
        processed_data = preprocessor.transform(df)
        
        predictions = model.predict(processed_data)
        probabilities = model.predict_proba(processed_data)[:, 1] # Probability of churn
        
        results = []
        for i in range(len(predictions)):
            results.append({"prediction": int(predictions[i]), "probability": float(probabilities[i])})

        return jsonify(results)

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400

# Load the trained model and preprocessor for the API
load_model_and_preprocessor()