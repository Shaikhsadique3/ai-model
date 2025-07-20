import logging
import os
import logging
import pickle
import pandas as pd
from flask import Flask, request, jsonify
from src.train import ChurnPredictor

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

if __name__ == "__main__":
    # Train the model if it hasn't been trained or if you want to retrain on startup
    # This part can be removed if you only want to serve the API and train separately
    try:
        logging.info("Starting Churnaizer application...")
        predictor = ChurnPredictor()
        predictor.run() # This will train and save the model/preprocessor
        logging.info("Churnaizer application finished successfully.")
    except Exception as e:
        logging.critical(f"An unhandled error occurred during Churnaizer training: {e}")

    # Load the trained model and preprocessor for the API
    load_model_and_preprocessor()

    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)