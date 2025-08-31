import logging
import os
import pandas as pd
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from flask_cors import CORS
import joblib
import traceback
import logging
from werkzeug.utils import secure_filename

from churnaizer.src.predict import ChurnPredictorService

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type", "X-API-Key", "X-SDK-Version"]}}) # Allow X-SDK-Version header

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Initialize ChurnPredictorService ---
churn_service = None

def initialize_churn_service():
    global churn_service
    try:
        churn_service = ChurnPredictorService()
        logging.info("✅ ChurnPredictorService initialized successfully.")
    except Exception as e:
        logging.error("❌ Error initializing ChurnPredictorService: %s", str(e))
        traceback.print_exc()

# --- File Upload Configuration ---
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv'}

# --- Input Schema ---
class UserData(BaseModel):
    user_id: str
    subscription_plan: str
    email: str
    days_since_signup: int
    monthly_revenue: float
    number_of_logins_last30days: int
    active_features_used: int
    support_tickets_opened: int
    last_login_days_ago: int
    email_opens_last30days: int
    billing_issue_count: int
    last_payment_status: str

from flask import send_from_directory

# --- Routes ---
@app.route('/')
def serve_test_page():
    return send_from_directory(os.path.abspath(os.path.join(app.root_path, '..')), 'test_page.html')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    if churn_service is None:
        return jsonify({"error": "Service not initialized"}), 500
    try:
        data = request.json
        logging.info(f"📦 Incoming data: {data}")

        # Validate input
        user_data = UserData(**data)
        
        # Convert Pydantic model to DataFrame for prediction service
        input_df = pd.DataFrame([user_data.dict()])
        
        # Call the predict_batch method for single prediction
        # The predict_batch expects 'user_id' column, so ensure it's present
        if 'user_id' not in input_df.columns:
            input_df['user_id'] = 'single_user_prediction'

        prediction_result_list = churn_service.predict_batch(input_df)
        
        # Extract the single prediction result (it's already a dictionary)
        prediction_result = prediction_result_list[0]

        # Determine email trigger and tone (simplified for now, can be expanded)
        trigger_email = False
        recommended_email_tone = None
        
        if prediction_result['risk_level'] == 'High':
            trigger_email = True
            # Simple logic for email tone based on insights (can be expanded)
            # For now, just a placeholder
            recommended_email_tone = "urgency"

        response = {
            "churn_probability": prediction_result['churn_probability'],
            "churn_score": prediction_result['churn_probability'], # Keeping for SDK compatibility
            "user_id": user_data.user_id,
            "message": f"Predicted churn risk is {prediction_result['risk_level']} risk.",
            "status": "success",
            "trigger_email": trigger_email,
            "shouldTriggerEmail": trigger_email, # For SDK compatibility
            "recommended_email_tone": recommended_email_tone,
            "reason": ", ".join(prediction_result['top_reasons']), # Use top reasons as reason
            "understanding_score": 0.85, # Placeholder
            "risk_level": prediction_result['risk_level'],
            "top_reasons": prediction_result['top_reasons'] # Add top_reasons to response
        }

        # Post Prediction Hook: Call Supabase Edge Function if trigger_email is true
        if trigger_email and user_data.email:
            logging.info(f"📧 Triggering email for user {user_data.user_id} with tone: {recommended_email_tone}")
            import requests
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
            if supabase_url and supabase_anon_key:
                try:
                    headers = {"apikey": supabase_anon_key, "Content-Type": "application/json"}
                    payload = {
                        "user_id": user_data.user_id,
                        "user_email": user_data.email,
                        "insights": user_data.dict(), # Pass all user data as insights
                        "recommended_tone": recommended_email_tone
                    }
                    supabase_response = requests.post(f"{supabase_url}/functions/v1/email/send", json=payload, headers=headers)
                    supabase_response.raise_for_status() # Raise an exception for HTTP errors
                    logging.info(f"✅ Supabase email send function called successfully for user {user_data.user_id}")
                except requests.exceptions.RequestException as req_e:
                    logging.error(f"❌ Error calling Supabase Edge Function: {req_e}")
            else:
                logging.warning("⚠️ Supabase URL or Anon Key not set. Skipping email trigger.")
        elif trigger_email and not user_data.email:
            logging.warning(f"⚠️ Cannot trigger email for user {user_data.user_id}: email address is missing.")

        logging.info(f"✅ Prediction completed for user {user_data.user_id} -> {prediction_result['churn_probability']}")
        return jsonify(response), 200

    except ValidationError as ve:
        logging.warning(f"⚠️ Validation error: {ve}")
        return jsonify({"error": "Invalid input", "details": ve.errors()}), 400

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

@app.route("/api/v1/predict-csv", methods=["POST"])
def predict_csv():
    if churn_service is None:
        return jsonify({"error": "Service not initialized"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"Uploaded file saved to {filepath}")

        try:
            df = pd.read_csv(filepath)
            if 'user_id' not in df.columns:
                return jsonify({"error": "CSV must contain a 'user_id' column."}), 400

            # Perform batch prediction
            predictions_list = churn_service.predict_batch(df)

            # Clean up the uploaded file
            os.remove(filepath)
            logging.info(f"Cleaned up uploaded file {filepath}")

            return jsonify(predictions_list), 200

        except Exception as e:
            logging.error(f"❌ Batch prediction error: {e}")
            traceback.print_exc()
            return jsonify({"error": "Batch prediction failed", "details": str(e)}), 500
    else:
        return jsonify({"error": "Allowed file types are csv"}), 400

# --- Run ---
if __name__ == "__main__":
    initialize_churn_service()
    app.run(debug=False, host="0.0.0.0", port=5000)