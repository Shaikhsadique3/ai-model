import logging
import os
import pandas as pd
from flask import Flask, request, jsonify
from pydantic import BaseModel, ValidationError
from flask_cors import CORS
import joblib
import traceback

# --- Flask App Setup ---
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*", "allow_headers": ["Content-Type", "X-API-Key", "X-SDK-Version"]}}) # Allow X-SDK-Version header

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load Model and Preprocessor ---
model = None
preprocessor = None

def load_model_and_preprocessor():
    global model, preprocessor
    try:
        base_dir = os.path.dirname(__file__)
        model_path = os.path.join(base_dir, "models", "churnaizer_saas_model.pkl")
        preprocessor_path = os.path.join(base_dir, "models", "one_hot_encoder.pkl")

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        logging.info("✅ Model and preprocessor loaded successfully.")
    except Exception as e:
        logging.error("❌ Error loading model or preprocessor: %s", str(e))
        traceback.print_exc()

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

@app.route("/api/v1/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        logging.info(f"📦 Incoming data: {data}")

        # Validate input
        user_data = UserData(**data)
        # Select only the features that the model expects
        # Assuming 'user_id' and 'email' are not features for the model
        model_features = [
            'days_since_signup',
            'monthly_revenue',
            'number_of_logins_last30days',
            'active_features_used',
            'support_tickets_opened',
            'last_login_days_ago',
            'email_opens_last30days',
            'billing_issue_count',
            'last_payment_status',
            'subscription_plan'
        ]
        # Separate numerical and categorical features
        numerical_features = [
            'days_since_signup',
            'monthly_revenue',
            'number_of_logins_last30days',
            'active_features_used',
            'support_tickets_opened',
            'last_login_days_ago',
            'email_opens_last30days',
            'billing_issue_count'
        ]
        categorical_features = [
            'subscription_plan',
            'last_payment_status'
        ]

        # Create DataFrame from user_data, excluding user_id and email
        input_data = user_data.dict()
        del input_data['user_id']
        del input_data['email']
        input_df = pd.DataFrame([input_data])

        # Apply preprocessor to categorical features
        X_categorical = preprocessor.transform(input_df[categorical_features])
        ohe_feature_names = preprocessor.get_feature_names_out(categorical_features)
        X_categorical_df = pd.DataFrame(X_categorical, columns=ohe_feature_names, index=input_df.index)

        # Combine numerical and processed categorical features
        X_processed = pd.concat([input_df[numerical_features], X_categorical_df], axis=1)

        # Preprocess and predict
        # The previous X_processed was incorrect as it was transforming the original input_df again.
        # The correct X_processed is already created by concatenating numerical and one-hot encoded categorical features.
        churn_prob = float(model.predict_proba(X_processed)[0][1])

        # Determine email trigger and tone
        trigger_email = False
        recommended_email_tone = None
        insights = {
            "days_since_signup": user_data.days_since_signup,
            "monthly_revenue": user_data.monthly_revenue,
            "number_of_logins_last30days": user_data.number_of_logins_last30days,
            "active_features_used": user_data.active_features_used,
            "support_tickets_opened": user_data.support_tickets_opened,
            "last_login_days_ago": user_data.last_login_days_ago,
            "email_opens_last30days": user_data.email_opens_last30days,
            "billing_issue_count": user_data.billing_issue_count,
            "subscription_plan": user_data.subscription_plan,
            "last_payment_status": user_data.last_payment_status
        }

        if churn_prob > 0.7:
            trigger_email = True
            # Simple logic for email tone based on insights (can be expanded)
            if insights["support_tickets_opened"] > 2 or insights["billing_issue_count"] > 0:
                recommended_email_tone = "empathetic"
            elif insights["active_features_used"] < 3 and insights["number_of_logins_last30days"] < 5:
                recommended_email_tone = "value-driven"
            else:
                recommended_email_tone = "urgency"

        response = {
            "churn_probability": round(churn_prob, 4),
            "churn_score": round(churn_prob, 4),
            "user_id": user_data.user_id,
            "message": f"Predicted churn risk is {'High risk' if churn_prob > 0.7 else 'Low risk' if churn_prob < 0.3 else 'Medium risk'}.",
            "status": "success",
            "trigger_email": trigger_email,
            "shouldTriggerEmail": trigger_email, # For SDK compatibility
            "recommended_email_tone": recommended_email_tone,
            "reason": "User behavior analysis", # Placeholder
            "understanding_score": 0.85, # Placeholder
            "risk_level": 'High' if churn_prob > 0.7 else 'Low' if churn_prob < 0.3 else 'Medium' # Derived from churn_prob
        }

        # Post Prediction Hook: Call Supabase Edge Function if trigger_email is true
        if trigger_email and user_data.email:
            logging.info(f"📧 Triggering email for user {user_data.user_id} with tone: {recommended_email_tone}")
            # Placeholder for Supabase Edge Function call
            # You would typically use a library like 'requests' or 'httpx' here
            import requests
            supabase_url = os.getenv("SUPABASE_URL")
            supabase_anon_key = os.getenv("SUPABASE_ANON_KEY")
            if supabase_url and supabase_anon_key:
                try:
                    headers = {"apikey": supabase_anon_key, "Content-Type": "application/json"}
                    payload = {
                        "user_id": user_data.user_id,
                        "user_email": user_data.email,
                        "insights": insights,
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


        logging.info(f"✅ Prediction completed for user {user_data.user_id} -> {churn_prob}")
        return jsonify(response), 200

    except ValidationError as ve:
        logging.warning(f"⚠️ Validation error: {ve}")
        return jsonify({"error": "Invalid input", "details": ve.errors()}), 400

    except Exception as e:
        logging.error(f"❌ Prediction error: {e}")
        traceback.print_exc()
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500

# --- Run ---
if __name__ == "__main__":
    load_model_and_preprocessor()
    app.run(debug=False, host="0.0.0.0", port=5000)

# For gunicorn on Render, load model at startup
load_model_and_preprocessor()