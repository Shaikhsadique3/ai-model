from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import sqlite3
from typing import Optional, Dict, List
import math

# Initialize FastAPI app
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn using machine learning",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained model
model = joblib.load('churnaizer_model.pkl')

# Get feature importance from the model
feature_importances = None
if hasattr(model, 'feature_importances_'):
    feature_importances = model.feature_importances_

# API Key header
API_KEY_HEADER = APIKeyHeader(name="X-API-Key")

# Input data model
class CustomerData(BaseModel):
    days_since_signup: int
    monthly_revenue: float
    subscription_plan: str
    number_of_logins_last30days: int
    active_features_used: int
    support_tickets_opened: int
    last_payment_status: str

# Verify API key
async def verify_api_key(api_key: str = Depends(API_KEY_HEADER)):
    conn = sqlite3.connect('api_keys.db')
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM api_keys WHERE key = ?', (api_key,))
    result = cursor.fetchone()
    conn.close()
    
    if not result:
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )
    return api_key

def estimate_time_to_churn(probability: float, customer_data: dict) -> int:
    """Estimate time to churn based on probability and customer data."""
    # Base time estimation (in days)
    if probability < 0.3:
        base_time = 180  # Low risk: ~6 months
    elif probability < 0.6:
        base_time = 90   # Medium risk: ~3 months
    else:
        base_time = 30   # High risk: ~1 month
    
    # Adjust based on customer engagement
    login_factor = 1.0
    if customer_data['number_of_logins_last30days'] < 5:
        login_factor = 0.7  # Accelerate churn for inactive users
    elif customer_data['number_of_logins_last30days'] > 20:
        login_factor = 1.3  # Extend time for active users
    
    # Adjust based on payment status
    payment_factor = 1.0
    if customer_data['last_payment_status'] == 'Failed':
        payment_factor = 0.5  # Significantly accelerate churn for payment issues
    
    # Calculate final estimate
    estimated_days = math.ceil(base_time * login_factor * payment_factor)
    
    return estimated_days

def identify_churn_reasons(customer_data: dict, encoded_data: pd.DataFrame) -> List[Dict]:
    """Identify top reasons for potential churn based on feature importance."""
    reasons = []
    
    # Check if we have feature importances from the model
    if feature_importances is not None and len(feature_importances) == len(encoded_data.columns):
        # Get feature values and their importance
        feature_values = {}
        for i, col in enumerate(encoded_data.columns):
            feature_values[col] = {
                'value': encoded_data.iloc[0, i],
                'importance': feature_importances[i]
            }
        
        # Add specific reasons based on customer data
        if customer_data['number_of_logins_last30days'] < 5:
            reasons.append({
                'factor': 'Low engagement',
                'description': 'Customer logged in less than 5 times in the last 30 days',
                'impact': 'high' if feature_values.get('number_of_logins_last30days', {}).get('importance', 0) > 0.1 else 'medium'
            })
        
        if customer_data['active_features_used'] < 3:
            reasons.append({
                'factor': 'Limited feature usage',
                'description': f'Customer only using {customer_data["active_features_used"]} features',
                'impact': 'medium'
            })
        
        if customer_data['support_tickets_opened'] > 3:
            reasons.append({
                'factor': 'High support needs',
                'description': f'Customer opened {customer_data["support_tickets_opened"]} support tickets',
                'impact': 'high'
            })
            
        if customer_data['last_payment_status'] == 'Failed':
            reasons.append({
                'factor': 'Payment issues',
                'description': 'Last payment attempt failed',
                'impact': 'critical'
            })
    
    # If no specific reasons found, provide generic ones based on probability
    if not reasons:
        reasons.append({
            'factor': 'Multiple factors',
            'description': 'A combination of engagement, usage, and account factors',
            'impact': 'medium'
        })
    
    return reasons

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_churn(customer: CustomerData):
    try:
        # Convert input data to DataFrame
        customer_dict = {
            'days_since_signup': customer.days_since_signup,
            'monthly_revenue': customer.monthly_revenue,
            'subscription_plan': customer.subscription_plan,
            'number_of_logins_last30days': customer.number_of_logins_last30days,
            'active_features_used': customer.active_features_used,
            'support_tickets_opened': customer.support_tickets_opened,
            'last_payment_status': customer.last_payment_status
        }
        
        input_data = pd.DataFrame([customer_dict])

        # Encode categorical variables
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]
        
        # Estimate time to churn
        time_to_churn = estimate_time_to_churn(probability, customer_dict)
        
        # Identify reasons for churn
        churn_reasons = identify_churn_reasons(customer_dict, input_encoded)

        return {
            "churn_prediction": bool(prediction),
            "churn_probability": float(probability),
            "estimated_days_to_churn": time_to_churn,
            "churn_reasons": churn_reasons,
            "message": "High risk of churn" if prediction == 1 else "Low risk of churn"
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "loaded"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)