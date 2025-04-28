from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import sqlite3
from typing import Optional

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

@app.post("/predict", dependencies=[Depends(verify_api_key)])
async def predict_churn(customer: CustomerData):
    try:
        # Convert input data to DataFrame
        input_data = pd.DataFrame([{
            'days_since_signup': customer.days_since_signup,
            'monthly_revenue': customer.monthly_revenue,
            'subscription_plan': customer.subscription_plan,
            'number_of_logins_last30days': customer.number_of_logins_last30days,
            'active_features_used': customer.active_features_used,
            'support_tickets_opened': customer.support_tickets_opened,
            'last_payment_status': customer.last_payment_status
        }])

        # Encode categorical variables
        input_encoded = pd.get_dummies(input_data, drop_first=True)
        
        # Make prediction
        prediction = model.predict(input_encoded)[0]
        probability = model.predict_proba(input_encoded)[0][1]

        return {
            "churn_prediction": bool(prediction),
            "churn_probability": float(probability),
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