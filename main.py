from fastapi import FastAPI, Request, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
from typing import Optional, Dict
import os
import time
from datetime import datetime
import secrets
import json
from pathlib import Path

# API Key configuration
KEYS_FILE = 'api_keys.json'
api_key_header = APIKeyHeader(name='X-API-Key')

# Rate limiting configuration
REQUEST_LIMIT = 100  # requests per hour
request_history: Dict[str, list] = {}

# Load or initialize API keys
def load_api_keys():
    try:
        if Path(KEYS_FILE).exists():
            with open(KEYS_FILE, 'r') as f:
                return json.load(f)
        else:
            # Initialize with a default admin key
            default_key = generate_api_key()
            keys = {'keys': [{'key': default_key, 'created_at': datetime.now().isoformat()}]}
            save_api_keys(keys)
            return keys
    except Exception as e:
        print(f"Error loading API keys: {e}")
        return {'keys': []}

def save_api_keys(keys_data):
    with open(KEYS_FILE, 'w') as f:
        json.dump(keys_data, f, indent=2)

def generate_api_key():
    return secrets.token_urlsafe(32)

# Initialize API keys
api_keys = load_api_keys()

def check_api_key(api_key: str = Depends(api_key_header)):
    if not any(k['key'] == api_key for k in api_keys['keys']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

def check_rate_limit(api_key: str):
    current_time = time.time()
    hour_ago = current_time - 3600
    
    # Clean up old requests
    request_history[api_key] = [t for t in request_history.get(api_key, []) if t > hour_ago]
    
    if len(request_history.get(api_key, [])) >= REQUEST_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Rate limit exceeded. Maximum {REQUEST_LIMIT} requests per hour."
        )
    
    request_history.setdefault(api_key, []).append(current_time)

# FastAPI app creation with CORS and documentation
app = FastAPI(
    title="Churn Prediction API",
    description="API for predicting customer churn with ML model",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/admin", response_class=HTMLResponse)
async def admin_dashboard(request: Request):
    # Get the first API key for display
    current_key = api_keys['keys'][0]['key'] if api_keys['keys'] else 'No API key available'
    
    # Calculate usage statistics
    current_usage = len(request_history.get(current_key, []))
    remaining_requests = REQUEST_LIMIT - current_usage
    usage_percentage = (current_usage / REQUEST_LIMIT) * 100
    
    return templates.TemplateResponse("admin.html", {
        "request": request,
        "api_key": current_key,
        "rate_limit": REQUEST_LIMIT,
        "current_usage": current_usage,
        "remaining_requests": remaining_requests,
        "usage_percentage": round(usage_percentage, 1)
    })

@app.post("/admin/regenerate-key")
async def regenerate_api_key():
    try:
        new_key = generate_api_key()
        api_keys['keys'] = [{
            'key': new_key,
            'created_at': datetime.now().isoformat()
        }]
        save_api_keys(api_keys)
        
        # Clear rate limit history for old keys
        request_history.clear()
        
        return JSONResponse({
            "status": "success",
            "new_key": new_key
        })
    except Exception as e:
        return JSONResponse({
            "status": "error",
            "message": str(e)
        }, status_code=500)

# 2. Model load karo
model = joblib.load('churnaizer_model_v4.pkl')

# 3. Define input data format
class CustomerData(BaseModel):
    days_since_signup: int
    monthly_revenue: int
    number_of_logins_last30days: int
    active_features_used: int
    support_tickets_opened: int
    last_payment_status: str
    subscription_plan: str

# API endpoint for churn prediction
@app.post("/api/v1/predict", 
    response_model=dict,
    summary="Predict customer churn",
    description="Predicts whether a customer is likely to churn based on their usage and behavioral data",
    responses={
        200: {"description": "Successful prediction"},
        401: {"description": "Invalid API key"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
async def predict_churn(
    data: CustomerData,
    api_key: str = Depends(check_api_key)
):
    # Check rate limit
    check_rate_limit(api_key)
    try:
        # Validate input ranges
        if data.days_since_signup < 0:
            return {"status": "error", "message": "Days since signup cannot be negative"}
        if data.monthly_revenue < 0:
            return {"status": "error", "message": "Monthly revenue cannot be negative"}
        if data.number_of_logins_last30days < 0:
            return {"status": "error", "message": "Number of logins cannot be negative"}
        if data.active_features_used < 0:
            return {"status": "error", "message": "Active features used cannot be negative"}
        if data.support_tickets_opened < 0:
            return {"status": "error", "message": "Support tickets opened cannot be negative"}
        
        # Create initial DataFrame
        input_data = pd.DataFrame([data.dict()])
        
        # One-hot encode categorical variables
        input_data = pd.get_dummies(input_data, columns=['last_payment_status', 'subscription_plan'], drop_first=True)
        
        # Ensure all expected columns are present
        expected_columns = model.feature_names_in_
        for col in expected_columns:
            if col not in input_data.columns:
                input_data[col] = 0
        
        # Reorder columns to match training data
        input_data = input_data[expected_columns]
        
        # Make prediction
        prediction = model.predict(input_data)
        churn_risk = "Churn" if prediction[0] == 1 else "Not Churn"
        
        # Calculate time-to-churn estimate based on risk factors
        days_to_churn = None
        risk_score = 0
        risk_factors = []
        retention_factors = []
        
        # Analyze engagement metrics
        if data.number_of_logins_last30days < 3:
            risk_score += 3
            risk_factors.append("Critical: Very low login activity in the last 30 days")
            days_to_churn = 4  # High risk of immediate churn
        elif data.number_of_logins_last30days < 10:
            risk_score += 2
            risk_factors.append("Warning: Below average login activity")
            days_to_churn = 7
        else:
            retention_factors.append(f"Strong engagement with {data.number_of_logins_last30days} logins in last 30 days")

        # Analyze feature usage
        if data.active_features_used < 3:
            risk_score += 2
            risk_factors.append("Limited product usage - only using basic features")
            days_to_churn = min(days_to_churn or 14, 10)
        elif data.active_features_used >= 7:
            retention_factors.append(f"Power user utilizing {data.active_features_used} product features")

        # Analyze payment status
        if data.last_payment_status == "Failed":
            risk_score += 3
            risk_factors.append("Payment failure - immediate attention needed")
            days_to_churn = min(days_to_churn or 14, 7)
        else:
            retention_factors.append("Consistent payment history")

        # Analyze support tickets
        if data.support_tickets_opened > 4:
            risk_score += 2
            risk_factors.append("High support needs - customer may be struggling")
            days_to_churn = min(days_to_churn or 14, 10)
        elif data.support_tickets_opened == 0:
            retention_factors.append("Self-sufficient user with no support tickets")

        # Adjust days_to_churn based on subscription plan
        if data.subscription_plan == "Enterprise":
            days_to_churn = days_to_churn * 1.5 if days_to_churn else None
            retention_factors.append("Enterprise customer with longer commitment")
        
        # Calculate comprehension score (0-100) based on clarity of prediction results
        comprehension_score = 0
        comprehension_breakdown = {}
        
        # Base score - having a clear prediction adds 20 points
        comprehension_score += 20
        comprehension_breakdown["prediction_clarity"] = 20
        
        # Risk score clarity - up to 20 points
        risk_score_points = min(20, risk_score * 5) if risk_score > 0 else 0
        comprehension_score += risk_score_points
        comprehension_breakdown["risk_score_clarity"] = risk_score_points
        
        # Time-to-churn estimate clarity - up to 30 points
        time_estimate_points = 30 if days_to_churn is not None else 0
        comprehension_score += time_estimate_points
        comprehension_breakdown["time_estimate_clarity"] = time_estimate_points
        
        # Risk/retention factors clarity - up to 30 points
        factors_count = len(risk_factors) if churn_risk == "Churn" else len(retention_factors)
        factors_points = min(30, factors_count * 10)
        comprehension_score += factors_points
        comprehension_breakdown["factors_clarity"] = factors_points
        
        # Ensure score is within 0-100 range
        comprehension_score = max(0, min(100, comprehension_score))
        
        response = {
            "status": "success",
            "prediction": churn_risk,
            "risk_score": risk_score,
            "days_to_churn": round(days_to_churn) if days_to_churn else None,
            "comprehension_score": comprehension_score,
            "comprehension_breakdown": comprehension_breakdown
        }

        if churn_risk == "Churn":
            response["risk_factors"] = risk_factors
            response["action_needed"] = "Immediate attention required" if days_to_churn <= 7 else "Monitor closely"
        else:
            response["retention_factors"] = retention_factors
            
        return response
    except ValueError as ve:
        return {"status": "error", "message": f"Invalid input data: {str(ve)}"}
    except Exception as e:
        return {"status": "error", "message": "An unexpected error occurred during prediction. Please try again."}
