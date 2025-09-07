from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, EmailStr
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
import os

app = FastAPI()

class UserData(BaseModel):
    user_id: str
    plan: str
    usage_score: float
    support_tickets: int
    email: EmailStr
    # plus any other metrics needed by your model

# Dummy predict_churn function - to be replaced with real model loading
def predict_churn(user: UserData) -> dict:
    # Load API keys or model paths from environment variables
    ai_model_path = os.getenv("AI_MODEL_PATH")
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    resend_api_key = os.getenv("RESEND_API_KEY")

    # In a real scenario, you would load your model here using ai_model_path
    # and use it to predict churn based on user data.
    # For now, we return dummy data.
    print(f"AI_MODEL_PATH: {ai_model_path}")
    print(f"OPENROUTER_API_KEY: {openrouter_api_key}")
    print(f"RESEND_API_KEY: {resend_api_key}")

    # Dummy prediction logic
    churn_probability = 0.15 # Example value
    risk_level = "low"
    message = "User is unlikely to churn based on current data."
    trigger_email = False
    recommended_email_tone = "neutral"

    if user.usage_score < 0.3 or user.support_tickets > 5:
        churn_probability = 0.85
        risk_level = "high"
        message = "User is at high risk of churning due to low usage or high support tickets."
        trigger_email = True
        recommended_email_tone = "urgent"
    elif user.usage_score < 0.6 or user.support_tickets > 2:
        churn_probability = 0.45
        risk_level = "medium"
        message = "User shows some signs of potential churn."
        trigger_email = True
        recommended_email_tone = "concerned"

    return {
        "user_id": user.user_id,
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "message": message,
        "trigger_email": trigger_email,
        "recommended_email_tone": recommended_email_tone
    }

@app.get("/health")
async def health_check():
    return {"status": "ok"}

@app.post("/api/v1/predict")
async def predict(user_data: UserData):
    try:
        prediction_result = predict_churn(user_data)
        return prediction_result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    return {"status": "ok"}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "content_type": file.content_type}