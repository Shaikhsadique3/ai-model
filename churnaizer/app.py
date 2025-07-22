# app.py — Churnaizer Model API (Production-Ready)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

# ================================
# ✅ Your prediction logic here
# ================================
def predict_churn(user):
    # Replace this stub with real model inference
    usage = user.usage_score
    tickets = user.support_tickets

    # Fake model logic: higher usage = lower churn
    score = max(0.01, min(0.99, 1 - (usage / 100)))
    reason = "Low engagement" if score > 0.7 else "Healthy"
    insight = "Predicted using usage + support data"
    understanding = round(1 - score, 2)

    return {
        "churn_probability": round(score, 3),
        "reason": reason,
        "message": insight,
        "understanding_score": understanding
    }

# ================================
# ✅ FastAPI Setup
# ================================
app = FastAPI()

# Allow any frontend (can restrict later)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_headers=["*"],
    allow_methods=["*"],
)

# Health check
@app.get("/")
def root():
    return {"status": "Churnaizer API is live"}

# Request schema
class UserData(BaseModel):
    user_id: str
    plan: str
    usage_score: float
    support_tickets: int
    email: str

# Prediction route
@app.post("/api/v1/predict")
def predict(user: UserData):
    result = predict_churn(user)
    return result

# Run app if local
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000)