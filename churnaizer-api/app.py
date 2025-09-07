from fastapi import FastAPI
import uvicorn
import os

app = FastAPI()

from pydantic import BaseModel

class UserData(BaseModel):
    user_id: str
    days_since_signup: float
    monthly_revenue: float
    number_of_logins_last30days: float
    active_features_used: float
    support_tickets_opened: float
    last_login_days_ago: float
    email_opens_last30days: float
    billing_issue_count: float
    subscription_plan: str
    last_payment_status: str
    email: str

@app.get("/")
async def health_check():
    return {"status": "Churn API running"}

@app.post("/predict")
async def predict(user_data: UserData):
    # Dummy prediction logic
    churn_probability = 0.15
    risk_level = "low"
    top_reasons = ["dummy_reason_1", "dummy_reason_2"]

    return {
        "user_id": user_data.user_id,
        "churn_probability": churn_probability,
        "risk_level": risk_level,
        "top_reasons": top_reasons
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)