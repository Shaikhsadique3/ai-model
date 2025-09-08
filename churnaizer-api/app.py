from fastapi import FastAPI
import uvicorn
import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel

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

@app.post("/upload-csv/")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    try:
        df = pd.read_csv(file.file)

        # 2. Data Validation
        required_columns = ["user_id", "plan_name", "last_login_date", "billing_status"]
        for col in required_columns:
            if col not in df.columns:
                raise HTTPException(status_code=400, detail=f"Missing required column: {col}")

        # Validate data types and values
        # user_id: unique identifier (handled by pandas read_csv for now, uniqueness checked later if needed)
        # plan_name: non-empty string
        if not df['plan_name'].astype(str).str.strip().all():
            raise HTTPException(status_code=400, detail="plan_name cannot be empty.")

        # last_login_date: valid date
        try:
            df['last_login_date'] = pd.to_datetime(df['last_login_date'])
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid date format in last_login_date. Expected YYYY-MM-DD.")

        # billing_status: must be "active" or "failed"
        if not df['billing_status'].isin(["active", "failed"]).all():
            raise HTTPException(status_code=400, detail="billing_status must be 'active' or 'failed'.")

        # mrr: must be numeric (if provided)
        if 'mrr' in df.columns:
            df['mrr'] = pd.to_numeric(df['mrr'], errors='coerce')
            if df['mrr'].isnull().any():
                raise HTTPException(status_code=400, detail="mrr column must contain numeric values.")

        # 3. Preprocessing
        # Convert last_login_date into a "days_since_last_login" column.
        df['days_since_last_login'] = (datetime.now() - df['last_login_date']).dt.days

        # Normalize billing_status into numeric flags (active = 0, failed = 1).
        df['billing_flag'] = df['billing_status'].apply(lambda x: 1 if x == 'failed' else 0)

        # Fill missing values with safe defaults (e.g., mrr = 0 if missing).
        if 'mrr' not in df.columns:
            df['mrr'] = 0  # Add mrr column if it doesn't exist and fill with 0
        else:
            df['mrr'] = df['mrr'].fillna(0)

        # 4. Model Input Preparation
        # Create a clean dataframe ready for the churn prediction model.
        processed_df = df[["user_id", "days_since_last_login", "billing_flag", "plan_name", "mrr"]]

        # 5. Output
        # For testing, print the cleaned dataframe (first 10 rows).
        print("Processed DataFrame (first 10 rows):")
        print(processed_df.head(10))

        # Save the processed CSV as `processed_data.csv` locally.
        processed_df.to_csv("processed_data.csv", index=False)

        return {"message": "CSV uploaded, validated, preprocessed, and saved successfully!", "processed_rows": len(processed_df)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing CSV: {e}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)