from fastapi import FastAPI
import uvicorn
import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sys
import logging
from churnaizer.src.predict import ChurnPredictorService
from churnaizer.src.report_generator import generate_report_pdf

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

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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

@app.post("/generate-report")
async def generate_report(file: UploadFile = File(...)):
    logging.info(f"Report generation started for file: {file.filename}")

    UPLOADS_DIR = "./uploads"
    REPORTS_DIR = "./reports"
    os.makedirs(UPLOADS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)

    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    # Limit file size to 5MB
    MAX_FILE_SIZE_MB = 5
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail=f"File size exceeds {MAX_FILE_SIZE_MB}MB limit.")

    temp_filepath = os.path.join(UPLOADS_DIR, file.filename)
    with open(temp_filepath, "wb") as buffer:
        buffer.write(contents)

    try:
        processed_df, stats_summary, warnings = process_csv(temp_filepath)
        if warnings:
            for warning in warnings:
                logging.warning(f"CSV processing warning: {warning}")

        if processed_df is None:
            os.remove(temp_filepath)
            logging.error(f"CSV processing failed for {file.filename}")
            return jsonify({"status": "error", "message": "CSV processing failed: " + ". ".join(warnings)}), 400

        logging.info(f"Processed {len(processed_df)} rows from {file.filename}")

        # Save processed_data.csv
        processed_data_filename = f"processed_data_{uuid.uuid4()}.csv"
        processed_data_filepath = os.path.join(UPLOADS_DIR, processed_data_filename)
        processed_df.to_csv(processed_data_filepath, index=False)

        # Save stats_summary.json
        stats_summary_filename = f"stats_summary_{uuid.uuid4()}.json"
        stats_summary_filepath = os.path.join(UPLOADS_DIR, stats_summary_filename)
        with open(stats_summary_filepath, 'w') as f:
            json.dump(stats_summary, f, indent=4)

        # ii. Run churn prediction model
        predictor = ChurnPredictorService()
        processed_with_predictions_df = predictor.predict_batch(processed_df)
        logging.info(f"Churn prediction completed for {file.filename}")

        # Save processed_with_predictions.csv
        processed_with_predictions_filename = f"processed_with_predictions_{uuid.uuid4()}.csv"
        processed_with_predictions_filepath = os.path.join(UPLOADS_DIR, processed_with_predictions_filename)
        processed_with_predictions_df.to_csv(processed_with_predictions_filepath, index=False)

        # iii. Run visualization & report builder
        report_filename = f"churn_report_{uuid.uuid4()}.pdf"
        report_filepath = os.path.join(REPORTS_DIR, report_filename)
        generate_report_pdf(processed_with_predictions_df, stats_summary, report_filepath)
        logging.info(f"PDF report generated at {report_filepath}")

        # iv. Save final PDF report (already done in generate_report_pdf)

        os.remove(temp_filepath)
        os.remove(processed_data_filepath)
        os.remove(stats_summary_filepath)
        os.remove(processed_with_predictions_filepath)
        logging.info(f"Report generation completed successfully for file: {file.filename}")
        return {"status": "success", "report_url": f"/reports/{report_filename}"}

    except Exception as e:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)
        logging.error(f"An unexpected error occurred during processing for {file.filename}: {str(e)}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred during processing: {str(e)}"}), 500

    os.remove(temp_filepath)
    return {"status": "success", "report_url": "/reports/dummy_report.pdf"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)