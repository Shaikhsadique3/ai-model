import os
import shutil
import uuid
import pandas as pd
import subprocess
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from typing import List
from pydantic import BaseModel
from processing.process_csv import DataProcessingError

# Configure logging
log_directory = "logs"
os.makedirs(log_directory, exist_ok=True)
logging.basicConfig(filename=os.path.join(log_directory, "api.log"), level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI()

@app.get("/health")
async def health():
    logging.info("Health check endpoint called.")
    return {"status": "ok", "message": "API running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    logging.info(f"Predict endpoint called with file: {file.filename}")
    if not file.filename.endswith(".csv"):
        logging.error(f"Invalid file format uploaded: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid file format. Please upload a CSV file.")

    # Limit file size to 5MB
    MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
    file_contents = await file.read()
    if len(file_contents) > MAX_FILE_SIZE:
        logging.error(f"File size exceeds limit for {file.filename}")
        raise HTTPException(status_code=400, detail="File size exceeds the 5MB limit.")

    upload_dir = "uploads"
    os.makedirs(upload_dir, exist_ok=True)
    file_location = os.path.join(upload_dir, f"{uuid.uuid4()}_{file.filename}")

    with open(file_location, "wb") as buffer:
        buffer.write(file_contents)
    logging.info(f"File saved to {file_location}")

    # Call processing/process_csv.py
    processed_csv_path = os.path.join(upload_dir, f"processed_{os.path.basename(file_location)}")
    try:
        logging.info(f"Calling process_csv.py for {file_location}")
        subprocess.run(["python", "processing/process_csv.py", "--input_csv", file_location, "--output_csv", processed_csv_path], check=True)
        logging.info(f"CSV processed and saved to {processed_csv_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"CSV processing failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {e}")
    except DataProcessingError as e:
        logging.error(f"Invalid CSV data for {file.filename}: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CSV data: {e}")

    # Call model/predict_model.py
    predictions_csv_path = os.path.join(upload_dir, f"predictions_{os.path.basename(file_location)}")
    try:
        logging.info(f"Calling predict_model.py for {processed_csv_path}")
        subprocess.run(["python", "model/predict_model.py", "--input_csv", processed_csv_path, "--output_csv", predictions_csv_path], check=True)
        logging.info(f"Predictions generated and saved to {predictions_csv_path}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Model prediction failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Model prediction failed: {e}")

    # Read predictions and generate summary
    try:
        predictions_df = pd.read_csv(predictions_csv_path)
        total_customers = len(predictions_df)
        high_risk_percent = (predictions_df['risk_level'] == 'high').sum() / total_customers * 100
        medium_risk_percent = (predictions_df['risk_level'] == 'medium').sum() / total_customers * 100
        low_risk_percent = (predictions_df['risk_level'] == 'low').sum() / total_customers * 100

        preview = predictions_df.head(10).to_dict(orient="records")
        logging.info(f"Prediction summary generated for {file.filename}")
        return {
            "total_customers": total_customers,
            "high_risk_percent": round(high_risk_percent, 2),
            "medium_risk_percent": round(medium_risk_percent, 2),
            "low_risk_percent": round(low_risk_percent, 2),
            "preview": preview
        }
    except Exception as e:
        logging.error(f"Error generating prediction summary for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating prediction summary: {e}")


class ReportRequest(BaseModel):
    file_path: str

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    logging.info(f"Generate report endpoint called for file: {request.file_path}")
    predictions_file_path = request.file_path
    if not os.path.exists(predictions_file_path):
        logging.error(f"Predictions file not found: {predictions_file_path}")
        raise HTTPException(status_code=404, detail="Predictions file not found.")

    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    report_filename = f"churn_report_{uuid.uuid4()}.pdf"
    report_path = os.path.join(report_dir, report_filename)

    try:
        logging.info(f"Calling report_generator.py for {predictions_file_path}")
        subprocess.run(["python", "report/report_generator.py", "--input_csv", predictions_file_path, "--output_pdf", report_path], check=True)
        logging.info(f"Report generated and saved to {report_path}")
        return {"status": "success", "report_url": f"/{report_path}"}
    except subprocess.CalledProcessError as e:
        logging.error(f"Report generation failed for {predictions_file_path}: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {e}")