# Standard library imports
import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

# Third-party library imports
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import asyncio
from pathlib import Path
import aiofiles

# Import our processing modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from processing.process_csv import process_csv
from model.predict_model import ChurnPredictorService
from report.report_generator import generate_report_pdf

# Configure logging
# Configure logging for the application
logging.basicConfig(
    level=logging.INFO,  # Set the logging level to INFO
    format='%(asctime)s - %(level)s - %(message)s',  # Define log message format
    handlers=[
        logging.FileHandler('logs/api.log'),  # Log to a file named api.log
        logging.StreamHandler()  # Also output logs to the console
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories if they don't already exist
os.makedirs('uploads', exist_ok=True)  # Directory for storing uploaded CSV files
os.makedirs('reports', exist_ok=True)  # Directory for storing generated PDF reports
os.makedirs('logs', exist_ok=True)      # Directory for storing application logs

# Initialize FastAPI application with a title and version
app = FastAPI(title="Churn Audit Service API", version="1.0.0")

# Configure CORS (Cross-Origin Resource Sharing) middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (for development purposes)
    allow_credentials=True,  # Allow cookies to be included in requests
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers in requests
)

# Pydantic models
class PredictRequest(BaseModel):
    file_id: str

class ReportRequest(BaseModel):
    file_id: str

# Global storage for file metadata (in production, use a database)
file_storage: Dict[str, Any] = {}

async def _process_and_predict_background(file_id: str, upload_path: str):
    """Background task to process CSV, generate predictions, and create report.

    This function orchestrates the entire churn prediction pipeline:
    1. Processes the uploaded CSV file.
    2. Generates churn predictions using the trained model.
    3. Creates a comprehensive PDF report based on the predictions.

    Args:
        file_id (str): The unique identifier for the uploaded file.
        upload_path (str): The file path where the original CSV was uploaded.
    """
    try:
        file_storage[file_id]["status"] = "processing"
        logger.info(f"[{file_id}] Starting CSV processing.")
        
        # 1. Process CSV
        processed_df = process_csv(upload_path)
        
        file_storage[file_id]["status"] = "predicting"
        logger.info(f"[{file_id}] Starting churn prediction.")

        # 2. Generate predictions
        predictor = ChurnPredictorService()
        predictions = predictor.predict_churn(processed_df)
        file_storage[file_id]["predictions"] = predictions.to_dict(orient="records")

        file_storage[file_id]["status"] = "reporting"
        logger.info(f"[{file_id}] Starting report generation.")

        # 3. Generate report
        report_filename = f"churn_report_{file_id}.pdf"
        report_path = os.path.join("reports", report_filename)
        generate_report_pdf(predictions, report_path)
        file_storage[file_id]["report_path"] = report_path

        file_storage[file_id]["status"] = "completed"
        logger.info(f"[{file_id}] Processing, prediction, and reporting completed successfully.")

    except Exception as e:
        file_storage[file_id]["status"] = "failed"
        file_storage[file_id]["error"] = str(e)
        logger.error(f"[{file_id}] Error during background processing: {e}")


@app.post("/upload", summary="Upload CSV for Churn Prediction", description="Uploads a CSV file containing customer data for churn prediction. The file is processed in the background, and predictions are generated. Returns a file_id to track the processing status.")
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload CSV file and start background processing.

    Args:
        background_tasks (BackgroundTasks): FastAPI's dependency for background tasks.
        file (UploadFile): The CSV file to upload.

    Returns:
        dict: A dictionary containing the file_id and initial status.
    """
    try:
        file_id = str(uuid.uuid4())
        upload_folder = "uploads"
        os.makedirs(upload_folder, exist_ok=True)
        upload_path = os.path.join(upload_folder, f"{file_id}_{file.filename}")
        
        async with aiofiles.open(upload_path, "wb") as out_file:
            while content := await file.read(1024):
                await out_file.write(content)

        file_storage[file_id] = {
            "filename": file.filename,
            "upload_path": upload_path,
            "status": "pending",
            "report_path": None,
            "predictions": None,
            "error": None,
            "timestamp": datetime.now()
        }

        background_tasks.add_task(_process_and_predict_background, file_id, file_storage[file_id]['upload_path'])

        logger.info(f"File uploaded and background task started for file_id: {file_id}")

        return {"file_id": file_id, "status": "pending"}

    except HTTPException as e:
        logger.error(f"HTTPException in upload_csv for file {file.filename}: {e.detail}")
        raise e
    except Exception as e:
        logger.error(f"Error in upload_csv for file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {e}")





@app.get("/download-report/{file_id}", summary="Download Churn Report", description="Downloads the generated PDF churn report for a given file_id. The report is only available after the processing status is 'completed'.")
async def download_report(file_id: str):
    report_path = file_storage[file_id]['report_path']
    if not report_path or not os.path.exists(report_path):
        raise HTTPException(status_code=404, detail="Report not found or not yet generated.")
    return FileResponse(report_path, media_type="application/pdf", filename=f"churn_report_{file_id}.pdf")

@app.get("/predictions/{file_id}", summary="Get Prediction Results", description="Retrieves the detailed churn prediction results for a processed file, identified by its file_id. This includes risk distribution, top churn reasons, and benchmark comparisons.")
async def get_predictions(file_id: str):
    predictions = file_storage[file_id].get('predictions')
    if not predictions:
        raise HTTPException(status_code=404, detail="Predictions not found or not yet generated.")
    return JSONResponse(content=predictions)

@app.get("/", summary="Service Status", description="Returns a simple message indicating the service is running and its version.")
async def root():
    """
    Returns a simple message indicating the service is running.

    Returns:
        dict: A dictionary with a welcome message and API version.
    """
    return {"message": "Churn Audit Service API", "version": "1.0.0"}

@app.get("/sample-csv", summary="Download Sample CSV", description="Downloads a sample CSV file to help users understand the required format for data upload.")
async def download_sample_csv():
    sample_csv_path = "./data/sample_data.csv"
    if not os.path.exists(sample_csv_path):
        raise HTTPException(status_code=404, detail="Sample CSV not found.")
    return FileResponse(sample_csv_path, media_type="text/csv", filename="sample_data.csv")



# Cleanup task to remove old files
# @app.on_event("startup")
# async def startup_event():
    """Cleanup old files on startup."""
    try:
        # Clean up files older than 24 hours
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        for directory in ['uploads', 'reports']:
            if os.path.exists(directory):
                for filename in os.listdir(directory):
                    filepath = os.path.join(directory, filename)
                    if os.path.isfile(filepath):
                        file_time = datetime.fromtimestamp(os.path.getctime(filepath))
                        if file_time < cutoff_time:
                            os.remove(filepath)
                            logger.info(f"Cleaned up old file: {filepath}")
        
        logger.info("Startup cleanup completed")
    except Exception as e:
        logger.error(f"Error during startup cleanup: {e}")

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application using Uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) # Listen on all available interfaces on port 8000