import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import asyncio
from pathlib import Path

# Import our processing modules
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from processing.process_csv import process_csv
from model.predict_model import ChurnPredictorService
from report.report_generator import generate_report_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(level)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs('uploads', exist_ok=True)
os.makedirs('reports', exist_ok=True)
os.makedirs('logs', exist_ok=True)

app = FastAPI(title="Churn Audit Service API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class PredictRequest(BaseModel):
    file_id: str

class ReportRequest(BaseModel):
    file_id: str

# Global storage for file metadata (in production, use a database)
file_storage = {}

async def _process_and_predict_background(file_id: str, upload_path: str):
    """Background task to process CSV, generate predictions, and create report."""
    logger.info(f"Starting background processing for file_id: {file_id}")
    file_storage[file_id]['status'] = 'in_progress'
    file_storage[file_id]['progress'] = 0

    try:
        # 1. Process the CSV file
        logger.info(f"Processing CSV for file_id: {file_id}")
        processed_df, stats_summary, warnings = process_csv(upload_path)

        if processed_df is None:
            file_storage[file_id]['error'] = "CSV processing failed or returned empty data. " + " ".join(warnings)
            raise ValueError("CSV processing failed or returned empty data.")

        processed_path = f'uploads/{file_id}_processed.csv'
        processed_df.to_csv(processed_path, index=False)

        file_storage[file_id]['processed_path'] = processed_path
        file_storage[file_id]['stats_summary'] = stats_summary
        file_storage[file_id]['processed'] = True
        file_storage[file_id]['progress'] = 33
        logger.info(f"CSV processed successfully for file_id: {file_id}")

        # 2. Generate predictions
        logger.info(f"Generating predictions for file_id: {file_id}")
        predictor = ChurnPredictorService()

        # Ensure required columns for prediction
        if 'user_id' not in processed_df.columns:
            if 'user_id_masked' in processed_df.columns:
                processed_df['user_id'] = processed_df['user_id_masked']
            else:
                processed_df['user_id'] = [f'user_{i}' for i in range(len(processed_df))]

        required_for_prediction = [
            'subscription_plan', 'days_since_signup', 'monthly_revenue',
            'number_of_logins_last30days', 'active_features_used',
            'support_tickets_opened', 'last_login_days_ago',
            'email_opens_last30days', 'billing_issue_count',
            'last_payment_status', 'avg_session_duration', 'trial_conversion_flag'
        ]

        for col in required_for_prediction:
            if col not in processed_df.columns:
                if col == 'subscription_plan':
                    processed_df[col] = processed_df.get('plan_name', 'basic')
                elif col == 'last_payment_status':
                    processed_df[col] = 'success'
                elif col == 'active_features_used':
                    processed_df[col] = 3
                elif col == 'avg_session_duration':
                    processed_df[col] = 0  # Default for avg_session_duration
                elif col == 'trial_conversion_flag':
                    processed_df[col] = 0  # Default for trial_conversion_flag
                else:
                    processed_df[col] = 0

        predictions_df = predictor.predict_batch(processed_df)

        predictions_path = f'uploads/{file_id}_predictions.csv'
        predictions_df.to_csv(predictions_path, index=False)
        file_storage[file_id]['predictions_path'] = predictions_path

        # Calculate summary statistics for results
        total_customers = len(predictions_df)
        high_risk_count = len(predictions_df[predictions_df['risk_level'] == 'High'])
        medium_risk_count = len(predictions_df[predictions_df['risk_level'] == 'Medium'])
        low_risk_count = len(predictions_df[predictions_df['risk_level'] == 'Low'])

        all_reasons = []
        for reasons in predictions_df['top_reasons']:
            if isinstance(reasons, str):
                all_reasons.extend([r.strip() for r in reasons.split(',')])
            elif isinstance(reasons, list):
                all_reasons.extend(reasons)

        reason_counts = pd.Series(all_reasons).value_counts().head(5)
        top_churn_reasons = [
            {
                'reason': reason,
                'count': count,
                'percentage': (count / total_customers) * 100
            }
            for reason, count in reason_counts.items()
        ]

        avg_churn_score = predictions_df['churn_probability'].mean()
        industry_average = 0.06
        your_churn_rate = avg_churn_score

        if your_churn_rate < industry_average * 0.9:
            performance = 'above'
        elif your_churn_rate > industry_average * 1.1:
            performance = 'below'
        else:
            performance = 'average'

        results = {
            'total_customers': total_customers,
            'risk_distribution': {
                'high_risk_count': high_risk_count,
                'medium_risk_count': medium_risk_count,
                'low_risk_count': low_risk_count,
                'high_risk_percent': (high_risk_count / total_customers) * 100,
                'medium_risk_percent': (medium_risk_count / total_customers) * 100,
                'low_risk_percent': (low_risk_count / total_customers) * 100
            },
            'top_churn_reasons': top_churn_reasons,
            'average_churn_score': avg_churn_score,
            'benchmark_comparison': {
                'your_churn_rate': your_churn_rate * 100,
                'industry_average': industry_average * 100,
                'performance': performance
            }
        }

        file_storage[file_id]['prediction_results'] = results
        file_storage[file_id]['progress'] = 66
        logger.info(f"Predictions generated successfully for file_id: {file_id}")

        # 3. Generate PDF report
        logger.info(f"Generating report for file_id: {file_id}")
        report_filename = f'{file_id}_churn_report.pdf'
        report_path = os.path.join('reports', report_filename)
        logger.info(f"Report path for file_id {file_id}: {report_path}")

        generate_report_pdf(predictions_df, results, report_path, chart_dir="temp")

        file_storage[file_id]['report_path'] = report_path
        file_storage[file_id]['report_url'] = f'/api/download-report/{file_id}'
        file_storage[file_id]['status'] = 'completed'
        file_storage[file_id]['progress'] = 100
        logger.info(f"Report generated successfully for file_id: {file_id}")

    except Exception as e:
        logger.error(f"Background processing failed for file_id: {file_id} - {e}")
        file_storage[file_id]['status'] = 'failed'
        file_storage[file_id]['error'] = str(e)

@app.post("/upload")
async def upload_csv(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Upload CSV file and start background processing."""
    try:
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")

        file_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()

        upload_path = f'uploads/{file_id}_{file.filename}'
        with open(upload_path, 'wb') as f:
            content = await file.read()
            f.write(content)

        file_storage[file_id] = {
            'filename': file.filename,
            'upload_path': upload_path,
            'uploaded_at': timestamp,
            'status': 'pending',
            'progress': 0
        }

        background_tasks.add_task(_process_and_predict_background, file_id, upload_path)

        logger.info(f"File uploaded and background task started for file_id: {file_id}")

        return {"file_id": file_id, "status": "pending"}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@app.get("/status/{file_id}")
async def get_processing_status(file_id: str):
    """Get the current processing status of a file."""
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_storage[file_id]
    return {
        "file_id": file_id,
        "status": file_info.get('status'),
        "progress": file_info.get('progress', 0),
        "report_url": file_info.get('report_url'),
        "error": file_info.get('error')
    }

@app.get("/download-report/{file_id}")
async def download_report(file_id: str):
    """Download generated report."""
    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")

        file_info = file_storage[file_id]

        if file_info.get('status') != 'completed' or 'report_path' not in file_info:
            raise HTTPException(status_code=404, detail="Report not found or not yet generated.")

        report_path = file_info['report_path']

        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report file not found")

        media_type = 'application/pdf' if report_path.endswith('.pdf') else 'text/plain'
        filename = f'churn_audit_report_{file_id}.{"pdf" if report_path.endswith(".pdf") else "txt"}'

        return FileResponse(
            report_path,
            media_type=media_type,
            filename=filename
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error downloading report: {e}")
        raise HTTPException(status_code=500, detail="Report download failed")

@app.get("/predictions/{file_id}")
async def get_predictions(file_id: str):
    """Get prediction results for a processed file."""
    if file_id not in file_storage:
        raise HTTPException(status_code=404, detail="File not found")

    file_info = file_storage[file_id]
    if file_info.get('status') != 'completed' or 'prediction_results' not in file_info:
        raise HTTPException(status_code=404, detail="Predictions not available or processing not completed.")

    return file_info['prediction_results']

@app.get("/")
async def root():
    return {"message": "Churn Audit Service API", "version": "1.0.0"}

@app.get("/sample-csv")
async def download_sample_csv():
    """Download a sample CSV file with the required format."""
    try:
        # Create sample data
        sample_data = {
            'user_id': ['user_001', 'user_002', 'user_003', 'user_004', 'user_005'],
            'signup_date': ['2023-01-15', '2023-02-20', '2023-03-10', '2023-01-05', '2023-04-12'],
            'last_login_timestamp': ['2024-01-10 14:30:00', '2024-01-08 09:15:00', '2024-01-12 16:45:00', '2023-12-20 11:20:00', '2024-01-11 13:10:00'],
            'billing_status': ['active', 'active', 'failed', 'active', 'active'],
            'plan_name': ['Pro', 'Basic', 'Pro', 'Enterprise', 'Basic'],
            'monthly_revenue': [99.99, 29.99, 99.99, 299.99, 29.99],
            'support_tickets_opened': [2, 0, 5, 1, 1],
            'email_opens_last30days': [15, 8, 3, 20, 12],
            'last_payment_status': ['success', 'success', 'failed', 'success', 'success']
        }
        
        df = pd.DataFrame(sample_data)
        sample_path = 'uploads/sample_customer_data.csv'
        df.to_csv(sample_path, index=False)
        
        return FileResponse(
            sample_path,
            media_type='text/csv',
            filename='sample_customer_data.csv'
        )
    except Exception as e:
        logger.error(f"Error creating sample CSV: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate sample CSV")

@app.post("/predict")
async def predict_churn(request: PredictRequest):
    """Generate churn predictions for uploaded data."""
    raise HTTPException(status_code=405, detail="Use /upload endpoint for processing and predictions.")

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate PDF report with visualizations."""
    raise HTTPException(status_code=405, detail="Report generation is part of background processing via /upload.")

@app.post("/upload-csv")
async def upload_csv_old(file: UploadFile = File(...)):
    """Deprecated: Use /upload endpoint instead."""
    raise HTTPException(status_code=405, detail="Use /upload endpoint instead of /upload-csv.")

@app.get("/preview")
async def get_preview_old(file_id: str = Query(...)):
    """Deprecated: Preview data is returned directly from /upload or can be inferred from status."""
    raise HTTPException(status_code=405, detail="Preview data is handled by the /upload and /status endpoints.")

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
    uvicorn.run(app, host="0.0.0.0", port=8000)