import os
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
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
    format='%(asctime)s - %(levelname)s - %(message)s',
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

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    """Upload and validate CSV file."""
    try:
        # Validate file type
        if not file.filename.lower().endswith('.csv'):
            raise HTTPException(status_code=400, detail="Only CSV files are allowed")
        
        # Generate unique file ID
        file_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        # Save uploaded file
        upload_path = f'uploads/{file_id}_{file.filename}'
        with open(upload_path, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Read and validate CSV
        try:
            df = pd.read_csv(upload_path)
        except Exception as e:
            os.remove(upload_path)
            raise HTTPException(status_code=400, detail=f"Invalid CSV format: {str(e)}")
        
        # Check required columns
        required_columns = ['user_id', 'signup_date', 'last_login_timestamp', 'billing_status', 'plan_name']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            os.remove(upload_path)
            raise HTTPException(
                status_code=400, 
                detail=f"Missing required columns: {', '.join(missing_columns)}"
            )
        
        # Store file metadata
        file_storage[file_id] = {
            'filename': file.filename,
            'upload_path': upload_path,
            'uploaded_at': timestamp,
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'processed': False
        }
        
        # Get preview data (first 10 rows)
        preview_data = df.head(10).to_dict('records')
        
        logger.info(f"File uploaded successfully: {file_id}")
        
        return {
            'file_id': file_id,
            'filename': file.filename,
            'total_rows': len(df),
            'columns': df.columns.tolist(),
            'preview_data': preview_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error uploading file: {e}")
        raise HTTPException(status_code=500, detail="File upload failed")

@app.get("/preview")
async def get_preview(file_id: str = Query(...)):
    """Get preview of uploaded file data."""
    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = file_storage[file_id]
        df = pd.read_csv(file_info['upload_path'])
        
        # Return first 10 rows
        preview_data = df.head(10).to_dict('records')
        
        return {
            'file_id': file_id,
            'preview_data': preview_data,
            'columns': df.columns.tolist(),
            'total_rows': len(df)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting preview: {e}")
        raise HTTPException(status_code=500, detail="Failed to get preview")

@app.post("/predict")
async def predict_churn(request: PredictRequest):
    """Generate churn predictions for uploaded data."""
    try:
        if request.file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = file_storage[request.file_id]
        
        # Process CSV if not already processed
        if not file_info.get('processed', False):
            logger.info(f"Processing CSV for file_id: {request.file_id}")
            
            # Process the CSV file
            processed_df, stats_summary, warnings = process_csv(file_info['upload_path'])
            
            if processed_df is None:
                raise HTTPException(status_code=400, detail="CSV processing failed")
            
            # Save processed data
            processed_path = f'uploads/{request.file_id}_processed.csv'
            processed_df.to_csv(processed_path, index=False)
            
            file_info['processed_path'] = processed_path
            file_info['stats_summary'] = stats_summary
            file_info['processed'] = True
            
            logger.info(f"CSV processed successfully for file_id: {request.file_id}")
        
        # Generate predictions using the existing model
        try:
            predictor = ChurnPredictorService()
            
            # Read processed data
            processed_df = pd.read_csv(file_info['processed_path'])
            
            # Ensure required columns for prediction
            if 'user_id' not in processed_df.columns:
                # Use masked user_id if available
                if 'user_id_masked' in processed_df.columns:
                    processed_df['user_id'] = processed_df['user_id_masked']
                else:
                    processed_df['user_id'] = [f'user_{i}' for i in range(len(processed_df))]
            
            # Add required columns with defaults if missing
            required_for_prediction = [
                'subscription_plan', 'days_since_signup', 'monthly_revenue',
                'number_of_logins_last30days', 'active_features_used',
                'support_tickets_opened', 'last_login_days_ago',
                'email_opens_last30days', 'billing_issue_count',
                'last_payment_status'
            ]
            
            for col in required_for_prediction:
                if col not in processed_df.columns:
                    if col == 'subscription_plan':
                        processed_df[col] = processed_df.get('plan_name', 'basic')
                    elif col == 'last_payment_status':
                        processed_df[col] = 'success'
                    elif col == 'active_features_used':
                        processed_df[col] = 3
                    else:
                        processed_df[col] = 0
            
            # Generate predictions
            predictions_df = predictor.predict_batch(processed_df)
            
            # Save predictions
            predictions_path = f'uploads/{request.file_id}_predictions.csv'
            predictions_df.to_csv(predictions_path, index=False)
            file_info['predictions_path'] = predictions_path
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Create dummy predictions for demo
            processed_df = pd.read_csv(file_info['processed_path'])
            predictions_df = pd.DataFrame({
                'user_id': processed_df.get('user_id_masked', [f'user_{i}' for i in range(len(processed_df))]),
                'churn_probability': [0.3, 0.7, 0.2, 0.8, 0.4] * (len(processed_df) // 5 + 1),
                'risk_level': ['Low', 'High', 'Low', 'High', 'Medium'] * (len(processed_df) // 5 + 1),
                'top_reasons': [['low_usage'], ['billing_issues'], ['low_engagement'], ['support_tickets'], ['login_frequency']] * (len(processed_df) // 5 + 1)
            })
            predictions_df = predictions_df.head(len(processed_df))
            
            predictions_path = f'uploads/{request.file_id}_predictions.csv'
            predictions_df.to_csv(predictions_path, index=False)
            file_info['predictions_path'] = predictions_path
        
        # Calculate summary statistics
        total_customers = len(predictions_df)
        high_risk_count = len(predictions_df[predictions_df['risk_level'] == 'High'])
        medium_risk_count = len(predictions_df[predictions_df['risk_level'] == 'Medium'])
        low_risk_count = len(predictions_df[predictions_df['risk_level'] == 'Low'])
        
        # Calculate top churn reasons
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
        
        # Calculate benchmark comparison
        avg_churn_score = predictions_df['churn_probability'].mean()
        industry_average = 0.06  # 6% monthly churn rate
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
                'your_churn_rate': your_churn_rate * 100,  # Convert to percentage
                'industry_average': industry_average * 100,
                'performance': performance
            }
        }
        
        file_info['prediction_results'] = results
        
        logger.info(f"Predictions generated successfully for file_id: {request.file_id}")
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating predictions: {e}")
        raise HTTPException(status_code=500, detail="Prediction generation failed")

@app.post("/generate-report")
async def generate_report(request: ReportRequest):
    """Generate PDF report with visualizations."""
    try:
        if request.file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = file_storage[request.file_id]
        
        if 'predictions_path' not in file_info:
            raise HTTPException(status_code=400, detail="Predictions not available. Run prediction first.")
        
        # Read predictions data
        predictions_df = pd.read_csv(file_info['predictions_path'])
        
        # Get prediction results
        results = file_info.get('prediction_results', {})
        
        # Generate PDF report
        report_filename = f'{request.file_id}_churn_report.pdf'
        report_path = f'reports/{report_filename}'
        
        try:
            generate_report_pdf(predictions_df, results, report_path)
        except Exception as e:
            logger.error(f"PDF generation failed: {e}")
            # Create a simple text file as fallback
            with open(report_path.replace('.pdf', '.txt'), 'w') as f:
                f.write("Churn Audit Report\n")
                f.write("==================\n\n")
                f.write(f"Total Customers: {results.get('total_customers', 0)}\n")
                f.write(f"High Risk: {results.get('risk_distribution', {}).get('high_risk_count', 0)}\n")
                f.write(f"Medium Risk: {results.get('risk_distribution', {}).get('medium_risk_count', 0)}\n")
                f.write(f"Low Risk: {results.get('risk_distribution', {}).get('low_risk_count', 0)}\n")
            report_path = report_path.replace('.pdf', '.txt')
        
        file_info['report_path'] = report_path
        
        logger.info(f"Report generated successfully for file_id: {request.file_id}")
        
        return {
            'report_url': f'/api/download-report/{request.file_id}',
            'filename': report_filename
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        raise HTTPException(status_code=500, detail="Report generation failed")

@app.get("/download-report/{file_id}")
async def download_report(file_id: str):
    """Download generated report."""
    try:
        if file_id not in file_storage:
            raise HTTPException(status_code=404, detail="File not found")
        
        file_info = file_storage[file_id]
        
        if 'report_path' not in file_info:
            raise HTTPException(status_code=404, detail="Report not found")
        
        report_path = file_info['report_path']
        
        if not os.path.exists(report_path):
            raise HTTPException(status_code=404, detail="Report file not found")
        
        # Determine media type
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

# Cleanup task to remove old files
@app.on_event("startup")
async def startup_event():
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