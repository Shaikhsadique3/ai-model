from fastapi import FastAPI
import uvicorn
import os
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import sys
import logging
import os
import logging

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from churnaizer.src.predict import ChurnPredictorService
from churnaizer.src.report_generator import generate_report_pdf
from churnaizer.src.process_csv import process_csv
from fastapi.middleware.cors import CORSMiddleware

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse
import uvicorn
import os
import shutil
import uuid
import sys
import logging
import io
import os
import shutil
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from starlette.background import BackgroundTasks
import io
import sys
import uuid

# Ensure the churnaizer package is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'churnaizer')))

from churnaizer.src.predict import ChurnPredictorService
from churnaizer.src.report_generator import generate_report_pdf
from churnaizer.src.process_csv import process_csv

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3002"],  # Allow requests from your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

UPLOADS_DIR = "./uploads"
REPORTS_DIR = "./reports"
TEMP_DIR = "./temp" # New temporary directory

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True) # Create the new temporary directory

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Churn Analyzer API!"}

@app.post("/upload-csv")
async def upload_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Invalid file type. Only CSV files are allowed.")

    file_location = os.path.join(UPLOADS_DIR, file.filename)
    try:
        with open(file_location, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process CSV to get rows and columns
        processed_df, stats_summary, warnings = process_csv(file_location)
        num_rows = len(processed_df) if processed_df is not None else 0
        columns = processed_df.columns.tolist() if processed_df is not None else []
        
        logging.info(f"Uploaded file: {file.filename}, Rows: {num_rows}, Columns: {columns}")

        return {"status": "ok", "file_path": file_location, "rows": num_rows, "columns": columns}
    except Exception as e:
        logging.error(f"Error processing uploaded file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing CSV file: {e}")

@app.post("/generate-report")
async def generate_report(file: UploadFile = File(...)):
    logging.info(f"Received request to generate report for file: {file.filename}")
    temp_csv_path = ""
    pdf_output_path = ""
    try:
        # Save the uploaded file temporarily
        temp_csv_path = os.path.join(TEMP_DIR, f"{uuid.uuid4()}_{file.filename}")
        with open(temp_csv_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logging.info(f"File saved temporarily at: {temp_csv_path}")

        # Process the CSV file
        processed_df, stats_summary, warnings = process_csv(temp_csv_path)
        if warnings:
            for warning in warnings:
                logging.warning(f"CSV processing warning: {warning}")

        if processed_df is None:
            logging.error("CSV processing failed: No processed data returned.")
            raise HTTPException(status_code=400, detail="Failed to process CSV data.")

        logging.info(f"Successfully processed {len(processed_df)} rows from CSV.")

        # Generate PDF report
        pdf_output_path = os.path.join(TEMP_DIR, f"report_{uuid.uuid4()}.pdf")
        generate_report_pdf(processed_df, stats_summary, pdf_output_path)
        logging.info(f"PDF report generated at: {pdf_output_path}")

        # Read the generated PDF
        with open(pdf_output_path, "rb") as pdf_file:
            pdf_content = pdf_file.read()

        return StreamingResponse(io.BytesIO(pdf_content), media_type="application/pdf",
                                 headers={"Content-Disposition": f"attachment; filename=churn_report_{uuid.uuid4()}.pdf"})

    except Exception as e:
        logging.error(f"Error generating report: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_csv_path):
            os.remove(temp_csv_path)
            logging.info(f"Cleaned up temporary file: {temp_csv_path}")
        if os.path.exists(pdf_output_path):
            os.remove(pdf_output_path)
            logging.info(f"Cleaned up temporary PDF file: {pdf_output_path}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)