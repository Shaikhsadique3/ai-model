from flask import Flask, request, jsonify
from flask_cors import CORS
import os
from churnaizer.src.process_csv import process_csv
from churnaizer.src.predict import ChurnPredictorService
from churnaizer.src.report_generator import generate_report_pdf
import pandas as pd
import uuid
import logging
from apscheduler.schedulers.background import BackgroundScheduler
import atexit
import time
import pytz

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
REPORTS_FOLDER = 'reports'
LOGS_FOLDER = 'logs'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(REPORTS_FOLDER):
    os.makedirs(REPORTS_FOLDER)
if not os.path.exists(LOGS_FOLDER):
    os.makedirs(LOGS_FOLDER)

@app.route('/generate-report', methods=['POST'])
def generate_report():
    job_id = request.json.get('job_id')
    log_filepath = os.path.join(LOGS_FOLDER, f"{job_id}.log")
    logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if not job_id:
        logger.error("job_id is required for report generation.")
        return jsonify({"status": "error", "message": "job_id is required"}), 400

    processed_with_predictions_filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_processed_with_predictions.csv")

    if not os.path.exists(processed_with_predictions_filepath):
        logger.error(f"File not found for job_id: {job_id} at {processed_with_predictions_filepath}")
        return jsonify({"status": "error", "message": f"File not found for job_id: {job_id}"}), 404

    try:
        predictions_df = pd.read_csv(processed_with_predictions_filepath)
        report_filename = f"{job_id}_report.pdf"
        report_filepath = os.path.join(REPORTS_FOLDER, report_filename)
        
        generate_report_pdf(predictions_df, report_filepath)
        logger.info(f"PDF report generated at {report_filepath}")

        return jsonify({"status": "success", "report_url": f"/{REPORTS_FOLDER}/{report_filename}"}), 200

    except Exception as e:
        logger.error(f"Error during report generation: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/upload-csv', methods=['POST'])
def upload_csv():
    job_id = str(uuid.uuid4())
    log_filepath = os.path.join(LOGS_FOLDER, f"{job_id}.log")
    logging.basicConfig(filename=log_filepath, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    if 'file' not in request.files:
        logger.error("No file part in the request")
        return jsonify({"status": "error", "message": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        logger.error("No selected file")
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file:
        raw_filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_raw_{file.filename}")
        file.save(raw_filepath)
        logger.info(f"Raw file saved: {raw_filepath}")

        try:
            # Process CSV
            processed_df = process_csv(raw_filepath)
            logger.info(f"CSV processed. Rows: {len(processed_df)}")
            
            # Run churn prediction
            predictions_df = predict(processed_df)
            logger.info("Churn prediction completed.")

            # Mask user_id for privacy in reports
            if 'user_id' in predictions_df.columns:
                predictions_df['user_id_masked'] = predictions_df['user_id'].apply(lambda x: f"****{str(x)[-4:]}")
            else:
                predictions_df['user_id_masked'] = 'N/A'
            logger.info("User IDs masked.")
            
            # Save processed data with predictions
            processed_with_predictions_filepath = os.path.join(UPLOAD_FOLDER, f"{job_id}_processed_with_predictions.csv")
            predictions_df.to_csv(processed_with_predictions_filepath, index=False)
            logger.info(f"Processed data with predictions saved: {processed_with_predictions_filepath}")

            # Delete raw upload file
            os.remove(raw_filepath)
            logger.info(f"Raw file deleted: {raw_filepath}")

            # Calculate summary statistics
            total_customers = len(predictions_df)
            churn_risk_counts = predictions_df['churn_risk'].value_counts(normalize=True) * 100
            high_risk_percent = churn_risk_counts.get('high', 0.0)
            medium_risk_percent = churn_risk_counts.get('medium', 0.0)
            low_risk_percent = churn_risk_counts.get('low', 0.0)

            if 'churn_reason' in predictions_df.columns:
                top_churn_reasons = predictions_df['churn_reason'].value_counts().nlargest(3).to_dict()
            else:
                top_churn_reasons = {}

            summary = {
                "total_customers": total_customers,
                "high_risk_percent": round(high_risk_percent, 2),
                "medium_risk_percent": round(medium_risk_percent, 2),
                "low_risk_percent": round(low_risk_percent, 2),
                "top_churn_reasons": top_churn_reasons
            }
            logger.info(f"Summary statistics calculated: {summary}")

            # Prepare preview (first 10 rows with predictions)
            preview = predictions_df.head(10).to_dict(orient='records')
            logger.info("Preview generated.")

            return jsonify({"status": "ok", "rows": total_customers, "summary": summary, "preview": preview}), 200

        except Exception as e:
            logger.error(f"Error during file processing: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500

def cleanup_old_files():
    now = time.time()
    for folder in [UPLOAD_FOLDER, REPORTS_FOLDER, LOGS_FOLDER]:
        for filename in os.listdir(folder):
            filepath = os.path.join(folder, filename)
            if os.path.isfile(filepath):
                # Files expire in 24 hours (86400 seconds) unless archived (not implemented yet)
                if now - os.path.getmtime(filepath) > 86400:
                    os.remove(filepath)
                    logging.info(f"Cleaned up old file: {filepath}")

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=cleanup_old_files, trigger="interval", hours=1, timezone=pytz.utc)
    scheduler.start()
    atexit.register(lambda: scheduler.shutdown())
    app.run(debug=True)