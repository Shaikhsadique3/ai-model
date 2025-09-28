# Model Utilization Guide

This document outlines the requirements and procedures for effectively utilizing the churn prediction model.

## 1. System Requirements


### Minimum Hardware Specifications
- **CPU**: Multi-core processor recommended for faster inference.
- **RAM**: Minimum 4GB, 8GB+ recommended for handling larger datasets or concurrent requests.
- **GPU**: Not explicitly required, as the model is primarily CPU-bound.

### Required Software Dependencies and Libraries
The following Python libraries are required. These can be installed using `pip` and the provided `requirements.txt` files.

**Root `requirements.txt`:**
- `Flask`
- `flask-cors`
- `pandas`
- `scikit-learn`
- `reportlab`
- `matplotlib`
- `APScheduler`
- `uvicorn`
- `numpy`
- `joblib`
- `requests`

**Backend `requirements.txt`:**
- `fastapi==0.104.1`
- `uvicorn[standard]==0.24.0`
- `python-multipart==0.0.6`
- `pandas==2.1.3`
- `scikit-learn==1.3.2`
- `numpy==1.25.2`
- `matplotlib==3.8.2`
- `reportlab==4.0.7`
- `python-dateutil==2.8.2`
- `pydantic==2.5.0`
- `aiofiles==23.2.1`

## 2. Model Compatibility and Persistence

### Problem: Version Mismatch
- Scikit-learn models saved as `.pkl` files are tightly coupled with the version of scikit-learn used during training. Loading a model trained with an older version (e.g., 0.22) into a newer version (e.g., 1.0+) can lead to warnings or errors.

### Solution: Retrain or Convert to ONNX
- **Retrain the Model**: If feasible, retrain the model using the current stable version of scikit-learn (e.g., 1.3+). This ensures full compatibility and leverages the latest improvements.
- **Convert to ONNX Format**: For long-term compatibility and deployment across different frameworks, consider converting the model to ONNX format using `skl2onnx`. This provides a standardized, interoperable format.

### Actual Version Requirements
- **Scikit-learn**: Version 1.3+ is recommended for stability and compatibility with modern Python environments. Ensure your deployment environment matches the training environment's scikit-learn version.

## 3. Input Requirements

### Required Input Data Format
- The API expects a CSV file containing customer data. Each row represents a unique customer, and columns represent various attributes.

### Mandatory Columns
- `customer_id`: Unique identifier for each customer.
- `signup_date`: Date of customer signup (YYYY-MM-DD).
- `last_activity_date`: Date of last customer activity (YYYY-MM-DD).
- `subscription_type`: Type of subscription (e.g., 'basic', 'premium', 'enterprise').
- `monthly_charges`: Monthly charges for the subscription.
- `data_usage_gb`: Gigabytes of data used per month.
- `call_minutes`: Minutes of calls made per month.
- `sms_count`: Number of SMS sent per month.
- `customer_service_interactions`: Number of interactions with customer service.
- `contract_renewal_date`: Date of contract renewal (YYYY-MM-DD).
- `payment_method`: Customer's payment method (e.g., 'credit_card', 'paypal', 'bank_transfer').
- `churn_risk_score`: (Optional) Existing churn risk score if available.

### Engineered Features
- **`engagement_score`**: A composite score reflecting customer activity and interaction, calculated from `data_usage_gb`, `call_minutes`, and `sms_count`. Higher scores indicate greater engagement.
- **`satisfaction_trend`**: A metric indicating the trend in customer satisfaction over time, derived from `customer_service_interactions` and other behavioral patterns. A decreasing trend might signal dissatisfaction.

### Data Volume and Frequency
- **Volume**: Up to 5MB per CSV file.
- **Frequency**: Data can be uploaded as needed, typically daily or weekly for churn prediction updates.


## 4. Execution Platform

### Supported Platforms
The model is integrated into a FastAPI application, which can be deployed on:
- **Local Machine**: Running the application directly on your development environment.
- **Cloud Service**: Deployable on various cloud platforms (e.g., AWS, Google Cloud, Azure) that support Python applications and Docker containers.

### Installation Instructions
1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd <repository_directory>
   ```
2. **Create a virtual environment (recommended)**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate # On Windows
   # source .venv/bin/activate # On Linux/macOS
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install -r backend/requirements.txt
   ```
4. **Run the FastAPI application**:
   ```bash
   uvicorn backend.main:app --host 0.0.0.0 --port 8000
   ```
   The API will be accessible at `http://localhost:8000`.

## 5. Model Execution

### How to Initiate Model Runs
The model runs are initiated via the FastAPI application's API endpoints.

- **Prediction Endpoint**:
  - **Method**: `POST`
  - **URL**: `/upload`
  - **Description**: Uploads a CSV file for churn prediction. The processing and prediction happen in the background.
  - **Input**: `file` (UploadFile - CSV file)
  - **Output**: `file_id` to track the processing status.

- **Report Generation Endpoint**:
  - **Method**: `GET`
  - **URL**: `/download-report/{file_id}`
  - **Description**: Downloads the generated PDF churn report for a given `file_id`.
  - **Output**: PDF report.

- **Get Predictions Endpoint**:
  - **Method**: `GET`
  - **URL**: `/predictions/{file_id}`
  - **Description**: Retrieves the detailed churn prediction results.
  - **Output**: JSON containing prediction results.

### Input File Submission Process
Input CSV files are submitted via the `/upload` API endpoint. The API handles saving the file, initiating background processing, and returning a `file_id` for tracking.

### Expected Runtime and Resource Usage
- **Inference Latency**: Expected to be low (milliseconds) for single predictions.
- **Throughput**: Can handle hundreds to thousands of predictions per second on typical server hardware, depending on batch size and hardware.
- **Resource Utilization**: Primarily CPU-bound during inference, with a low memory footprint per prediction.

## 6. Output Delivery

### Output Format and Structure
- **Prediction Results**: Available via the `/predictions/{file_id}` endpoint as a JSON response. It includes:
  - `user_id`
  - `churn_probability`
  - `risk_level` (High, Medium, Low)
  - `top_reasons` for churn.
- **Reports**: Generated as PDF files and can be downloaded via the `/download-report/{file_id}` endpoint. The report includes:
  - Model Specifications
  - Performance Capabilities (Accuracy, Precision, Recall, F1-Score, AUC-ROC)
  - Operational Limitations
  - Comparative Analysis
  - Deployment Specifications

### Where and How Results Will Be Delivered
- **API Responses**: Prediction results are returned directly as JSON from the `/predictions/{file_id}` endpoint.
- **File Downloads**: PDF reports are delivered as file downloads from the `/download-report/{file_id}` endpoint.
- **Internal Storage**: Processed files, predictions, and reports are temporarily stored in `uploads/` and `reports/` directories on the server.

### Post-processing Steps Required for Interpretation
- **Prediction Results**: The JSON output provides `churn_probability`, `risk_level`, and `top_reasons`, which can be directly used for decision-making.
- **PDF Reports**: The generated PDF reports offer a comprehensive overview and analysis, suitable for stakeholders and technical review.

## 6. Troubleshooting

### Common Error Messages and Solutions
- **`400 Bad Request - Invalid file format. Please upload a CSV file.`**:
  - **Cause**: Attempted to upload a file that is not a CSV.
  - **Solution**: Ensure the uploaded file has a `.csv` extension and is a valid CSV format.
- **`400 Bad Request - File size exceeds the 5MB limit.`**:
  - **Cause**: The uploaded CSV file is larger than 5MB.
  - **Solution**: Reduce the size of the input CSV file.
- **`400 Bad Request - Invalid CSV data: Error: Missing required column: [column_name]`**:
  - **Cause**: The input CSV is missing one or more mandatory columns (e.g., `user_id`, `signup_date`).
  - **Solution**: Verify that your CSV file contains all the required columns as specified in Section 2.
- **`500 Internal Server Error - CSV processing failed: [error_details]`**:
  - **Cause**: An unexpected error occurred during the CSV processing step. This could be due to malformed data within the CSV that bypasses initial checks.
  - **Solution**: Review the server logs (`logs/api.log`) for more detailed error messages. Ensure data types and formats within columns are consistent.
- **`500 Internal Server Error - Model prediction failed: [error_details]`**:
  - **Cause**: An error occurred during the model prediction phase. This might indicate an issue with the loaded model or the preprocessed data.
  - **Solution**: Check server logs for specifics. Ensure the model (`churnaizer_saas_model.pkl`) and preprocessor (`one_hot_encoder.pkl`) files are correctly located and not corrupted.
- **`404 Not Found - Report not found or not yet generated.`**:
  - **Cause**: The report for the given `file_id` is either still being generated in the background or the `file_id` is incorrect/expired.
  - **Solution**: Wait for the background processing to complete (check the status if such an endpoint exists, or retry after a short delay). Verify the `file_id`.

### Performance Expectations and Monitoring
- **Problem**: Overly optimistic metrics during development versus real-world performance.
- **Solution**: Set realistic accuracy expectations (75-85%) for production environments. Implement continuous monitoring for model performance, data drift, and outliers.
- **Reality Check**: Real-world performance may vary. For example, a churn prediction model might achieve 82% accuracy in production after accounting for data drift. Regularly retrain models (e.g., quarterly) or as needed when significant performance degradation is observed.

### Contact Information for Technical Support
For further assistance, please contact the development team at [support@example.com] or refer to the project's internal documentation.