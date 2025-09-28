# AI Churn Prediction Model

This repository contains a machine learning model for predicting customer churn and a FastAPI application to serve these predictions.

## Project Structure

```
.
├── backend/
│   ├── main.py                     # FastAPI application entry point
│   ├── model/                      # Churn prediction model service
│   │   └── predict_model.py
│   ├── processing/                 # CSV processing utilities for the API
│   │   └── process_csv.py
│   ├── report/                     # Report generation utilities for the API
│   │   └── report_generator.py
│   └── requirements.txt            # Backend specific dependencies
├── config/
│   └── config.ini                  # Configuration file
├── data/                           # Directory for raw and processed data
├── model/                          # Core model training and evaluation scripts
│   ├── predict_model.py
│   └── train_model.py
├── models/                         # Directory to store trained model artifacts
├── processing/                     # Core data processing and feature engineering scripts
│   ├── evaluate.py
│   ├── feature_engineering.py
│   ├── generate_dummy_data.py
│   ├── preprocessing.py
│   ├── recommendation_engine.py
│   ├── reporting_engine.py
│   └── root_cause_analysis.py
├── report/                         # Core report generation and analysis scripts
│   ├── report_generator.py
│   └── technical_report.md
├── requirements.txt                # Project-wide Python dependencies
├── tests/                          # Unit and integration tests
│   └── test_model.py
├── .gitignore                      # Git ignore file
├── Makefile                        # Makefile for common commands
├── README.md                       # Project README file
└── churnaizer_colab.py             # Colab notebook for Churnaizer
```

## Setup and Installation

This project uses Python 3.9+. It is recommended to use a virtual environment.

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd ai-churn-prediction-model
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    ```
    -   **On Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    -   **On macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

3.  **Install dependencies:**
    Install core project dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    Install backend-specific dependencies:
    ```bash
    pip install -r backend/requirements.txt
    ```

## Usage

### 1. FastAPI Churn Audit Service

The primary way to interact with the churn prediction model is through the FastAPI service. Ensure the backend dependencies are installed (`pip install -r backend/requirements.txt`).

To run the FastAPI application:
```bash
python backend/main.py
```

The API will be available at `http://127.0.0.1:8000`.

#### API Documentation

Once the FastAPI service is running, you can access the interactive API documentation (Swagger UI) at:

*   `http://127.0.0.1:8000/docs`

Or the alternative ReDoc documentation at:

*   `http://127.0.0.1:8000/redoc`

These interfaces provide detailed information about each endpoint, including expected parameters, response formats, and the ability to test the API directly.

#### Key Endpoints:

*   **`/upload` (POST):** Upload a CSV file for churn prediction. Returns a `file_id` to track processing.
*   **`/status/{file_id}` (GET):** Get the current processing status of an uploaded file.
*   **`/predictions/{file_id}` (GET):** Retrieve detailed churn prediction results.
*   **`/download-report/{file_id}` (GET):** Download the generated PDF churn report.
*   **`/sample-csv` (GET):** Download a sample CSV file template.

### 2. Model Training (Standalone)

To train the churn prediction model independently:

```bash
python model/train_model.py
```
This script will train the model using the data in the `data/` directory and save the trained model artifacts to the `models/` directory.

### 3. Data Processing (Standalone)

To generate dummy data or process your own CSV files outside the API:

*   **Generate Dummy Data:**
    ```bash
    python processing/generate_dummy_data.py
    ```

*   **Process CSV Data (Note: This is primarily used internally by the API now, but can be run standalone):**
    ```bash
    python processing/process_csv.py <path_to_your_raw_csv.csv> <path_for_processed_output.csv>
    ```

### 4. Model Evaluation (Standalone)

To evaluate the trained model:

```bash
python processing/evaluate.py
```
This script will evaluate the model's performance and output metrics.

### 5. Running Tests

To run the tests for the project:

```bash
python -m pytest tests/
```

## Best Practices

*   **Virtual Environments:** Always use a virtual environment to manage project dependencies.
*   **Dependency Management:** Keep `requirements.txt` and `backend/requirements.txt` up-to-date with all project dependencies.
*   **Version Control:** Use `.gitignore` to exclude temporary files, logs, large datasets, and sensitive information from version control.
*   **Code Clarity:** Write clear, concise, and well-commented code. Include docstrings for all functions, classes, and modules.
*   **Coding Standards:** Adhere to consistent coding standards (e.g., PEP 8 for Python) throughout the codebase.
*   **Modularity:** Design components with modularity and separation of concerns in mind to enhance scalability and maintainability.
*   **Testing:** Write unit and integration tests for critical components to ensure reliability and prevent regressions.
*   **Security:** Follow security best practices for data handling, storage, and API authentication.
*   **Documentation:** Ensure all documentation (inline comments, docstrings, README) is synchronized with code changes and provides comprehensive explanations.