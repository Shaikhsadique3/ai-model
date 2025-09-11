# AI Churn Prediction Model

This repository contains a machine learning model for predicting customer churn.

## Project Structure

```
/model
   train_model.py
   predict_model.py
/processing
   process_csv.py
   feature_engineering.py
   evaluate.py
   generate_dummy_data.py
/report
   report_generator.py
/tests
   test_model.py
requirements.txt
README.md
```

## Setup and Installation

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
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Processing

You can generate dummy data or process your own CSV files.

-   **Generate Dummy Data:**
    ```bash
    python processing/generate_dummy_data.py
    ```

-   **Process CSV Data:**
    Use `processing/process_csv.py` to preprocess your raw data.
    ```bash
    python processing/process_csv.py <path_to_your_raw_csv.csv> <path_for_processed_output.csv>
    ```

### 2. Model Training

Train the churn prediction model using `model/train_model.py`.

```bash
python model/train_model.py
```
This will train the model and save it to the `models/` directory.

### 3. Model Prediction

Predict churn on new, processed data using `model/predict_model.py`.

```bash
python model/predict_model.py <path_to_processed_data.csv>
```
This script will output predictions based on the provided processed data.

### 4. Report Generation

Generate a performance report for the model using `report/report_generator.py`.

```bash
python report/report_generator.py
```
This will generate a report based on the model's performance.

### 5. Running Tests

To run the tests for the model, use:

```bash
python -m pytest tests/test_model.py
```

## Best Practices

-   Ensure your `requirements.txt` is up-to-date with all project dependencies.
-   Use `.gitignore` to exclude temporary files, logs, and large datasets from version control.
-   Comment your code for clarity and maintainability.