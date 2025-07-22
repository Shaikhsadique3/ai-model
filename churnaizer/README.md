# Churnaizer: SaaS Customer Churn Prediction

This project provides a robust and scalable solution for predicting customer churn in a SaaS environment. It leverages machine learning to identify customers at risk of churning, allowing businesses to take proactive measures to retain them.

## Features

- **Data Preprocessing**: Handles missing values, converts data types, and performs one-hot encoding for categorical features.
- **Feature Selection**: Utilizes RandomForestClassifier and SelectFromModel to identify and select the most relevant features for churn prediction.
- **Model Training**: Trains an XGBoost classifier with GridSearchCV for hyperparameter tuning and SMOTE for handling imbalanced datasets.
- **Model Evaluation**: Generates comprehensive evaluation reports including accuracy, F1 score, confusion matrix, and ROC-AUC score.
- **Model Analysis**: Provides insights into model performance, feature importances, and overfitting risks.
- **API Endpoint**: Exposes a `/predict` endpoint using Flask for real-time churn predictions.
- **Logging**: Implements structured logging for better monitoring and debugging.
- **Configuration Management**: Uses a `config.json` file for easy management of application settings.

## Project Structure

```
churnaizer/
├── config/
│   └── config.json             # Configuration file for paths, features, etc.
├── data/
│   ├── generate_dummy_data.py  # Script to generate dummy dataset
│   └── enhanced_saas_churn_data.csv # Generated dummy dataset
├── logs/
│   └── churnaizer.log          # Application logs
├── models/
│   ├── churnaizer_saas_model.pkl # Trained churn prediction model
│   └── one_hot_encoder.pkl     # Trained preprocessor (OneHotEncoder)
├── src/
│   ├── __init__.py
│   ├── evaluate.py             # Model evaluation functions
│   ├── feature_selector.py     # Feature selection logic
│   ├── preprocessing.py        # Data preprocessing functions
│   └── train.py                # Model training pipeline
├── main.py                     # Flask application for prediction API
├── model_analyzer.py           # Script for in-depth model analysis
├── requirements.txt            # Python dependencies
└── README.md                   # Project README
```

## Setup and Installation

1.  **Clone the repository**:

    ```bash
    git clone <repository_url>
    cd churnaizer
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    python -m venv venv
    .\venv\Scripts\activate  # On Windows
    # source venv/bin/activate  # On macOS/Linux
    ```

3.  **Install dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

4.  **Generate Dummy Data** (optional, for initial setup):

    ```bash
    python data/generate_dummy_data.py
    ```

    This will create `enhanced_saas_churn_data.csv` in the `data/` directory.

## Usage

### 1. Train the Model

To train the churn prediction model, run the `train.py` script:

```bash
python src/train.py
```

This will save the trained model (`churnaizer_saas_model.pkl`) and the preprocessor (`one_hot_encoder.pkl`) in the `models/` directory.

### 2. Analyze the Model

To get a detailed analysis of the trained model, run the `model_analyzer.py` script:

```bash
python model_analyzer.py
```

This will print a summary report to the console and log it to `logs/churnaizer.log`.

### 3. Evaluate the Model

To evaluate the model's performance on a test set, run the `evaluate.py` script:

```bash
python src/evaluate.py
```

This will generate an `evaluation_report.txt` in the root `churnaizer` directory.

### 4. Run the Prediction API

To start the Flask API for real-time predictions, run `main.py`:

```bash
python main.py
```

The API will be available at `http://127.0.0.1:5000/`.

#### Prediction Endpoint

-   **POST** `/predict`
-   **Content-Type**: `application/json`
-   **Request Body Example**:

    ```json
    {
        "days_since_signup": 500,
        "monthly_revenue": 120.50,
        "number_of_logins_last30days": 25,
        "active_features_used": 3,
        "support_tickets_opened": 1,
        "last_login_days_ago": 7,
        "email_opens_last30days": 15,
        "billing_issue_count": 0,
        "last_payment_status": "Success",
        "subscription_plan": "Basic"
    }
    ```

-   **Response Example**:

    ```json
    {
        "churn_prediction": 0,
        "confidence": 0.9876
    }
    ```

## Configuration

Adjust settings in `churnaizer/config/config.json`:

```json
{
  "dataset_path": "data/enhanced_saas_churn_data.csv",
  "model_path": "models/churnaizer_saas_model.pkl",
  "preprocessor_path": "models/one_hot_encoder.pkl",
  "categorical_features": ["subscription_plan", "last_payment_status"],
  "target_column": "churn"
}
```

## Logging

Logs are configured to be written to `logs/churnaizer.log` and also output to the console. The logging level is set to `INFO` by default.