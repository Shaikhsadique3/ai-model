# Churn Prediction Streamlit App

This repository contains a Streamlit application for churn prediction using two machine learning models: XGBoost and Random Forest.

## Table of Contents

- [Features](#features)
- [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
- [File Structure](#file-structure)
- [Maintenance](#maintenance)

## Features

- Upload CSV files for churn prediction.
- Utilizes pre-trained XGBoost and Random Forest models.
- Displays churn predictions and probabilities for each entry in the uploaded dataset.

## Setup

Follow these steps to set up and run the application locally:

1.  **Clone the repository (if you haven't already):**

    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    -   **On Windows:**

        ```bash
        .\venv\Scripts\activate
        ```

    -   **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

2.  **Open your web browser** and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Upload a CSV file** containing your customer data. The application will then display the churn predictions and probabilities.

## Models

This application uses two pre-trained models:

-   `churnaizer_saas_model.pkl`: An XGBoost model for churn prediction.
-   `churnaizer_model.pkl`: A Random Forest model for churn prediction.

**Important Note on Model Compatibility:**

There is a known issue with loading the pre-trained models (`.pkl` files) due to potential version incompatibilities with `joblib` and `scikit-learn`. If you encounter a `KeyError: 239` or similar issues during model loading, it indicates that the models were likely saved with a different version of these libraries than what is currently installed. 

To resolve this, you will need to:

1.  **Retrain the models** using the versions of `joblib` and `scikit-learn` specified in `requirements.txt`.
2.  **Save the retrained models** in the `.pkl` format, ensuring they are compatible with the current environment.

These models should be present in the same directory as `app.py`.

## File Structure

```
. \
├── README.md
├── app.py
├── churnaizer_model.pkl
├── churnaizer_saas_model.pkl
└── requirements.txt
```

## Model Details and Specifications

### Training Process
The models were trained using the `churn_training_data.csv` dataset. The process involved:
1.  **Data Loading**: The CSV data was loaded into a Pandas DataFrame.
2.  **Preprocessing**: Categorical features (`plan_type`, `payment_status`) were one-hot encoded using `pd.get_dummies`.
3.  **Data Splitting**: The dataset was split into training and testing sets with an 80/20 ratio, using `random_state=42` for reproducibility and `stratify=y` to maintain the class distribution in both sets.
4.  **Model Training**: Both XGBoost (`XGBClassifier`) and RandomForest (`RandomForestClassifier`) models were trained on the preprocessed training data.
5.  **Model Evaluation**: The trained models were evaluated on the test set using accuracy, precision, recall, and F1-score.
6.  **Model Saving**: The trained models were saved as `.pkl` files (`churnaizer_model.pkl` for XGBoost and `churnaizer_saas_model.pkl` for RandomForest) in the `model/` directory using `joblib`.

### Performance Metrics
The following performance metrics were observed on the test set:

**XGBoost Model**:
-   **Accuracy**: 0.9153
-   **Precision**: 0.8335
-   **Recall**: 0.8240
-   **F1-Score**: 0.8287

**RandomForest Model**:
-   **Accuracy**: 0.9177
-   **Precision**: 0.8418
-   **Recall**: 0.8242
-   **F1-Score**: 0.8329

### Validation Methodology
A standard 80/20 train-test split was used for model validation. The `random_state` was set to 42 to ensure reproducibility of the split, and `stratify=y` was applied to maintain the proportion of the target variable (`churned`) in both the training and testing sets, which is crucial for imbalanced datasets.

### Input Data Requirements
The application expects a CSV file as input. The columns in this CSV file should match the features used during the training of the churn prediction models. The specific feature names and their data types are crucial for accurate predictions. Based on `churn_training_data.csv`, the expected columns are:
-   `plan_type`: Categorical (e.g., 'Basic', 'Enterprise', 'Pro', 'Free')
-   `monthly_revenue`: Numerical
-   `payment_status`: Categorical (e.g., 'On-time', 'Late', 'Failed')
-   `days_since_signup`: Numerical
-   `last_login_days_ago`: Numerical
-   `logins_last30days`: Numerical
-   `active_features_used`: Numerical
-   `tickets_opened`: Numerical
-   `NPS_score`: Numerical

### Output Data Format
Upon successful prediction, the application will output the original DataFrame with the following additional columns:
- `churn_prediction_xgb`: Binary prediction (0 or 1) from the XGBoost model, indicating whether a customer is predicted to churn.
- `churn_probability_xgb`: Probability of churn from the XGBoost model (a value between 0 and 1).
- `churn_prediction_rf`: Binary prediction (0 or 1) from the RandomForest model, indicating whether a customer is predicted to churn.
- `churn_probability_rf`: Probability of churn from the RandomForest model (a value between 0 and 1).


## Maintenance

-   **Updating Dependencies:** If new libraries are added or existing ones are updated, make sure to update `requirements.txt` accordingly.
-   **Model Retraining:** When new models are trained, replace the `.pkl` files in the root directory with the updated models.
-   **Troubleshooting:** Check the terminal for any error messages if the application is not behaving as expected.


## Deployment on Streamlit Cloud

To deploy this application on Streamlit Cloud, follow these steps:

1.  **Ensure `requirements.txt` is up-to-date:** Make sure your `requirements.txt` file (located in the root directory) lists all the Python libraries your application depends on. This is crucial for Streamlit Cloud to install the correct environment.

2.  **Push your code to a GitHub repository:** Streamlit Cloud deploys directly from GitHub. Ensure your entire project, including `app.py`, `pages/`, `model/`, `requirements.txt`, and any data files, is pushed to a public or private GitHub repository.

3.  **Go to Streamlit Cloud:** Navigate to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.

4.  **Deploy a new app:** Click on the "New app" button.

5.  **Connect your repository:** Select your GitHub repository, the branch you want to deploy from (e.g., `main` or `master`), and set the main file path to `app.py`.

6.  **Advanced settings (if needed):** If your application requires specific Python versions or other environment variables, you can configure these in the "Advanced settings" section.

7.  **Deploy!** Click the "Deploy!" button. Streamlit Cloud will then build your application, install dependencies, and deploy it. This might take a few minutes.

8.  **Monitor deployment:** You can monitor the deployment process and view logs directly on the Streamlit Cloud interface. If there are any issues, the logs will provide details for troubleshooting.