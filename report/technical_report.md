
# AI Model Technical Report

**Version**: 1.0
**Date**: 2025-09-27 21:41:01
**Author**: AI Assistant


### 1. Model Specifications

#### 1.1 Architecture
- **Model Type**: Gradient Boosting Classifier (e.g., LightGBM, XGBoost, CatBoost) or similar tree-based ensemble model.
- **Framework**: Scikit-learn, LightGBM, XGBoost, or CatBoost.

#### 1.2 Training Methodology
- **Approach**: Supervised Learning (Binary Classification).
- **Objective**: Predict customer churn (binary outcome: churned/not churned).
- **Hyperparameters**: Tuned using techniques like GridSearchCV or RandomizedSearchCV (details would be in `model/train_model.py`).
- **Optimization**: Gradient descent-based optimization (specifics depend on the chosen model).

#### 1.3 Input Requirements
- **Data Format**: CSV file containing preprocessed numerical features.
- **Feature Specifications**: Expects a fixed set of numerical features (e.g., `monthly_revenue`, `days_since_signup`, `last_login_days_ago`, `active_features_used`, `NPS_score`, `engagement_score`, `satisfaction_trend`, `plan_type` (one-hot encoded), `payment_status` (one-hot encoded)).
- **Preprocessing Steps**: 
    - Handling missing values (e.g., imputation).
    - Encoding categorical variables (e.g., One-Hot Encoding).
    - Feature scaling (e.g., StandardScaler) if required by the model.
    - Feature engineering (e.g., creating `last_login_days_ago` from `last_login_date`).
    (Details in `processing/preprocessing.py` and `processing/feature_engineering.py`)

#### 1.4 Output Specifications
- **Output Format**: Probability score (float between 0 and 1) representing the likelihood of churn, and a binary prediction (0 or 1).
- **Interpretation**: Higher probability scores indicate a higher likelihood of churn. A threshold (e.g., 0.5) is used to classify as churned (1) or not churned (0).
- **Confidence Metrics**: The probability score itself can be interpreted as a confidence level. Further calibration might be applied for better confidence estimates.


### 2. Performance Capabilities

#### 2.1 Supported Prediction Types
- **Task**: Binary Classification (Churn Prediction).
- **Output**: Predicts whether a customer will churn (1) or not churn (0).

#### 2.2 Performance Metrics (Validation Results)
- **Accuracy**: 0.9830
- **Precision**: 0.9538
- **Recall**: 0.9812
- **F1-Score**: 0.9673
- **AUC-ROC**: 0.9992

#### 2.3 Operational Limitations
- **Data Drift**: Performance may degrade if the distribution of input data changes significantly over time.
- **Feature Importance Stability**: Feature importance can shift, requiring periodic model retraining.
- **Interpretability**: While SHAP values provide local interpretability, the underlying ensemble model can be complex.
- **Bias**: Potential for bias if training data is not representative of the target population.
- **Edge Cases**: May perform poorly on customers with highly unusual behavior patterns not seen during training.


### 3. Comparative Analysis

#### 3.1 Industry Benchmarking
- **Current Status**: No direct industry benchmarks are available within the scope of this project. Typically, this would involve comparing model performance against publicly reported benchmarks for similar churn prediction tasks in the SaaS industry.

#### 3.2 Alternative Model Comparison
- **Current Status**: This report focuses on the selected model. A comprehensive comparison would involve evaluating other models (e.g., Logistic Regression, SVM, Neural Networks) on the same dataset and metrics.

#### 3.3 Evaluation Framework
- **Metrics**: Accuracy, Precision, Recall, F1-Score, AUC-ROC.
- **Methodology**: Cross-validation on training data, final evaluation on a held-out test set.


### 4. Deployment Specifications

#### 4.1 System Requirements
- **Hardware**: 
    - **CPU**: Multi-core processor recommended for faster inference.
    - **RAM**: Minimum 4GB, 8GB+ recommended for handling larger datasets or concurrent requests.
    - **Disk Space**: Minimal for model artifacts and code (e.g., <100MB).
- **Software**: 
    - Python 3.8+
    - Required libraries: `pandas`, `scikit-learn`, `joblib`, `shap`, `numpy`, `lightgbm` (or other model-specific library).
    - Operating System: Platform-agnostic (Linux, Windows, macOS).

#### 4.2 Runtime Performance
- **Inference Latency**: Expected to be low (milliseconds) for single predictions, given the tree-based nature of the model.
- **Throughput**: Can handle hundreds to thousands of predictions per second on typical server hardware, depending on batch size and hardware.
- **Resource Utilization**: 
    - **CPU**: Primarily CPU-bound during inference.
    - **Memory**: Low memory footprint per prediction.

#### 4.3 Scalability Analysis
- **Horizontal Scaling**: Easily scalable by deploying multiple instances of the prediction service behind a load balancer.
- **Vertical Scaling**: Performance can be improved with more powerful CPU and RAM resources on a single instance.
- **Limitations**: 
    - **Data Volume**: Extremely large input data batches might increase latency.
    - **Model Size**: Very complex models with many trees could increase memory usage and inference time.
