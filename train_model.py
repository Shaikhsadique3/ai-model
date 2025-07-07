import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib
import warnings

warnings.filterwarnings("ignore")

class ChurnPredictor:
    def __init__(self, model_path="churnaizer_saas_model.pkl"):
        self.model_path = model_path
        self.model = None
        self.preprocessor = None

    def load_data(self, file_path):
        df = pd.read_csv(file_path)
        # Handle missing values
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col].fillna(df[col].mode()[0], inplace=True)
            else:
                df[col].fillna(df[col].mean(), inplace=True)
        return df

    def preprocess_data(self, df):
        # Convert 'days_since_signup' to numeric
        df['days_since_signup'] = pd.to_numeric(df['days_since_signup'], errors='coerce')
        df.dropna(subset=['days_since_signup'], inplace=True)

        # Convert 'churn' to integer
        df['churn'] = df['churn'].astype(int)

        # Convert relevant columns to numeric
        for col in ['monthly_revenue', 'number_of_logins_last30days', 'active_features_used', 'support_tickets_opened']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['monthly_revenue', 'number_of_logins_last30days', 'active_features_used', 'support_tickets_opened'], inplace=True)

        # One-hot encode categorical features
        categorical_features = ['subscription_plan', 'last_payment_status']
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
            ], remainder='passthrough'
        )
        
        X = df.drop("churn", axis=1)
        y = df["churn"]

        X_processed = preprocessor.fit_transform(X)
        self.preprocessor = preprocessor # Save preprocessor for later use
        return X_processed, y, X.columns # Return original columns for feature importance

    def train_model(self, X, y):
        # Balance classes using SMOTE
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        # XGBoost Classifier with GridSearchCV
        xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
        grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        return X_test, y_test

    def evaluate_model(self, X_test, y_test, original_columns):
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        print(f"\n✨ Model Accuracy: {accuracy:.2f}")
        print("\n📊 Classification Report:")
        print(report)
        print("\n🧱 Confusion Matrix:")
        print(cm)

        # Feature Importance
        feature_importances = self.model.feature_importances_
        # Get feature names after one-hot encoding
        ohe_feature_names = self.preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features)
        all_feature_names = np.concatenate((ohe_feature_names, original_columns.drop(categorical_features).values))

        importance_df = pd.DataFrame({
            'Feature': all_feature_names,
            'Importance': feature_importances
        }).sort_values(by='Importance', ascending=False)
        importance_df['Percentage'] = importance_df['Importance'] / importance_df['Importance'].sum() * 100
        print("\n⭐ Top Features:")
        print(importance_df)

    def save_model(self):
        joblib.dump(self.model, "churnaizer_model_v5.pkl")
        print(f"\n📦 Model saved as churnaizer_model_v5.pkl")

if __name__ == "__main__":
    predictor = ChurnPredictor()
    df = predictor.load_data("enhanced_saas_churn_data.csv")
    X_processed, y, original_columns = predictor.preprocess_data(df)
    X_test, y_test = predictor.train_model(X_processed, y)
    predictor.evaluate_model(X_test, y_test, original_columns)
    predictor.save_model()