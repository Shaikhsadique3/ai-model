import pandas as pd
import hashlib
import json
import logging
from datetime import datetime, timedelta
from model.predict_model import ChurnPredictorService

class DataProcessingError(Exception):
    """Custom exception for data processing errors."""
    pass

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(level)s - %(message)s')

def process_csv(input_file: str): # Removed output_file parameter
    warnings = []
    processed_rows = 0

    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully read {input_file}")
    except FileNotFoundError:
        error_msg = f"Error: input.csv not found at {input_file}"
        logging.error(error_msg)
        raise DataProcessingError(error_msg)
    except Exception as e:
        error_msg = f"Error reading CSV file: {e}"
        logging.error(error_msg)
        raise DataProcessingError(error_msg)

    required_columns = ['user_id', 'signup_date', 'last_login_timestamp', 'billing_status', 'plan_name']
    for col in required_columns:
        if col not in df.columns:
            error_msg = f"Error: Missing required column: {col}"
            logging.error(error_msg)
            raise DataProcessingError(error_msg)

    # Keep the original user_id for prediction
    df['original_user_id'] = df['user_id']

    # Data Cleaning and Conversion
    # Parse dates
    for date_col in ['signup_date', 'last_login_timestamp']:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        invalid_dates = df[df[date_col].isna()]
        if not invalid_dates.empty:
            for index, row in invalid_dates.iterrows():
                warnings.append(f"Skipped row {index} due to invalid date in {date_col}: {row[date_col]}")
                logging.warning(f"Skipped row {index} due to invalid date in {date_col}: {row[date_col]}")
            df = df.dropna(subset=[date_col])

    # Normalize billing_status
    df['billing_status'] = df['billing_status'].apply(lambda x: 0 if x == 'active' else 1 if x == 'failed' else x)

    # Fill missing optional values with defaults (assuming 0 for numeric, empty string for others)
    numeric_cols = ['support_tickets_opened', 'email_opens_last30days', 'monthly_revenue', 'last_payment_status']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0 # Add column if it doesn't exist

    # Drop duplicate user_ids
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask user_id for reporting
    df['user_id_masked'] = df['user_id'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8])

    # Derive new fields
    df['days_since_signup'] = (datetime.now() - df['signup_date']).dt.days
    df['last_login_days_ago'] = (datetime.now() - df['last_login_timestamp']).dt.days

    # Placeholder for days_until_renewal (requires more complex logic based on plan)
    df['days_until_renewal'] = 0

    # Placeholder for number_of_logins_last30days (requires login history data)
    df['number_of_logins_last30days'] = 0

    # Placeholder for time_to_first_value (requires more complex event data)
    df['time_to_first_value'] = 0

    # Placeholder for billing_issue_count (requires more detailed billing history)
    df['billing_issue_count'] = 0

    # Rename original_user_id back to user_id for the predictor
    df.rename(columns={'original_user_id': 'user_id'}, inplace=True)

    # Predict churn
    try:
        predictor = ChurnPredictorService()
        predictions_df = predictor.predict_batch(df)
        df = pd.merge(df, predictions_df, on='user_id', how='left')
        # Rename columns to match requirements
        df.rename(columns={'churn_probability': 'churn_score', 'top_reasons': 'reason'}, inplace=True)
        df['reason'] = df['reason'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

    except Exception as e:
        logging.error(f"Churn prediction failed: {e}")
        # Add placeholder columns if prediction fails
        df['churn_score'] = 0.0
        df['risk_level'] = 'N/A'
        df['reason'] = 'Prediction failed'

    # Ensure all output columns exist, even if placeholders
    output_columns = [
        'user_id_masked', 'plan_name', 'days_since_signup', 'days_until_renewal',
        'last_login_days_ago', 'number_of_logins_last30days', 'time_to_first_value',
        'support_tickets_opened', 'email_opens_last30days', 'billing_issue_count',
        'monthly_revenue', 'last_payment_status', 'churn_score', 'risk_level', 'reason'
    ]
    for col in output_columns:
        if col not in df.columns:
            df[col] = 0 # Default to 0 or appropriate placeholder

    # Removed: df[output_columns].to_csv(output_file, index=False)
    # Removed: logging.info(f"Processed data saved to {output_file}")

    # Generate stats_summary.json
    total_customers = len(df)
    active_customers = df[df['billing_status'] == 0].shape[0] if 'billing_status' in df.columns else 0
    failed_billing_customers = df[df['billing_status'] == 1].shape[0] if 'billing_status' in df.columns else 0
    average_revenue = df['monthly_revenue'].mean() if 'monthly_revenue' in df.columns else 0
    avg_login_gap = df['last_login_days_ago'].mean() if 'last_login_days_ago' in df.columns else 0
    
    # Churn distribution
    if 'risk_level' in df.columns:
        high_risk_percent = (df['risk_level'] == 'High').sum() / total_customers * 100 if total_customers > 0 else 0
        medium_risk_percent = (df['risk_level'] == 'Medium').sum() / total_customers * 100 if total_customers > 0 else 0
        low_risk_percent = (df['risk_level'] == 'Low').sum() / total_customers * 100 if total_customers > 0 else 0
        average_churn_score = df['churn_score'].mean() if 'churn_score' in df.columns else 0
    else:
        high_risk_percent = medium_risk_percent = low_risk_percent = average_churn_score = 0

    stats_summary = {
        "total_customers": total_customers,
        "active_customers": active_customers,
        "failed_billing_customers": failed_billing_customers,
        "average_revenue": average_revenue,
        "avg_login_gap": avg_login_gap,
        "churn_distribution": {
            "high_risk_percent": high_risk_percent,
            "medium_risk_percent": medium_risk_percent,
            "low_risk_percent": low_risk_percent,
            "average_churn_score": average_churn_score
        }
    }
    # Removed: with open('stats_summary.json', 'w') as f:
    # Removed:     json.dump(stats_summary, f, indent=4)
    # Removed: logging.info("Stats summary saved to stats_summary.json")

    # Log processing summary
    processed_rows = len(df)
    logging.info(f"Processing complete. Processed {processed_rows} rows.")
    if warnings:
        logging.info("Warnings during processing:")
        for warning in warnings:
            logging.info(warning)

    # Removed: Print first 10 rows of processed DataFrame
    # Removed: print(f"\nFirst 10 rows of {output_file}:")
    # Removed: print(df[output_columns].head(10).to_string())

    return df[output_columns], stats_summary, warnings # Return the processed DataFrame, stats_summary, and warnings

# Removed: if __name__ == "__main__":
# Removed:     process_csv()