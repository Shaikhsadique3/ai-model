import pandas as pd
import hashlib
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(input_file: str):
    """Process CSV file and return processed DataFrame, stats summary, and warnings."""
    warnings = []
    processed_rows = 0

    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully read {input_file}")
    except FileNotFoundError:
        error_msg = f"Error: CSV file not found at {input_file}"
        logging.error(error_msg)
        warnings.append(error_msg)
        return None, None, warnings
    except Exception as e:
        error_msg = f"Error reading CSV file: {e}"
        logging.error(error_msg)
        warnings.append(error_msg)
        return None, None, warnings

    required_columns = ['user_id', 'signup_date', 'last_login_timestamp', 'billing_status', 'plan_name']
    for col in required_columns:
        if col not in df.columns:
            error_msg = f"Error: Missing required column: {col}"
            logging.error(error_msg)
            warnings.append(error_msg)
            return None, None, warnings

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

    # Fill missing optional values with defaults
    numeric_cols = ['support_tickets_opened', 'email_opens_last30days', 'monthly_revenue']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        else:
            df[col] = 0

    # Add last_payment_status if not present
    if 'last_payment_status' not in df.columns:
        df['last_payment_status'] = 'success'

    # Drop duplicate user_ids
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask user_id for reporting
    df['user_id_masked'] = df['user_id'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8])

    # Derive new fields
    df['days_since_signup'] = (datetime.now() - df['signup_date']).dt.days
    df['last_login_days_ago'] = (datetime.now() - df['last_login_timestamp']).dt.days

    # Add placeholder fields for prediction model
    df['days_until_renewal'] = 30  # Placeholder
    df['number_of_logins_last30days'] = df.get('number_of_logins_last30days', 10)  # Default value
    df['time_to_first_value'] = 7  # Placeholder
    df['billing_issue_count'] = df['billing_status']  # Use billing status as proxy
    df['active_features_used'] = 3  # Default value

    # Rename columns to match model expectations
    df['subscription_plan'] = df['plan_name']

    # Ensure all numeric columns are properly typed
    numeric_columns = ['days_since_signup', 'last_login_days_ago', 'monthly_revenue', 
                      'support_tickets_opened', 'email_opens_last30days', 'billing_issue_count',
                      'number_of_logins_last30days', 'active_features_used']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Generate stats summary
    total_customers = len(df)
    active_customers = df[df['billing_status'] == 0].shape[0] if 'billing_status' in df.columns else 0
    failed_billing_customers = df[df['billing_status'] == 1].shape[0] if 'billing_status' in df.columns else 0
    average_revenue = df['monthly_revenue'].mean() if 'monthly_revenue' in df.columns else 0
    avg_login_gap = df['last_login_days_ago'].mean() if 'last_login_days_ago' in df.columns else 0

    stats_summary = {
        "total_customers": total_customers,
        "active_customers": active_customers,
        "failed_billing_customers": failed_billing_customers,
        "average_revenue": float(average_revenue),
        "avg_login_gap": float(avg_login_gap)
    }

    # Define output columns
    output_columns = [
        'user_id_masked', 'subscription_plan', 'days_since_signup', 'days_until_renewal',
        'last_login_days_ago', 'number_of_logins_last30days', 'time_to_first_value',
        'support_tickets_opened', 'email_opens_last30days', 'billing_issue_count',
        'monthly_revenue', 'last_payment_status'
    ]

    # Ensure all output columns exist
    for col in output_columns:
        if col not in df.columns:
            df[col] = 0

    processed_rows = len(df)
    logging.info(f"Processing complete. Processed {processed_rows} rows.")
    
    if warnings:
        logging.info("Warnings during processing:")
        for warning in warnings:
            logging.info(warning)

    return df[output_columns], stats_summary, warnings