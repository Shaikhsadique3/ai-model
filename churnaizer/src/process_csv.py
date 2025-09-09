import pandas as pd
import numpy as np
import hashlib
import json
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(input_file='input.csv'):
    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully read {input_file}. Rows: {len(df)}")
    except FileNotFoundError:
        logging.error(f"Error: {input_file} not found.")
        print(f"Error: {input_file} not found.")
        return
    except Exception as e:
        logging.error(f"Error reading CSV: {e}")
        print(f"Error reading CSV: {e}")
        return

    required_columns = ['user_id', 'signup_date', 'last_login_timestamp', 'billing_status', 'plan_name']
    if not all(col in df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in df.columns]
        logging.error(f"Error: Missing required columns: {', '.join(missing_cols)}")
        print(f"Error: Missing required columns: {', '.join(missing_cols)}")
        return

    # Data Cleaning and Conversion
    df.drop_duplicates(subset=['user_id'], inplace=True)
    logging.info(f"Dropped duplicate user_ids. Remaining rows: {len(df)}")

    # Parse dates
    for col in ['signup_date', 'last_login_timestamp']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
        if df[col].isnull().any():
            logging.warning(f"Warning: Missing or invalid dates in {col}. Rows with NaT: {df[col].isnull().sum()}")

    # Normalize billing_status
    df['billing_status'] = df['billing_status'].apply(lambda x: 0 if str(x).lower() == 'active' else 1)

    # Fill missing optional values with defaults (example, extend as needed)
    df['monthly_revenue'] = df['monthly_revenue'].fillna(0)
    df['support_tickets_opened'] = df['support_tickets_opened'].fillna(0)
    df['email_opens_last30days'] = df['email_opens_last30days'].fillna(0)
    df['number_of_logins_last30days'] = df['number_of_logins_last30days'].fillna(0)
    df['time_to_first_value'] = df['time_to_first_value'].fillna(0)
    df['last_payment_status'] = df['last_payment_status'].fillna('unknown')

    # Mask user_id
    df['user_id_masked'] = df['user_id'].apply(lambda x: hashlib.sha256(str(x).encode()).hexdigest()[:8])

    # Derive new fields
    current_date = datetime.now()
    df['days_since_signup'] = (current_date - df['signup_date']).dt.days
    df['last_login_days_ago'] = (current_date - df['last_login_timestamp']).dt.days

    # Placeholder for days_until_renewal and billing_issue_count (requires more complex logic)
    df['days_until_renewal'] = np.random.randint(1, 365, size=len(df)) # Example placeholder
    df['billing_issue_count'] = np.random.randint(0, 5, size=len(df)) # Example placeholder

    # Output processed_data.csv
    output_columns = [
        'user_id_masked', 'plan_name', 'days_since_signup', 'days_until_renewal',
        'last_login_days_ago', 'number_of_logins_last30days', 'time_to_first_value',
        'support_tickets_opened', 'email_opens_last30days', 'billing_issue_count',
        'monthly_revenue', 'last_payment_status', 'risk_level' # risk_level is TBD placeholder
    ]
    df['risk_level'] = 'TBD' # Placeholder
    df[output_columns].to_csv('processed_data.csv', index=False)
    logging.info(f"Processed data saved to processed_data.csv. Rows: {len(df)}")

    # Output stats_summary.json
    total_customers = len(df)
    active_customers = df[df['billing_status'] == 0].shape[0]
    failed_billing_customers = df[df['billing_status'] == 1].shape[0]
    average_revenue = df['monthly_revenue'].mean()
    avg_login_gap = df['last_login_days_ago'].mean() # Simplified, could be more complex
    churn_rate = 0 # Placeholder

    stats_summary = {
        'total_customers': total_customers,
        'active_customers': active_customers,
        'failed_billing_customers': failed_billing_customers,
        'average_revenue': average_revenue,
        'avg_login_gap': avg_login_gap,
        'churn_rate': churn_rate
    }

    with open('stats_summary.json', 'w') as f:
        json.dump(stats_summary, f, indent=4)
    logging.info("Stats summary saved to stats_summary.json.")

    print("CSV processing complete. Check processed_data.csv, stats_summary.json, and log.txt")

if __name__ == '__main__':
    process_csv()