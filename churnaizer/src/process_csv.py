import pandas as pd
import hashlib
import json
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(filename='log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(input_file='input.csv'):
    warnings = []
    processed_rows = 0

    try:
        df = pd.read_csv(input_file)
        logging.info(f"Successfully read {input_file}")
    except FileNotFoundError:
        logging.error(f"Error: input.csv not found at {input_file}")
        print(f"Error: input.csv not found at {input_file}")
        return
    except Exception as e:
        logging.error(f"Error reading CSV file: {e}")
        print(f"Error reading CSV file: {e}")
        return

    required_columns = ['user_id', 'signup_date', 'last_login_timestamp', 'billing_status', 'plan_name']
    for col in required_columns:
        if col not in df.columns:
            logging.error(f"Error: Missing required column: {col}")
            print(f"Error: Missing required column: {col}")
            return

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

    # Ensure all output columns exist, even if placeholders
    output_columns = [
        'user_id_masked', 'plan_name', 'days_since_signup', 'days_until_renewal',
        'last_login_days_ago', 'number_of_logins_last30days', 'time_to_first_value',
        'support_tickets_opened', 'email_opens_last30days', 'billing_issue_count',
        'monthly_revenue', 'last_payment_status', 'risk_level'
    ]
    for col in output_columns:
        if col not in df.columns:
            df[col] = 0 # Default to 0 or appropriate placeholder

    # Output processed_data.csv
    df[output_columns].to_csv('processed_data.csv', index=False)
    logging.info("Processed data saved to processed_data.csv")

    # Generate stats_summary.json
    total_customers = len(df)
    active_customers = df[df['billing_status'] == 0].shape[0]
    failed_billing_customers = df[df['billing_status'] == 1].shape[0]
    average_revenue = df['monthly_revenue'].mean() if 'monthly_revenue' in df.columns else 0
    avg_login_gap = df['last_login_days_ago'].mean() if 'last_login_days_ago' in df.columns else 0
    churn_rate = 0 # Placeholder

    stats_summary = {
        "total_customers": total_customers,
        "active_customers": active_customers,
        "failed_billing_customers": failed_billing_customers,
        "average_revenue": average_revenue,
        "avg_login_gap": avg_login_gap,
        "churn_rate": churn_rate
    }
    with open('stats_summary.json', 'w') as f:
        json.dump(stats_summary, f, indent=4)
    logging.info("Stats summary saved to stats_summary.json")

    # Log processing summary
    processed_rows = len(df)
    logging.info(f"Processing complete. Processed {processed_rows} rows.")
    if warnings:
        logging.info("Warnings during processing:")
        for warning in warnings:
            logging.info(warning)

    # Print first 10 rows of processed DataFrame
    print("\nFirst 10 rows of processed_data.csv:")
    print(df[output_columns].head(10).to_string())

if __name__ == "__main__":
    process_csv()