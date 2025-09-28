import pandas as pd
import hashlib
import json
import logging
from datetime import datetime, timedelta

# Configure logging for the module
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_csv(input_file: str):
    """
    Processes a raw CSV file containing customer data for churn prediction.

    This function performs the following steps:
    1. Reads the CSV file from the given path.
    2. Validates the presence of required columns.
    3. Cleans and converts data types (e.g., dates).
    4. Normalizes categorical features like 'billing_status'.
    5. Fills missing optional numerical values with defaults.
    6. Masks sensitive 'user_id' information.
    7. Derives new features such as 'days_since_signup' and 'last_login_days_ago'.
    8. Renames columns to align with the model's expectations.
    9. Ensures all numeric columns are correctly typed.
    10. Generates a summary of key statistics from the processed data.
    11. Returns the processed DataFrame, a summary of statistics, and any warnings encountered.

    Args:
        input_file (str): The absolute path to the input CSV file.

    Returns:
        tuple: A tuple containing:
            - pd.DataFrame: The processed DataFrame ready for model prediction.
            - dict: A dictionary summarizing key statistics of the processed data.
            - list: A list of warning messages encountered during processing.
            Returns (None, None, warnings) if critical errors occur (e.g., file not found, missing columns).
    """
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

    # Validate required columns are present in the DataFrame
    required_columns = ['user_id', 'signup_date', 'last_login_timestamp', 'billing_status', 'plan_name']
    for col in required_columns:
        if col not in df.columns:
            error_msg = f"Error: Missing required column: {col}"
            logging.error(error_msg)
            warnings.append(error_msg)
            return None, None, warnings

    # Keep the original user_id for potential later use or linking, before masking
    df['original_user_id'] = df['user_id']

    # --- Data Cleaning and Conversion ---
    # Convert date columns to datetime objects, coercing errors to NaT (Not a Time)
    for date_col in ['signup_date', 'last_login_timestamp']:
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')

    # Drop rows where date conversion failed for required date columns
    initial_rows = len(df)
    df.dropna(subset=['signup_date', 'last_login_timestamp'], inplace=True)
    if len(df) < initial_rows:
        warnings.append(f"Dropped {initial_rows - len(df)} rows due to invalid or missing dates in 'signup_date' or 'last_login_timestamp'.")
        logging.warning(f"Dropped {initial_rows - len(df)} rows due to invalid or missing dates in 'signup_date' or 'last_login_timestamp'.")

    # Normalize 'billing_status': 'active' to 0, 'failed' to 1, others remain as is (or handled by model preprocessing)
    df['billing_status'] = df['billing_status'].apply(lambda x: 0 if x == 'active' else 1 if x == 'failed' else x)

    # Fill missing optional numerical values with defaults
    numeric_cols_defaults = {
        'support_tickets_opened': 0,
        'email_opens_last30days': 0,
        'monthly_revenue': 0,
        'avg_session_duration': 0,
        'number_of_logins_last30days': 10,
        'time_to_first_value': 7,
        'active_features_used': 3
    }
    for col, default_value in numeric_cols_defaults.items():
        if col not in df.columns:
            df[col] = default_value
        else:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(default_value)

    # Add 'last_payment_status' if not present, defaulting to 'success'
    if 'last_payment_status' not in df.columns:
        df['last_payment_status'] = 'success'

    # Add 'subscription_plan' if not present; use 'plan_name' if available, else default to 'basic'
    if 'subscription_plan' not in df.columns:
        df['subscription_plan'] = df.get('plan_name', 'basic')

    # Ensure all numeric columns are properly typed after all transformations
    numeric_columns = ['days_since_signup', 'last_login_days_ago', 'monthly_revenue', 
                      'support_tickets_opened', 'email_opens_last30days', 'billing_issue_count',
                      'number_of_logins_last30days', 'active_features_used', 'avg_session_duration',
                      'trial_conversion_flag']
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)





























    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate user_ids, keeping the first occurrence
    df.drop_duplicates(subset='user_id', inplace=True)

    # Mask 'user_id' for privacy and reporting purposes using SHA256 hash
    df['user_id_masked'] = df['user_id'].astype(str).apply(lambda x: hashlib.sha256(x.encode()).hexdigest()[:8])

    # --- Derive new features ---
    current_time = datetime.now()
    # Calculate days since user signed up
    df['days_since_signup'] = (current_time - df['signup_date']).dt.days
    # Calculate days since last login
    df['last_login_days_ago'] = (current_time - df['last_login_timestamp']).dt.days
    # Flag for trial conversion: 1 if trial plan and positive revenue, else 0
    df['trial_conversion_flag'] = ((df['subscription_plan'].astype(str).str.lower().str.contains('trial')) & (df['monthly_revenue'] > 0)).astype(int)

    # Add placeholder fields that the prediction model might expect but are not directly in raw data
    df['days_until_renewal'] = 30  # Placeholder: assuming monthly renewal for simplicity
    df['billing_issue_count'] = df['billing_status']  # Using normalized billing_status as a proxy for issue count

    # Rename 'plan_name' to 'subscription_plan' to match model expectations if not already done
    if 'plan_name' in df.columns:
        df['subscription_plan'] = df['plan_name']

    # Remove duplicate