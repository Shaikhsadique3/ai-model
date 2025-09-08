import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def validate_mandatory_columns(df):
    """Validate that all mandatory columns are present."""
    mandatory_columns = [
        'user_id', 'signup_date', 'last_login_timestamp', 
        'billing_status', 'plan_name'
    ]
    
    missing_columns = [col for col in mandatory_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing mandatory columns: {', '.join(missing_columns)}")
    
    return True

def validate_date_column(df, column_name):
    """Validate and convert date columns to datetime."""
    try:
        df[column_name] = pd.to_datetime(df[column_name], errors='coerce')
        invalid_dates = df[column_name].isnull().sum()
        if invalid_dates > 0:
            logging.warning(f"Found {invalid_dates} invalid dates in {column_name}")
        return df
    except Exception as e:
        raise ValueError(f"Error parsing {column_name}: {str(e)}")

def derive_days_since_signup(df, today):
    """Calculate days since signup."""
    if 'signup_date' in df.columns:
        df['days_since_signup'] = (today - df['signup_date']).dt.days
    else:
        logging.warning("signup_date column missing - setting days_since_signup to NaN")
        df['days_since_signup'] = np.nan
    return df

def derive_days_until_renewal(df, today):
    """Calculate days until renewal."""
    if 'renewal_date' in df.columns:
        df['days_until_renewal'] = (df['renewal_date'] - today).dt.days
    else:
        logging.warning("renewal_date column missing - setting days_until_renewal to NaN")
        df['days_until_renewal'] = np.nan
    return df

def derive_last_login_days_ago(df, today):
    """Calculate days since last login."""
    if 'last_login_timestamp' in df.columns:
        df['last_login_days_ago'] = (today - df['last_login_timestamp']).dt.days
    return df

def derive_number_of_logins_last30days(df, today):
    """Calculate number of logins in last 30 days."""
    def count_logins_last30days(login_data):
        if pd.isna(login_data) or login_data == '':
            return 0
        
        try:
            # Handle different input formats
            if isinstance(login_data, (int, float)):
                return int(login_data)
            
            # Handle semicolon-separated timestamps
            events = str(login_data).split(';')
            valid_events = []
            
            for event in events:
                event = event.strip()
                if event:
                    try:
                        event_date = pd.to_datetime(event)
                        valid_events.append(event_date)
                    except:
                        continue
            
            # Count events within last 30 days
            cutoff_date = today - timedelta(days=30)
            return sum(1 for event in valid_events if event >= cutoff_date)
            
        except Exception as e:
            logging.warning(f"Error processing login data: {e}")
            return 0
    
    if 'login_events' in df.columns:
        df['number_of_logins_last30days'] = df['login_events'].apply(count_logins_last30days)
    else:
        # Try to derive from login timestamps if available
        if 'login_timestamps' in df.columns:
            df['number_of_logins_last30days'] = df['login_timestamps'].apply(count_logins_last30days)
        else:
            logging.warning("No login data columns found - setting number_of_logins_last30days to 0")
            df['number_of_logins_last30days'] = 0
    
    return df

def derive_time_to_first_value(df):
    """Calculate time to first value."""
    if 'first_key_event_date' in df.columns:
        df['time_to_first_value'] = (df['first_key_event_date'] - df['signup_date']).dt.days
    else:
        logging.warning("first_key_event_date column missing - setting time_to_first_value to NaN")
        df['time_to_first_value'] = np.nan
    return df

def normalize_billing_status(df):
    """Convert billing_status to numeric flag."""
    df['billing_issue_count'] = df['billing_status'].apply(
        lambda x: 1 if str(x).lower() == 'failed' else 0
    )
    return df

def process_csv(input_csv_path: str, output_csv_path: str = 'processed_data.csv'):
    """
    Process CSV file for churn prediction with derived fields.
    
    Args:
        input_csv_path (str): Path to input CSV file
        output_csv_path (str): Path to save processed CSV file
    """
    
    print(f"Starting CSV processing...")
    print(f"Input file: {input_csv_path}")
    
    try:
        # 1. Load the CSV
        df = pd.read_csv(input_csv_path)
        print(f"Loaded {len(df)} rows from CSV")
        
        # 2. Validate mandatory columns
        validate_mandatory_columns(df)
        
        # 3. Validate date columns
        date_columns = ['signup_date', 'last_login_timestamp']
        if 'renewal_date' in df.columns:
            date_columns.append('renewal_date')
        if 'first_key_event_date' in df.columns:
            date_columns.append('first_key_event_date')
            
        for col in date_columns:
            if col in df.columns:
                df = validate_date_column(df, col)
        
        # 4. Get today's date for calculations
        today = datetime.now()
        
        # 5. Derive all required fields
        df = derive_days_since_signup(df, today)
        df = derive_days_until_renewal(df, today)
        df = derive_last_login_days_ago(df, today)
        df = derive_number_of_logins_last30days(df, today)
        df = derive_time_to_first_value(df)
        df = normalize_billing_status(df)
        
        # 6. Handle optional columns
        optional_columns = {
            'support_tickets_opened': 0,
            'email_opens_last30days': 0,
            'monthly_revenue': 0,
            'last_payment_status': 'success'
        }
        
        for col, default_value in optional_columns.items():
            if col not in df.columns:
                logging.warning(f"Optional column '{col}' missing - using default value {default_value}")
                df[col] = default_value
        
        # 7. Prepare final DataFrame
        final_columns = [
            'user_id',
            'days_since_signup',
            'days_until_renewal',
            'last_login_days_ago',
            'number_of_logins_last30days',
            'time_to_first_value',
            'support_tickets_opened',
            'email_opens_last30days',
            'billing_issue_count',
            'subscription_plan',  # from plan_name
            'monthly_revenue',
            'last_payment_status'
        ]
        
        # Rename plan_name to subscription_plan
        df = df.rename(columns={'plan_name': 'subscription_plan'})
        
        # Ensure all final columns exist
        missing_final_columns = [col for col in final_columns if col not in df.columns]
        if missing_final_columns:
            raise ValueError(f"Missing final columns: {missing_final_columns}")
        
        # Select only final columns
        processed_df = df[final_columns]
        
        # 8. Save processed data
        processed_df.to_csv(output_csv_path, index=False)
        print(f"Processed data saved to: {output_csv_path}")
        
        # 9. Print preview
        print("\nFirst 10 rows of processed data:")
        print(processed_df.head(10))
        
        # 10. Summary statistics
        print(f"\nProcessing completed successfully!")
        print(f"Total rows processed: {len(processed_df)}")
        print(f"Columns: {list(processed_df.columns)}")
        
        return processed_df
        
    except Exception as e:
        logging.error(f"Error processing CSV: {str(e)}")
        raise

if __name__ == "__main__":
    # Example usage
    try:
        process_csv('input.csv')
    except Exception as e:
        print(f"Error: {e}")