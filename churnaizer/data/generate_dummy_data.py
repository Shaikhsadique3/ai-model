import pandas as pd
"""Module for generating dummy SaaS churn data."""

import pandas as pd
import numpy as np
import logging
import os

def generate_dummy_data(num_rows: int = 500) -> pd.DataFrame:
    """Generates a dummy dataset for SaaS churn prediction.

    Args:
        num_rows (int): The number of rows (samples) to generate.

    Returns:
        pd.DataFrame: A DataFrame containing the generated dummy data.
    """
    np.random.seed(42)

    data = {
        'days_since_signup': np.random.randint(1, 1000, num_rows),
        'monthly_revenue': np.random.uniform(10, 500, num_rows),
        'number_of_logins_last30days': np.random.randint(0, 60, num_rows),
        'active_features_used': np.random.randint(0, 10, num_rows),
        'support_tickets_opened': np.random.randint(0, 15, num_rows),
        'last_login_days_ago': np.random.randint(0, 90, num_rows),
        'email_opens_last30days': np.random.randint(0, 30, num_rows),
        'billing_issue_count': np.random.randint(0, 5, num_rows),
        'last_payment_status': np.random.choice(['Success', 'Failed'], num_rows, p=[0.9, 0.1]),
        'subscription_plan': np.random.choice(['Free Trial', 'Basic', 'Pro', 'Enterprise'], num_rows, p=[0.2, 0.4, 0.3, 0.1])
    }

    df = pd.DataFrame(data)

    # Generate churn with desired distribution (30-35% positive class)
    num_churn = int(num_rows * np.random.uniform(0.30, 0.35))
    churn_indices = np.random.choice(df.index, num_churn, replace=False)
    df['churn'] = 0
    df.loc[churn_indices, 'churn'] = 1

    return df

if __name__ == "__main__":
    # Configure logging for standalone execution
    log_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs', 'churnaizer.log')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ])

    try:
        df = generate_dummy_data(num_rows=500)
        file_path = os.path.join(os.path.dirname(__file__), 'enhanced_saas_churn_data.csv')
        df.to_csv(file_path, index=False)
        logging.info(f"Dummy dataset saved to {file_path}")
        logging.info(f"Churn class ratio:\n{df['churn'].value_counts(normalize=True)}")
    except Exception as e:
        logging.error(f"An error occurred during dummy data generation: {e}")