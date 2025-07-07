# Generate a synthetic churn dataset with 10,000 rows and SaaS-specific features 
# Include real behavioral patterns and clear churn signals 
# Maintain churn rate between 30–35% 

import pandas as pd 
import numpy as np 
import random 

def generate_enhanced_churn_data(num_rows=10000): 
    data = { 
        "days_since_signup": np.random.randint(1, 365, num_rows), 
        "monthly_revenue": np.round(np.random.uniform(5, 999, num_rows), 2), 
        "subscription_plan": np.random.choice(["Free Trial", "Basic", "Pro", "Enterprise"], num_rows), 
        "number_of_logins_last30days": np.random.randint(0, 100, num_rows), 
        "active_features_used": np.random.randint(0, 15, num_rows), 
        "support_tickets_opened": np.random.randint(0, 10, num_rows), 
        "last_payment_status": np.random.choice(["Success", "Failed"], num_rows), 
        "avg_session_duration": np.random.uniform(2, 90, num_rows),  # in minutes 
        "last_login_days_ago": np.random.randint(0, 60, num_rows), 
        "email_opens_last30days": np.random.randint(0, 25, num_rows), 
        "billing_issue_count": np.random.randint(0, 5, num_rows), 
        "trial_conversion_flag": np.random.choice([0, 1], num_rows) 
    } 

    df = pd.DataFrame(data) 
    df["churn"] = 0 

    # Rule-based churn logic 
    churn_conditions = ( 
        (df["number_of_logins_last30days"] < 5) & 
        (df["active_features_used"] < 3) & 
        (df["last_payment_status"] == "Failed") | 
        (df["subscription_plan"] == "Free Trial") & 
        (df["days_since_signup"] < 10) & 
        (df["number_of_logins_last30days"] < 3) | 
        (df["billing_issue_count"] > 2) 
    ) 

    df.loc[churn_conditions, "churn"] = 1 

    # Balance churn ratio to 30–35% 
    target_churn_rate = 0.33 
    current_rate = df["churn"].mean() 

    if current_rate < target_churn_rate: 
        to_churn = df[df["churn"] == 0].sample( 
            int((target_churn_rate - current_rate) * num_rows)) 
        df.loc[to_churn.index, "churn"] = 1 
    elif current_rate > target_churn_rate: 
        to_unlabel = df[df["churn"] == 1].sample( 
            int((current_rate - target_churn_rate) * num_rows)) 
        df.loc[to_unlabel.index, "churn"] = 0 

    df.drop_duplicates(inplace=True) 
    df.reset_index(drop=True, inplace=True) 
    df.to_csv("enhanced_saas_churn_data.csv", index=False) 
    print("✅ Dataset created: enhanced_saas_churn_data.csv with", len(df), "rows.") 
    print(f"📊 Churn Rate: {df['churn'].mean():.2%}") 

generate_enhanced_churn_data()