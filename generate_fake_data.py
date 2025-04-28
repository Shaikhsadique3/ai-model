import pandas as pd
import random

# Generate 500 fake customers
customers = []
plans = ['Basic', 'Pro', 'Enterprise']

for i in range(500):
    customer_id = f"CUST{i+1:04d}"
    days_since_signup = random.randint(1, 1000)
    monthly_revenue = random.choice([49, 99, 199])
    subscription_plan = random.choice(plans)
    number_of_logins_last30days = random.randint(0, 50)
    active_features_used = random.randint(1, 10)
    support_tickets_opened = random.randint(0, 5)
    last_payment_status = random.choice(['Success', 'Failed'])
    
    # Balanced churn logic:
    # Customer churns if they have very low engagement AND payment issues
    if (number_of_logins_last30days < 3 and last_payment_status == 'Failed') or \
       (number_of_logins_last30days < 2 and active_features_used < 3) or \
       (support_tickets_opened > 4 and last_payment_status == 'Failed'):
        churn = 1
    else:
        churn = 0

    customers.append([
        customer_id, days_since_signup, monthly_revenue, subscription_plan,
        number_of_logins_last30days, active_features_used,
        support_tickets_opened, last_payment_status, churn
    ])

# Create dataframe
columns = ['customer_id', 'days_since_signup', 'monthly_revenue', 'subscription_plan',
           'number_of_logins_last30days', 'active_features_used',
           'support_tickets_opened', 'last_payment_status', 'churn']

df = pd.DataFrame(customers, columns=columns)

# Save to CSV
df.to_csv('fake_saas_churn_data.csv', index=False)

print("✅ Fake SaaS churn dataset ready!")
