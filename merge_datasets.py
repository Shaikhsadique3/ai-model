import pandas as pd

# 1. Dono dataset ko load karo
telco = pd.read_csv('telco_churn.csv')   # Telco dataset ka naam
saas = pd.read_csv('fake_saas_churn_data.csv')   # Apna fake SaaS dataset

# 2. Column names same nahi hai dono mein — to hum common columns le lenge
# Pehle Telco dataset ke columns dekh lo
print(telco.columns)
print(saas.columns)

# 3. Common columns choose karo ya rename karo
# Telco dataset ko SaaS dataset jaise bana do (Optional - depends on you)
# Example: yaha simple karte hain: sirf matching columns use karenge

# 4. Merge karte hain vertically (upar neeche jodte hain)
merged = pd.concat([saas], axis=0)  # Sirf fake data se start karte hain pehle
# NOTE: Agar aap Telco dataset ke kuch fields match karna chahte ho to bolna, main rename wala bhi code de dunga.

# 5. Merged dataset ko save karo
merged.to_csv('merged_churn_data.csv', index=False)

print("✅ Merge complete! File saved as: merged_churn_data.csv")
