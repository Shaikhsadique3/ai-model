import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the merged dataset
data = pd.read_csv('merged_churn_data.csv')   # Yeh wahi file hai jo merge ke baad bani thi

# 2. Data ko thoda clean/prepare karna (encoding text fields)
# Automatic text columns ko number mein convert karte hain
data_encoded = pd.get_dummies(data, drop_first=True)

# 3. Split input (X) and output (y)
X = data_encoded.drop('churn', axis=1)  # Input features
y = data_encoded['churn']               # Output label (0 or 1)

# 4. Train-test split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Model Training (Random Forest Classifier)
model = RandomForestClassifier()
model.fit(X_train, y_train)

# 6. Model Testing
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Training Complete!")
print(f"🎯 Model Accuracy: {accuracy * 100:.2f}%")

# 7. Save the trained model to a file
joblib.dump(model, 'churnaizer_model.pkl')

print("💾 Model saved as: churnaizer_model.pkl")
