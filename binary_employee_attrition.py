# binary_logistic_regression_employee_attrition.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample employee data
data = {
    'satisfaction_level': [0.38, 0.80, 0.11, 0.72, 0.37, 0.85, 0.23, 0.54],
    'number_project': [2, 5, 7, 5, 2, 6, 3, 4],
    'average_monthly_hours': [157, 262, 272, 223, 159, 280, 160, 210],
    'left': [1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Left company, 0 = Stayed
}

# Create DataFrame
df = pd.DataFrame(data)

# Features and Target
X = df[['satisfaction_level', 'number_project', 'average_monthly_hours']]
y = df['left']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: Print actual predictions
print("\nTest Data Predictions:", y_pred.tolist())
