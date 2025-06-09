import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sample dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 30, 35],
    'Salary': [15000, 29000, 48000, 60000, 58000, 52000, 64000, 70000, 32000, 40000],
    'Purchased': [0, 0, 1, 1, 1, 1, 1, 1, 0, 0]
}

df = pd.DataFrame(data)

# Features and target
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Results
print("Purchase Prediction")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Test Data Predictions:", y_pred.tolist())
