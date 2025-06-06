import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Sample data
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'Passed': [0, 0, 0, 0, 1, 1, 1, 1]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Features and target
X = df[['Hours_Studied']]  # Input
y = df['Passed']           # Output

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Results
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Test Data Predictions:", y_pred.tolist())
