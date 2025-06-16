import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Expanded Dataset
data = {
    'Math':          [95, 88, 45, 60, 30, 70, 55, 40, 85, 50, 68, 60, 45, 70, 65],
    'Biology':       [30, 45, 90, 85, 95, 35, 50, 90, 40, 88, 30, 20, 35, 40, 50],
    'English':       [70, 75, 80, 65, 60, 85, 78, 82, 88, 60, 75, 80, 85, 65, 78],
    'History':       [55, 60, 40, 50, 45, 70, 80, 88, 60, 75, 55, 45, 65, 70, 80],
    'Interest_Science': [1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0],
    'Interest_Art':     [0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 1],
    'Major': [
        'Engineering', 'Engineering', 'Medicine', 'Engineering', 'Medicine',
        'Engineering', 'Arts', 'Arts', 'Engineering', 'Medicine',
        'Engineering', 'Business', 'Arts', 'Business', 'Law'
    ]
}

# DataFrame & Preprocessing
df = pd.DataFrame(data)
X = df.drop('Major', axis=1)
y = df['Major']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Model
model = LogisticRegression(solver='lbfgs')
model.fit(X_scaled, y)

# Prediction Function
def predict_major():
    print("\n--- Academic Major Predictor ---")
    features = {
        'Math': float(input("Math Score: ")),
        'Biology': float(input("Biology Score: ")),
        'English': float(input("English Score: ")),
        'History': float(input("History Score: ")),
        'Interest_Science': int(input("Interest in Science? (1 = Yes, 0 = No): ")),
        'Interest_Art': int(input("Interest in Art? (1 = Yes, 0 = No): "))
    }

    input_df = pd.DataFrame([features])
    input_scaled = scaler.transform(input_df)

    predicted = model.predict(input_scaled)[0]
    confidence = max(model.predict_proba(input_scaled)[0]) * 100

    print(f"\n Predicted Major: {predicted}")
    print(f" Confidence: {confidence:.2f}%")

# Run
predict_major()
