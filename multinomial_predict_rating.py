import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Sample Vacation Rating Dataset
data = {
    'Budget': [5000, 20000, 10000, 8000, 25000, 15000, 12000, 6000, 30000, 10000],
    'Hotel_Cost': [3000, 12000, 7000, 5000, 15000, 9000, 8000, 3500, 18000, 6500],
    'Travel_Time': [5, 2, 4, 6, 1, 3, 4, 6, 2, 5],
    'Feel_Good': [7, 9, 8, 6, 10, 8, 9, 5, 10, 7],
    'Weather': [1, 2, 1, 3, 2, 1, 1, 3, 2, 1],
    'Rating': [2, 5, 3, 1, 5, 4, 5, 2, 5, 3]
}

# Prepare Data
df = pd.DataFrame(data)
X = df.drop('Rating', axis=1)
y = df['Rating']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the Logistic Regression model
model = LogisticRegression(max_iter=5000)
model.fit(X_scaled, y)

# Prediction Function
def predict_rating():
    print("\n--- Vacation Rating Predictor ---")
    
    # Get user inputs
    budget = float(input("Total budget (₹): "))
    hotel_cost = float(input("Expected hotel cost (₹): "))
    travel_time = float(input("Travel time (in hours): "))
    feel_good = int(input("Feel-good rating (1-10): "))
    weather = int(input("Weather (1=Cool, 2=Moderate, 3=Hot): "))

    # Create input as a DataFrame with feature names
    input_df = pd.DataFrame([{
        'Budget': budget,
        'Hotel_Cost': hotel_cost,
        'Travel_Time': travel_time,
        'Feel_Good': feel_good,
        'Weather': weather
    }])

    # Scale and predict
    input_scaled = scaler.transform(input_df)
    predicted = model.predict(input_scaled)[0]
    confidence = max(model.predict_proba(input_scaled)[0]) * 100

    print(f"\n Predicted Rating: {predicted} Star(s)")
    print(f" Confidence: {confidence:.2f}%")

# Run the predictor
predict_rating()
