import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the CSV file
df = pd.read_csv("/home/stemland/Downloads/bank_customer_type.csv") 

# Features and target
X = df[['Monthly_Income', 'Account_Balance', 'Transactions']]
y = df['Customer_Type']  # Target: Saver, Spender, Investor

# Encode the target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("\nClassification Report:\n")
print(classification_report(le.inverse_transform(y_test), le.inverse_transform(y_pred)))
print("Accuracy:", accuracy_score(le.inverse_transform(y_test), le.inverse_transform(y_pred)))

# User Input for Prediction
print("\nLet's predict a new customer's type:")

# Get input from user
income = float(input("Enter Monthly Income: "))
balance = float(input("Enter Account Balance: "))
transactions = int(input("Enter Number of Monthly Transactions: "))

# Preprocess user input
user_data = [[income, balance, transactions]]
user_data_scaled = scaler.transform(user_data)

# Predict
prediction = model.predict(user_data_scaled)
prediction_label = le.inverse_transform(prediction)[0]

# Show prediction
print(f"\nPredicted Customer Type: {prediction_label}")

# Model confidence
proba = model.predict_proba(user_data_scaled)[0]
print("\nPrediction Confidence:")
for i, label in enumerate(le.classes_):
    print(f"{label}: {proba[i]*100:.2f}%")
