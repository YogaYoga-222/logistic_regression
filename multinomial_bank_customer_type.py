import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Load dataset from CSV
df = pd.read_csv("/home/stemland/Downloads/bank_customer_type.csv") \

# Features and Target
X = df[['Monthly_Income', 'Account_Balance', 'Transactions']]
y = df['Customer_Type']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train multinomial logistic regression
model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# Take user input
print("\nEnter customer details to predict customer type:")
monthly_income = float(input("Monthly Income: "))
account_balance = float(input("Account Balance: "))
transactions = int(input("Number of Transactions: "))

# Prepare input
input_df = pd.DataFrame([[monthly_income, account_balance, transactions]],
                        columns=['Monthly_Income', 'Account_Balance', 'Transactions'])
input_scaled = scaler.transform(input_df)

# Predict
pred_label = model.predict(input_scaled)[0]
pred_class = le.inverse_transform([pred_label])[0]

# Output
print(f"\nPredicted Customer Type: **{pred_class}**")
