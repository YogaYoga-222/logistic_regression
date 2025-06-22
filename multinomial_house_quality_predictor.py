import pandas as pd
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.exceptions import ConvergenceWarning

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Load dataset 
df = pd.read_csv("/home/stemland/Documents/house_quality.csv")

# Features and target
X = df[['Square_Footage', 'Location_Score', 'Material_Quality']]
y = df['Quality']

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Train multinomial logistic regression 
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)

# Make predictions 
y_pred = model.predict(X_test)

# Decode predictions
actual_quality = le.inverse_transform(y_test)
predicted_quality = le.inverse_transform(y_pred)

# Create test results DataFrame 
test_results = pd.DataFrame(X_test, columns=['Square_Footage', 'Location_Score', 'Material_Quality'])
test_results['Actual_Quality'] = actual_quality
test_results['Predicted_Quality'] = predicted_quality

# Show classification report 
print("\nClassification Report:\n")
print(classification_report(actual_quality, predicted_quality, labels=le.classes_))
print("Accuracy:", accuracy_score(actual_quality, predicted_quality))
print("\nConfusion Matrix:\n", confusion_matrix(actual_quality, predicted_quality))

# Correct Predictions 
correct_preds = test_results[test_results['Actual_Quality'] == test_results['Predicted_Quality']]
print("\n Correct Predictions:\n")
print(correct_preds)

# Wrong Predictions 
wrong_preds = test_results[test_results['Actual_Quality'] != test_results['Predicted_Quality']]
if wrong_preds.empty:
    print("\n No wrong predictions. Great job!\n")
else:
    print("\n Wrong Predictions:\n")
    print(wrong_preds)
