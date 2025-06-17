import pandas as pd
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import LabelEncoder

# Load the CSV file
df = pd.read_csv("/home/stemland/Documents/Students_marks.csv")

# List of subjects (make sure these match your CSV column names)
subjects = ['Math', 'Science', 'English', 'Tamil', 'Social', 'Physics']

# Predict Term 3 marks and convert to grade
def marks_to_grade(m):
    if m >= 90: return 'O'
    elif m >= 80: return 'A+'
    elif m >= 70: return 'A'
    elif m >= 60: return 'B+'
    elif m >= 50: return 'C'
    elif m >= 40: return 'D'
    else: return 'F'

le = LabelEncoder()
for sub in subjects:
    # Features: Term 1 and Term 2 marks
    X = df[[f'{sub}_T1', f'{sub}_T2']]
    # Target: Estimated Term 3 marks (you can change the formula)
    df[f'{sub}_T3'] = (0.4 * df[f'{sub}_T1'] + 0.6 * df[f'{sub}_T2']).round(1)
    y = df[f'{sub}_T3'].apply(marks_to_grade)
    y_encoded = le.fit_transform(y)

    model = OrderedModel(y_encoded, X, distr='logit')
    results = model.fit(method='bfgs', disp=False)

    pred_probs = results.predict(X)
    pred_codes = pred_probs.values.argmax(axis=1)
    df[f"{sub}_Grade"] = le.inverse_transform(pred_codes)

# Count passed and failed subjects
def pass_fail(row):
    passed = sum(g != 'F' for g in row)
    failed = len(row) - passed
    return pd.Series([passed, failed])

grade_cols = [f"{s}_Grade" for s in subjects]
df[['Subjects_Passed', 'Subjects_Failed']] = df[grade_cols].apply(pass_fail, axis=1)

# Final Output
final_df = df[['Name'] + grade_cols + ['Subjects_Passed', 'Subjects_Failed']]
print("\nFinal Predicted Grades and Pass/Fail Summary:\n")
# print(final_df.to_string(index=False))
