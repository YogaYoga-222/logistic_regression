import pandas as pd
import numpy as np
import random
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import LabelEncoder

# Load your CSV file
df = pd.read_csv("/home/stemland/Documents/Students_marks.csv")

subjects = ['Math', 'Science', 'English', 'Tamil', 'Social', 'Physics']

# Grade to mark range
grade_to_range = {
    'O': (90, 100),
    'A+': (80, 89),
    'A': (70, 79),
    'B+': (60, 69),
    'C': (50, 59),
    'D': (40, 49),
    'F': (0, 39)
}

# Set a random seed for reproducibility
np.random.seed(42)

# Convert numeric score to grade
def marks_to_grade(m):
    if m >= 90: return 'O'
    elif m >= 80: return 'A+'
    elif m >= 70: return 'A'
    elif m >= 60: return 'B+'
    elif m >= 50: return 'C'
    elif m >= 40: return 'D'
    else: return 'F'

le = LabelEncoder()

# Predict Term 3 marks using Ordinal Logistic Regression
for sub in subjects:
    X = df[[f'{sub}_T1', f'{sub}_T2']]
    actual_score = (0.4 * df[f'{sub}_T1'] + 0.6 * df[f'{sub}_T2']).round(1)

    y_grades = actual_score.apply(marks_to_grade)
    y_encoded = le.fit_transform(y_grades)

    model = OrderedModel(y_encoded, X, distr='logit')
    results = model.fit(method='bfgs', disp=False)

    pred_probs = results.predict(X)
    pred_codes = pred_probs.values.argmax(axis=1)
    pred_grades = le.inverse_transform(pred_codes)

    # Convert grade to random mark from that range
    df[f'{sub}_T3'] = [np.random.randint(grade_to_range[g][0], grade_to_range[g][1] + 1) for g in pred_grades]

# Count passed/failed subjects
def pass_fail(row):
    passed = sum(row[f'{s}_T3'] >= 40 for s in subjects)
    failed = len(subjects) - passed
    return pd.Series([passed, failed])

df[['Subjects_Passed', 'Subjects_Failed']] = df.apply(pass_fail, axis=1)

# Total marks
df['Total_Marks'] = df[[f'{s}_T3' for s in subjects]].sum(axis=1)

# Assign rank only for fully passed students
df['Rank'] = 0 
passed_students = df[df['Subjects_Failed'] == 0].copy()
passed_students['Rank'] = passed_students['Total_Marks'].rank(ascending=False, method='min').astype(int)
df.update(passed_students)

# Final output
output_cols = ['Name'] + [f'{s}_T3' for s in subjects] + ['Total_Marks', 'Rank', 'Subjects_Passed', 'Subjects_Failed']

# Sort passed students by rank, add failed ones after
passed_df = df[df['Subjects_Failed'] == 0].sort_values(by='Rank')
failed_df = df[df['Subjects_Failed'] > 0]
final_df = pd.concat([passed_df, failed_df])[output_cols]

print("\n Final Predicted Marks, Ranks and Pass/Fail Summary:\n")
print(final_df.to_string(index=False))

# Class Summary
total = len(df)
pass_count = (df['Subjects_Failed'] == 0).sum()
fail_count = total - pass_count
print(f"\n Class Summary : Passed - {pass_count}, Failed - {fail_count}")
