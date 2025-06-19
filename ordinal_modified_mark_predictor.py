import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.preprocessing import LabelEncoder

# Load CSV
df = pd.read_csv("/home/stemland/Documents/Students_marks.csv")

subjects = ['Math', 'Science', 'English', 'Tamil', 'Social', 'Physics']

# Grade to mark range and average value
grade_to_range = {
    'O': (90, 100),
    'A+': (80, 89),
    'A': (70, 79),
    'B+': (60, 69),
    'C': (50, 59),
    'D': (40, 49),
    'F': (0, 39)
}

# Get midpoint (average) mark of each grade range
grade_to_avg = {grade: (low + high) // 2 for grade, (low, high) in grade_to_range.items()}

# Convert score to grade
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
    
    # Use equal weights now (0.5 + 0.5)
    actual_score = (0.5 * df[f'{sub}_T1'] + 0.5 * df[f'{sub}_T2']).round(1)
    
    y_grades = actual_score.apply(marks_to_grade)
    y_encoded = le.fit_transform(y_grades)

    model = OrderedModel(y_encoded, X, distr='logit')
    results = model.fit(method='bfgs', disp=False)

    pred_probs = results.predict(X)
    pred_codes = pred_probs.values.argmax(axis=1)
    pred_grades = le.inverse_transform(pred_codes)

    # Predict stable marks from grade average
    df[f'{sub}_T3'] = [grade_to_avg[g] for g in pred_grades]

# Count passed/failed subjects
def pass_fail(row):
    passed = sum(row[f'{s}_T3'] >= 40 for s in subjects)
    failed = len(subjects) - passed
    return pd.Series([passed, failed])

df[['Subjects_Passed', 'Subjects_Failed']] = df.apply(pass_fail, axis=1)

# Total and Rank
df['Total_Marks'] = df[[f'{s}_T3' for s in subjects]].sum(axis=1)

# Assign rank only for fully passed students
df['Rank'] = 0
passed_students = df[df['Subjects_Failed'] == 0].copy()
passed_students['Rank'] = passed_students['Total_Marks'].rank(ascending=False, method='min').astype(int)
df.update(passed_students)

# Output
output_cols = ['Name'] + [f'{s}_T3' for s in subjects] + ['Total_Marks', 'Rank', 'Subjects_Passed', 'Subjects_Failed']

# Sort passed students by rank, add failed ones after
passed_df = df[df['Subjects_Failed'] == 0].sort_values(by='Rank')
failed_df = df[df['Subjects_Failed'] > 0]
final_df = pd.concat([passed_df, failed_df])[output_cols]

print("\n Final Predicted Marks, Ranks and Pass/Fail Summary:\n")
print(final_df.to_string(index=False))

# Class summary
pass_count = (df['Subjects_Failed'] == 0).sum()
fail_count = len(df) - pass_count
print(f"\n Class Summary : Passed - {pass_count}, Failed - {fail_count}")

# Search for a student's result by name
while True:
    name = input("\nEnter student name to view result (or type 'exit' to quit): ").strip()
    if name.lower() == 'exit':
        break

    student_row = final_df[final_df['Name'].str.lower() == name.lower()]
    
    if not student_row.empty:
        print(f"\nResult for {name}:\n")
        print(student_row.to_string(index=False))
    else:
        print(f"\nNo student found with the name: {name}")
