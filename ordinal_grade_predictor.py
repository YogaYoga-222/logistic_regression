import pandas as pd
import numpy as np
from statsmodels.miscmodels.ordinal_model import OrderedModel

# Sample training data
marks = [90, 85, 75, 65, 55, 45, 35, 80, 70, 60]
attendance = [95, 90, 85, 80, 70, 60, 50, 88, 75, 65]
grades = ['O', 'A+', 'A', 'B+', 'C', 'D', 'F', 'A+', 'A', 'B+']

data = pd.DataFrame({
    'marks': marks,
    'attendance': attendance,
    'grade': grades
})

# Encode grades
grade_order = ['F', 'D', 'C', 'B+', 'A', 'A+', 'O']
data['grade'] = pd.Categorical(data['grade'], categories=grade_order, ordered=True)
data['grade_code'] = data['grade'].cat.codes

# Train model
model = OrderedModel(data['grade_code'], data[['marks', 'attendance']], distr='logit')
results = model.fit(method='bfgs', disp=False)

# Get input
try:
    m = int(input("Enter your marks (0 to 100): "))
    a = int(input("Enter your attendance (0 to 100): "))

    if 0 <= m <= 100 and 0 <= a <= 100:
        if a < 70:
            print("\nAttendance is too low. You are failed.")
            print("Predicted Grade: F")
        else:
            new_data = pd.DataFrame({'marks': [m], 'attendance': [a]})
            probs = results.predict(new_data)
            pred_code = probs.values.argmax()
            pred_grade = grade_order[pred_code]
            print(f"\nPredicted Grade: {pred_grade}")
    else:
        print("Marks and attendance must be between 0 and 100.")
except ValueError:
    print("Please enter valid numbers.")







# import pandas as pd
# from sklearn.linear_model import LinearRegression

# # Load your CSV file
# df = pd.read_csv("/home/stemland/Documents/Students_marks.csv")

# # Subjects list
# subjects = ['Math', 'Science', 'English', 'Tamil', 'Social', 'Physics']

# # Predict Term 3 marks
# for subject in subjects:
#     X = df[[f'{subject}_T1', f'{subject}_T2']]
#     y = 0.4 * df[f'{subject}_T1'] + 0.6 * df[f'{subject}_T2']  # You can change this logic
#     model = LinearRegression()
#     model.fit(X, y)
#     df[f'{subject}_T3_Predicted'] = model.predict(X).round(2)

# # Count passed and failed subjects
# def count_pass_fail(row):
#     passed = sum(row[f'{s}_T3_Predicted'] >= 40 for s in subjects)
#     return pd.Series([passed, len(subjects) - passed])

# df[['Subjects_Passed', 'Subjects_Failed']] = df.apply(count_pass_fail, axis=1)

# # Output results
# output_cols = ['Name'] + [f'{s}_T3_Predicted' for s in subjects] + ['Subjects_Passed', 'Subjects_Failed']
# final_df = df[output_cols]

# # Save and display
# final_df.to_csv("term3_predictions.csv", index=False)
# print("\n Term 3 Predictions:\n")
# print(final_df.to_string(index=False))
