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
