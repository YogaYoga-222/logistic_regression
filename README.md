# Logistic Regression Projects

This project includes **Thirteen Classification Problems** solved using **Logistic Regression** with the help of `scikit-learn`.

Each program is written in its own Python file for better clarity and organization.

---

## Files

- `logistic_regression_iris.py`: Iris flower classification
- `logistic_regression_breast_cancer.py`: Breast cancer prediction
- `logistic_regression_spam_detection.py`: Spam vs. ham message classification
- `binary_pass_prediction.py`: Predict if a student passes based on study hours
- `binary_employee_attrition.py`: Predict if an employee will leave the company  
- `binary_purchase_prediction.py`: Predict if a person will purchase a product
- `binary_introvert_vs_extrovert.py`: Predict if a person is introvert or extrovert based on personality traits
- `multinomial_predict_rating.py`: Predict vacation place ratings (1–5 stars)
- `multinomial_academic_major_prediction.py`: Predict student’s academic major
- `multinomial_house_quality_predictor.py`: Predict house quality as Basic, Average, or Premium using square footage, location score, and material quality.
- `ordinal_grade_predictor.py`: Predict students grade based on marks and attendance
- `ordinal_marks_predictor.py` : Predict Term 3 subject grades using marks from Term 1 and 2 and show pass/fail summary
- `ordinal_modified_mark_predictor.py`:  Predict Term 3 marks using Term 1 and 2. Show total marks, how many subjects each student passed or failed, and give ranks only to students who passed all subjects.

---

## Project Descriptions

### 1. Iris Flower Classification

- **Dataset**: Iris dataset (from scikit-learn)
- **Target**: Classify flowers based on petal and sepal measurements
- **Goal**: Predict the species of iris flower (Setosa, Versicolor, Virginica)

-----

### 2. Breast Cancer Classification

- **Dataset**: Breast Cancer dataset (from scikit-learn)
- **Target**: Predict whether a tumor is malignant or benign
- **Goal**: Help in early detection of breast cancer using logistic regression

---

### 3. Spam Detection

- **Dataset**: SMS messages (sample or real-world dataset)  
- **Target**: Classify messages as spam or ham  
- **Goal**: Build a text classifier using TF-IDF and logistic regression to detect spam  

---

### 4. Student Pass Prediction

- **Dataset**: Custom simple dataset  
- **Target**: Predict whether a student will pass based on hours studied  
- **Goal**: Demonstrate binary classification using logistic regression

---

### 5. Employee Attrition Prediction

- **Dataset**: Custom simple dataset  
- **Target**: Predict whether an employee will leave the company  
- **Goal**: Show how logistic regression can be used in HR analytics  

---

### 6. Purchase Prediction
- **Dataset**: Custom simple dataset  
- **Target**: Predict whether a person will buy a product based on age and salary  
- **Goal**: Use logistic regression for marketing-based decisions  

---

### 7. Introvert vs Extrovert Prediction 
- **Dataset**: Personality traits dataset (custom or collected)
- **Target**: Predict whether a person is an Introvert or Extrovert
- **Goal**: Apply logistic regression to classify individuals based on social behavior and habits (like time spent alone, stage fear, and event participation)
- **Note**: Confirm you loaded the personality_data.csv in `binary_introvert_vs_extrovert.py`

---

### 8. Vacation Place Rating Prediction
- **Dataset**: Custom data with 10 records
- **Target**: Predict a vacation spot’s rating (1–5 stars)
- **Goal**: Use Multinomial Logistic Regression to predict the rating of a place based on - Budget, Expected Hotel Cost, Travel Time, Feel-Good Score, Weather Condition

---

### 9. Academic Major Prediction
- **Dataset**: Based on subject scores and student interests
- **Target**: Predict the most suitable academic major for a student
- **Goal**: Use logistic regression to predict a student’s major based on their marks in Math, Biology, English, and History, and their interest in Science and Art

---

### 10. House Quality Rating Prediction
- **Dataset**: house_quality.csv
- **Target**: Predict the house quality as Basic, Average, or Premium
- **Goal**: Use Multinomial Logistic Regression to predict the quality rating of a house based on features like Square Footage, Location Score, and Material Quality
- **Note**: Ensure house_quality.csv is loaded in multinomial_house_quality.py

---

### 11. Student Grade Prediction
- **Dataset**: Student marks and attendance
- **Target**: Predict the grade (F to O)
- **Goal**: Predict a student's grade using marks and attendance. If attendance is below 70%, the grade is 'F'. The model uses ordinal logistic regression to do this.

---

### 12. Term 3 Grade and Pass/Fail Prediction
- **Dataset**: Students_marks.csv with Term 1 and 2 marks
- **Task**: Predict Term 3 grades for 6 subjects
- **Goal**: Use ordinal logistic regression to convert marks to grades and count passed/failed subjects
- **Note**: Confirm you loaded the Students_marks.csv in `ordinal_marks_predictor.py`

---

## 13. Modified Term 3 Mark and Rank Prediction
- **Dataset**: Students_marks.csv with Term 1 and 2 marks
- **Task**: This is a modified version of the previous grade prediction task. Instead of grades, it predicts actual marks for Term 3 in 6 subjects.
- **Goal**: Use ordinal logistic regression to estimate marks, calculate total marks, count how many subjects each student passed or failed, and rank only the students who passed all subjects.
- **Note**: Confirm you loaded the Students_marks.csv in ordinal_modified_mark_predictor.py

---

## Required Libraries

Before running the programs, make sure you have the following Python library installed:

```bash
pip install scikit-learn
```
```bash
pip install numpy pandas scikit-learn statsmodels
```

# Run the Programs :

```bash
python3 logistic_regression_iris.py
```
```bash
python3 logistic_regression_breast_cancer.py
```
```bash
python3 logistic_regression_spam_detection.py
```
```bash
python3 binary_pass_prediction.py
```
```bash
python3 binary_employee_attrition.py
```
```bash
python3 binary_purchase_prediction.py
```
```bash
python3 binary_introvert_vs_extrovert.py
```
```bash
python3 multinomial_predict_rating.py
```
```bash
python3 multinomial_academic_major_prediction.py
```
```bash
python3 ordinal_grade_predictor.py
```
```bash
python3 ordinal_marks_predicto.py
```
```bash
python3 ordinal_modified_mark_predictor.py
```
```bash
python3 multinomial_house_quality_predictor.py
```

# Sample Output :

## Iris Classification
```
Output for iris dataset
Accuracy: 1.0
Confusion Matrix:
[[10  0]
 [ 0 10]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00        10
           1       1.00      1.00      1.00        10
```
## Breast Cancer Classification
```
Breast Cancer Classification
Accuracy: 0.96
Confusion Matrix:
[[70  2]
 [ 2 42]]
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97        72
           1       0.95      0.95      0.95        44
```
## Spam Detection
```
Accuracy: 0.67
Confusion Matrix:
[[1 1]
 [0 1]]
Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.50      0.67         2
           1       0.50      1.00      0.67         1
```
## Student Pass Prediction
```
Accuracy: 1.0
Test Data Predictions: [0, 1, 0]
```
## Employee Attrition Prediction
```
Accuracy: 1.0

Confusion Matrix:
 [[2 0]
 [0 1]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         2
           1       1.00      1.00      1.00         1

    accuracy                           1.00         3
   macro avg       1.00      1.00      1.00         3
weighted avg       1.00      1.00      1.00         3


Test Data Predictions: [0, 0, 1]
```
## Purchase Prediction
```
Purchase Prediction
Accuracy: 1.0
Confusion Matrix:
 [[2 0]
 [0 1]]
Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00         2
           1       1.00      1.00      1.00         1

    accuracy                           1.00         3
   macro avg       1.00      1.00      1.00         3
weighted avg       1.00      1.00      1.00         3

Test Data Predictions: [0, 0, 1]
```
## Introvert VS Extrovert
```
Accuracy: 0.9220430107526881
Confusion Matrix:
 [[341  20]
 [ 38 345]]
Classification Report:
               precision    recall  f1-score   support

           0       0.90      0.94      0.92       361
           1       0.95      0.90      0.92       383

    accuracy                           0.92       744
   macro avg       0.92      0.92      0.92       744
weighted avg       0.92      0.92      0.92       744
```
## House Quality Rating Prediction
```
Classification Report:

              precision    recall  f1-score   support

     Average       1.00      1.00      1.00         2
       Basic       1.00      1.00      1.00         2
     Premium       1.00      1.00      1.00         1

    accuracy                           1.00         5
   macro avg       1.00      1.00      1.00         5
weighted avg       1.00      1.00      1.00         5

Accuracy: 1.0

Confusion Matrix:
 [[2 0 0]
 [0 2 0]
 [0 0 1]]

Correct Predictions:

   Square_Footage  Location_Score  Material_Quality Actual_Quality Predicted_Quality
0       -0.592218       -0.387298         -0.426872        Average           Average
1       -1.371451       -1.678293         -1.682377          Basic             Basic
2       -1.482770       -0.387298         -0.426872          Basic             Basic
3       -0.035622       -0.387298          0.200881        Average           Average
4        1.188888        1.549193          1.456386        Premium           Premium

No wrong predictions. Great job!
```
## Vacation Place Rating Prediction
```
--- Vacation Rating Predictor ---
Total budget (₹): 5000
Expected hotel cost (₹): 3000
Travel time (in hours): 6
Feel-good rating (1-10): 3
Weather (1=Cool, 2=Moderate, 3=Hot): 2

 Predicted Rating: 2 Star(s)
 Confidence: 77.43%
```
##  Academic Major Prediction
```
--- Academic Major Predictor ---
Math Score: 90
Biology Score: 95
English Score: 92
History Score: 85
Interest in Science? (1 = Yes, 0 = No): 1
Interest in Art? (1 = Yes, 0 = No): 0

 Predicted Major: Engineering
 Confidence: 96.17%
```
## Students Grade Prediction
```
Enter your marks (0 to 100): 85
Enter your attendance (0 to 100): 92

Predicted Grade: A+
```
## Term 3 Grade and Pass/Fail Prediction
```
Final Predicted Grades and Pass/Fail Summary:

   Name Math_Grade Science_Grade English_Grade Tamil_Grade Social_Grade Physics_Grade  Subjects_Passed  Subjects_Failed
  Alice          D            B+            B+           A            A            A+                6                0
    Bob          C            B+             F           A           B+             C                5                1
Charlie          C             O            B+          B+           B+             A                6                0
  David          F            B+             F           F            F             D                2                4
    Eve          C            B+            B+          B+            F            A+                5                1
  Frank          C            B+             D          B+           B+             C                6                0
  Grace          A             O             A           A            A            A+                6                0
 Hannah          F            B+             F           F            F             F                1                5
    Ivy         B+            B+             D           A            F             C                5                1
   Jack          C            B+             F          B+            D            A+                5                1
```
## Modified Mark and Pass/Fail Prediction
``` Final Predicted Marks, Ranks and Pass/Fail Summary:

   Name  Math_T3  Science_T3  English_T3  Tamil_T3  Social_T3  Physics_T3  Total_Marks  Rank  Subjects_Passed  Subjects_Failed
  Grace       84          95          74        74         74          84          485     1                6                0
  Alice       54          95          84        74         74          84          465     2                6                0
Charlie       54          95          64        64         54          74          405     3                6                0
    Bob       54          64          44        74         64          64          364     4                6                0
  Frank       54          64          44        64         64          54          344     5                6                0
  David       44          64          19        19         19          44          209     0                3                3
    Eve       54          64          64        64         19          64          329     0                5                1
 Hannah       19          64          19        19         19          19          159     0                1                5
    Ivy       64          64          44        74         19          64          329     0                5                1
   Jack       64          64          19        64         54          84          349     0                5                1

 Class Summary : Passed - 5, Failed - 5

Enter student name to view result (or type 'exit' to quit): jack

Result for jack:

Name  Math_T3  Science_T3  English_T3  Tamil_T3  Social_T3  Physics_T3  Total_Marks  Rank  Subjects_Passed  Subjects_Failed
Jack       64          64          19        64         54          84          349     0                5                1

Enter student name to view result (or type 'exit' to quit): Ivy

Result for Ivy:

Name  Math_T3  Science_T3  English_T3  Tamil_T3  Social_T3  Physics_T3  Total_Marks  Rank  Subjects_Passed  Subjects_Failed
 Ivy       64          64          44        74         19          64          329     0                5                1

Enter student name to view result (or type 'exit' to quit): exit
```

# What You Will Learn

* How Logistic Regression works for binary and multi-class classification

* How to use scikit-learn’s built-in datasets and custom data

* How to work with both numerical and text inputs

* How to split datasets into training/testing sets

* How to evaluate a model using:

  * Accuracy

  * Confusion Matrix

  * Classification Report
