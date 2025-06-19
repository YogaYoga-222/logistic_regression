# Logistic Regression Projects

This project includes **Twelve Classification Problems** solved using **Logistic Regression** with the help of `scikit-learn`.

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
- `ordinal_grade_predictor.py`: Predict students grade based on marks and attendance
- `ordinal_marks_predictor.py` : Predict Term 3 subject grades using marks from Term 1 and 2 and show pass/fail summary
- `ordinal_modified_mark_predictor.py`: 

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

### 10. Student Grade Prediction
- **Dataset**: Student marks and attendance
- **Target**: Predict the grade (F to O)
- **Goal**: Predict a student's grade using marks and attendance. If attendance is below 70%, the grade is 'F'. The model uses ordinal logistic regression to do this.

---

### 11. Term 3 Grade and Pass/Fail Prediction
- **Dataset**: Students_marks.csv with Term 1 and 2 marks
- **Task**: Predict Term 3 grades for 6 subjects
- **Goal**: Use ordinal logistic regression to convert marks to grades and count passed/failed subjects
- **Note**: Confirm you loaded the Students_marks.csv in `ordinal_marks_predictor.py`

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
```
 Final Predicted Marks, Ranks and Pass/Fail Summary:

   Name  Math_T3  Science_T3  English_T3  Tamil_T3  Social_T3  Physics_T3  Total_Marks  Rank  Subjects_Passed  Subjects_Failed
  Grace       74          90          79        73         78          85          479     1                6                0
Charlie       53          95          63        67         66          74          418     2                6                0
  Alice       43          65          66        70         76          85          405     3                6                0
  Frank       51          63          42        61         69          58          344     4                6                0
    Bob       54          69          21        72         67          55          338     0                5                1
  David       39          67          36         1         29          47          219     0                2                4
    Eve       54          60          63        66          6          89          338     0                5                1
 Hannah       30          68           8        32         28          37          203     0                1                5
    Ivy       60          68          40        73         20          56          317     0                5                1
   Jack       53          64          23        64         44          86          334     0                5                1

 Class Summary : Passed - 4, Failed - 6
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
