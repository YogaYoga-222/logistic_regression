# Logistic Regression Projects

This project includes **three classification problems** solved using **Logistic Regression** with the help of `scikit-learn`.

Each program is written in its own Python file for better clarity and organization.

---

## Files

- `logistic_regression_iris.py`: Iris flower classification
- `logistic_regression_breast_cancer.py`: Breast cancer prediction
- `logistic_regression_spam_detection.py`: Spam vs. ham message classification 

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

## Required Libraries

Before running the programs, make sure you have the following Python library installed:

```bash
pip install scikit-learn
```

# Run the Programs :

```bash
python3 python logistic_regression_iris.py
```
```bash
python3 python logistic_regression_breast_cancer.py
```
```bash
python3 logistic_regression_spam_detection.py
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

# What You Will Learn

* How Logistic Regression works for binary and multi-class classification

* How to use scikit-learn built-in datasets and text data

* How to split datasets into training/testing sets

* How to evaluate a model using:

  * Accuracy

  * Confusion Matrix

  * Classification Report
