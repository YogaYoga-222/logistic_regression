# Logistic Regression Projects

This project includes **two classification problems** solved using **Logistic Regression** with the help of `scikit-learn`.

Each program is written in its own Python file for better clarity and organization.

---

## Files

- `logistic_regression_iris.py`: Iris flower classification
- `logistic_regression_breast_cancer.py`: Breast cancer prediction

---

## Project Descriptions

### 1. Iris Flower Classification

- **Dataset**: Iris dataset (from scikit-learn)
- **Target**: Classify flowers based on petal and sepal measurements
- **Goal**: Predict the species of iris flower (Setosa, Versicolor, Virginica)

### 2. Breast Cancer Classification

- **Dataset**: Breast Cancer dataset (from scikit-learn)
- **Target**: Predict whether a tumor is malignant or benign
- **Goal**: Help in early detection of breast cancer using logistic regression

---

## Required Libraries

Before running the programs, make sure you have the following Python library installed:

```bash
pip install scikit-learn
```

# Run the Programs :

```bash
python logistic_regression_iris.py
python logistic_regression_breast_cancer.py
```

# Sample Output :

## Iris Classification
```bash 
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
lua
Copy
Edit
Accuracy: 0.96
Confusion Matrix:
[[70  2]
 [ 2 42]]
Classification Report:
              precision    recall  f1-score   support

           0       0.97      0.97      0.97        72
           1       0.95      0.95      0.95        44
```
# What You Will Learn

* How Logistic Regression works for classification

* How to use scikit-learn’s built-in datasets

* How to split datasets into training/testing sets

* How to evaluate a model using:

  * Accuracy

  * Confusion Matrix

  * Classification Report
