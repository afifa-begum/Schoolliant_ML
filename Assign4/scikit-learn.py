# task-1
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print("Dataset Name : Breast Cancer Wisconsin Dataset")
print("Rows :", X.shape[0])
print("Columns :", X.shape[1])
print("Target Column :", data.target_names)
print()

#task-2
print("=" * 60)
print("PROBLEM DESCRIPTION")
print("=" * 60)

print("Goal:")
print("Predict whether the tumor is Malignant or Benign.")
print()

print("Input Features:")
print(list(X.columns))
print()

print("Target Variable:")
print("0 = Malignant")
print("1 = Benign")
print()

#task-3
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("=" * 60)
print("DATA SPLITTING")
print("=" * 60)

print("Training Samples :", len(X_train))
print("Testing Samples :", len(X_test))
print("Ratio Used : 80:20")
print()

#task-4
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(max_iter=1000))
])

#task-5
param_grid = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__solver': ['liblinear', 'lbfgs']
}

grid = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy'
)

grid.fit(X_train, y_train)

print("=" * 60)
print("GRID SEARCH RESULTS")
print("=" * 60)

print("Best Parameters:")
print(grid.best_params_)
print()

print("Best Cross Validation Score:")
print(round(grid.best_score_, 4))
print()
best_model = grid.best_estimator_
predictions = best_model.predict(X_test)

#task-6
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
f1 = f1_score(y_test, predictions)

print("=" * 60)
print("MODEL EVALUATION")
print("=" * 60)

print("Accuracy :", round(accuracy, 4))
print("Precision :", round(precision, 4))
print("Recall :", round(recall, 4))
print("F1 Score :", round(f1, 4))

print()

print("Classification Report")
print(classification_report(y_test, predictions))

print()

print("Confusion Matrix")
cm = confusion_matrix(y_test, predictions)
print(cm)

#task-7
print()
print("=" * 60)
print("RULE BASED BENCHMARK")
print("=" * 60)
threshold = X_train["mean radius"].mean()

rule_predictions = []

for value in X_test["mean radius"]:

    if value > threshold:
        rule_predictions.append(0)
    else:
        rule_predictions.append(1)

rule_predictions = np.array(rule_predictions)

rule_accuracy = accuracy_score(y_test, rule_predictions)
rule_precision = precision_score(y_test, rule_predictions)
rule_recall = recall_score(y_test, rule_predictions)
rule_f1 = f1_score(y_test, rule_predictions)

print("Threshold Used :", round(threshold, 2))
print()

print("Rule Accuracy :", round(rule_accuracy, 4))
print("Rule Precision :", round(rule_precision, 4))
print("Rule Recall :", round(rule_recall, 4))
print("Rule F1 :", round(rule_f1, 4))

'''Comparing:
1.Rule-based model performance
2.ML model performance'''
print()
print("=" * 60)
print("FINAL COMPARISON")
print("=" * 60)

comparison = pd.DataFrame({

    "Metric": [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ],

    "Rule Based": [
        rule_accuracy,
        rule_precision,
        rule_recall,
        rule_f1
    ],

    "Machine Learning": [
        accuracy,
        precision,
        recall,
        f1
    ]

})
print(comparison)
print()
print("=" * 60)
print("SUCCESSFULL")
print("=" * 60)