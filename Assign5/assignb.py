import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, confusion_matrix, make_scorer

# Dataset
data = {
    "transaction_amount": [200, 4500, 120, 8000, 90, 15000, 300, 700, 20000, 150],
    "transaction_time": [10, 2, 14, 3, 16, 1, 11, 13, 2, 17],
    "previous_failed_attempts": [0, 3, 0, 4, 0, 5, 1, 0, 6, 0],
    "fraud": [0, 1, 0, 1, 0, 1, 0, 0, 1, 0]
}

df = pd.DataFrame(data)

# 1. Separate features and target
X = df[
    ["transaction_amount",
     "transaction_time",
     "previous_failed_attempts"]
]

y = df["fraud"]

# 2. Create Pipeline
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(class_weight="balanced"))
])

# 3. Custom scorer
def min_precision_recall(y_true, y_pred):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return min(precision, recall)

custom_scorer = make_scorer(min_precision_recall)
param_grid = {
    "model__C": [0.01, 0.1, 1, 10]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    scoring=custom_scorer,
    cv=5
)

grid.fit(X, y)

print("Best C:", grid.best_params_["model__C"])
print("Best Score:", grid.best_score_)

final_model = grid.best_estimator_

y_pred = final_model.predict(X)

precision = precision_score(y, y_pred, zero_division=0)
recall = recall_score(y, y_pred, zero_division=0)

print("\nPrecision:", precision)
print("Recall:", recall)

cm = confusion_matrix(y, y_pred)

print("\nConfusion Matrix:")
print(cm)