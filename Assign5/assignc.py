import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    PolynomialFeatures,
    OneHotEncoder,
    StandardScaler
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Dataset
data = {
    "study_hours": [1, 2, 3, 4, 5, 6, 7, 8],
    "attendance": [40, 50, 55, 60, 70, 75, 85, 90],
    "learning_mode": [
        "online",
        "offline",
        "online",
        "offline",
        "hybrid",
        "hybrid",
        "offline",
        "online"
    ],
    "passed": [0, 0, 0, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)
X = df[
    ["study_hours", "attendance", "learning_mode"]
]

y = df["passed"]

numerical_columns = [
    "study_hours",
    "attendance"
]

categorical_columns = [
    "learning_mode"
]

numeric_pipeline = Pipeline([
    ("polynomial", PolynomialFeatures(degree=2, include_bias=False)),
    ("scaler", StandardScaler())
])

categorical_pipeline = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numerical_columns),
    ("cat", categorical_pipeline, categorical_columns)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression())
])

model.fit(X, y)

new_student = pd.DataFrame({
    "study_hours": [5],
    "attendance": [72],
    "learning_mode": ["online"]
})

prediction = model.predict(new_student)

if prediction[0] == 1:
    print("The student is predicted to PASS.")
else:
    print("The student is predicted to FAIL.")