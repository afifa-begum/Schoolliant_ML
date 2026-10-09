import pandas as pd
import numpy as np

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    GridSearchCV
)

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


# ============================================================
# STEP 6: CROSS-VALIDATION, HYPERPARAMETER TUNING
# AND CLUSTER STABILITY
# ============================================================


# ============================================================
# 1. LOAD DATA
# ============================================================

df = pd.read_csv("cleaned_learner_data.csv")

print("Dataset Shape:", df.shape)


# ============================================================
# 2. PREPARE DATA FOR CLASSIFICATION
# ============================================================

X = df.drop(columns=[
    "student_id",
    "student_name",
    "engagement_tier",
    "completed_course"
])

y = df["engagement_tier"]


# Identify numerical and categorical features
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object"]
).columns.tolist()


# ============================================================
# 3. PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numerical_features),

    ("cat", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])


# ============================================================
# 4. DEFINE CLASSIFICATION MODELS
# ============================================================

models = {

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "KNN": KNeighborsClassifier(),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        random_state=42
    ),

    "SVM": SVC()
}


# ============================================================
# 5. CROSS-VALIDATION
# ============================================================

print("\n" + "=" * 60)
print("CROSS-VALIDATION RESULTS")
print("=" * 60)

cv_results = {}

for name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    scores = cross_val_score(
        pipeline,
        X,
        y,
        cv=5,
        scoring="accuracy"
    )

    cv_results[name] = scores.mean()

    print(
        f"{name:<20} "
        f"Mean Accuracy: {scores.mean():.4f} "
        f"(+/- {scores.std():.4f})"
    )


# ============================================================
# 6. FIND BEST CLASSIFICATION MODEL
# ============================================================

best_model_name = max(
    cv_results,
    key=cv_results.get
)

print("\n" + "=" * 60)
print("BEST MODEL FROM CROSS-VALIDATION")
print("=" * 60)

print("Model:", best_model_name)
print(
    f"Mean CV Accuracy: "
    f"{cv_results[best_model_name]:.4f}"
)


# ============================================================
# 7. HYPERPARAMETER TUNING
# ============================================================

print("\n" + "=" * 60)
print("HYPERPARAMETER TUNING")
print("=" * 60)


# Parameter grids for the models
parameter_grids = {

    "Logistic Regression": {
        "model__C": [0.01, 0.1, 1, 10, 100]
    },

    "KNN": {
        "model__n_neighbors": [3, 5, 7, 9],
        "model__weights": ["uniform", "distance"]
    },

    "Decision Tree": {
        "model__max_depth": [None, 5, 10, 15, 20],
        "model__min_samples_split": [2, 5, 10]
    },

    "Random Forest": {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 10, 20],
        "model__min_samples_split": [2, 5]
    },

    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__kernel": ["linear", "rbf"],
        "model__gamma": ["scale", "auto"]
    }
}


best_model = models[best_model_name]

best_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", best_model)
])


grid_search = GridSearchCV(
    best_pipeline,
    parameter_grids[best_model_name],
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)


grid_search.fit(X, y)


# ============================================================
# 8. DISPLAY BEST PARAMETERS
# ============================================================

print("\nBest Parameters:")

for parameter, value in grid_search.best_params_.items():
    print(f"{parameter}: {value}")


print(
    f"\nBest CV Accuracy: "
    f"{grid_search.best_score_:.4f}"
)


# ============================================================
# 9. CLUSTER STABILITY CHECK
# ============================================================

print("\n" + "=" * 60)
print("CLUSTER STABILITY CHECK")
print("=" * 60)


# Load the clustered dataset created in Step 5
clustered_df = pd.read_csv("learner_segments.csv")


# Features used for clustering
clustering_features = [
    "login_frequency_per_week",
    "avg_session_duration_min",
    "video_completion_pct",
    "quiz_attempts",
    "forum_posts",
    "weekend_activity_pct",
    "consistency_score",
    "weekend_weekday_ratio"
]


X_cluster = clustered_df[clustering_features]


# Scale clustering features
cluster_scaler = StandardScaler()

X_cluster_scaled = cluster_scaler.fit_transform(X_cluster)


# Run K-Means multiple times
cluster_labels = []

for seed in range(10):

    kmeans = KMeans(
        n_clusters=3,
        random_state=seed,
        n_init=10
    )

    labels = kmeans.fit_predict(X_cluster_scaled)

    cluster_labels.append(labels)


# Compare clustering results using Adjusted Rand Index
ari_scores = []

for i in range(len(cluster_labels)):

    for j in range(i + 1, len(cluster_labels)):

        ari = adjusted_rand_score(
            cluster_labels[i],
            cluster_labels[j]
        )

        ari_scores.append(ari)


# Display stability results
print("Number of runs:", len(cluster_labels))

print(
    f"Mean Adjusted Rand Index: "
    f"{np.mean(ari_scores):.4f}"
)

print(
    f"Minimum Adjusted Rand Index: "
    f"{np.min(ari_scores):.4f}"
)

print(
    f"Maximum Adjusted Rand Index: "
    f"{np.max(ari_scores):.4f}"
)


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("STEP 6 COMPLETED SUCCESSFULLY")
print("=" * 60)