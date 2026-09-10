import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV
# Dataset
data = {
    "area_sqft": [750, 900, 1200, 1500, 1800, 2100, 2400, 2700, 3000, 3500],
    "distance_school_km": [1.2, 2.1, 1.0, 3.5, 2.8, 4.0, 3.0, 5.2, 4.5, 6.0],
    "house_price": [45, 52, 65, 72, 85, 95, 110, 125, 140, 165]
}

df = pd.DataFrame(data)

X = df[["area_sqft", "distance_school_km"]]
y = df["house_price"]

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", KNeighborsRegressor())
])

param_grid = {
    "model__n_neighbors": [1, 2, 3, 4, 5]
}

grid = GridSearchCV(
    pipe,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

# Train
grid.fit(X, y)

print("Best Parameter:", grid.best_params_)
print("Best Score:", grid.best_score_)

results = pd.DataFrame(grid.cv_results_)

print("\nGrid Search Results:")
print(results[
    ["param_model__n_neighbors",
     "mean_test_score",
     "std_test_score",
     "rank_test_score"]
])