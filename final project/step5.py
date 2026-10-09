import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


df = pd.read_csv("cleaned_learner_data.csv")

print("Dataset Shape:", df.shape)

df["consistency_score"] = (
    df["login_frequency_per_week"] *
    df["video_completion_pct"]
) / 100

df["weekend_weekday_ratio"] = (
    df["weekend_activity_pct"] /
    (100 - df["weekend_activity_pct"] + 0.01)
)

print("\nEngineered Features:")
print(df[
    [
        "consistency_score",
        "weekend_weekday_ratio"
    ]
].head())


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

X = df[clustering_features]

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

inertia = []

for k in range(2, 9):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(X_scaled)

    inertia.append(kmeans.inertia_)


print("\nElbow Method Results:")

for k, value in zip(range(2, 9), inertia):
    print(f"K = {k}, Inertia = {value:.2f}")


plt.figure(figsize=(8, 5))

plt.plot(
    range(2, 9),
    inertia,
    marker="o"
)

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method for Optimal Number of Clusters")

plt.xticks(range(2, 9))
plt.grid(True)

plt.show()


optimal_k = 3

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

df["cluster"] = kmeans.fit_predict(X_scaled)


print("\n" + "=" * 50)
print("CLUSTER DISTRIBUTION")
print("=" * 50)

print(df["cluster"].value_counts().sort_index())


print("\n" + "=" * 50)
print("CLUSTER PROFILES")
print("=" * 50)

cluster_profile = df.groupby("cluster")[clustering_features].mean()

print(cluster_profile.round(2))

df.to_csv(
    "learner_segments.csv",
    index=False
)

print("\nClustered dataset saved as: learner_segments.csv")

print("\n" + "=" * 50)
print("STEP 5 COMPLETED SUCCESSFULLY")
print("=" * 50)