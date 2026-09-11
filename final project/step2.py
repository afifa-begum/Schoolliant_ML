import pandas as pd

df = pd.read_csv("schoolliant_learner_segmentation (2).csv")
print("Missing values:")
print(df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
df = df.drop_duplicates().copy()
df["forum_posts"] = df["forum_posts"].fillna(
    df["forum_posts"].median()
)
print("Rows after cleaning:", len(df))
print("Missing values after cleaning:")
print(df.isnull().sum())
df.to_csv("cleaned_learner_data.csv", index=False)
